"""Opt-in GPU integration checks for the real inference backends.

AUTELLIX_GPU_BACKEND=vllm|sglang AUTELLIX_TEST_MODEL=/path/to/model
python -m unittest discover -s tests -p test_gpu_runtime.py -v
"""
import json
import math
import os
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

from autellix.runtime import InferenceEngine, ReplicaConfig, PolicyConfig


BACKEND = os.environ.get("AUTELLIX_GPU_BACKEND")
MODEL = os.environ.get("AUTELLIX_TEST_MODEL")
DEVICE = os.environ.get("AUTELLIX_TEST_DEVICE", "0")
VLLM_MEMORY = float(os.environ.get("AUTELLIX_TEST_VLLM_MEMORY", ".35"))
VLLM_DTYPE = os.environ.get("AUTELLIX_TEST_VLLM_DTYPE", "half")


@unittest.skipUnless(BACKEND and MODEL, "requires AUTELLIX_GPU_BACKEND and AUTELLIX_TEST_MODEL")
class RealInferenceTests(unittest.TestCase):
    def tearDown(self):
        if BACKEND == "vllm":
            import gc
            import torch

            # Direct vLLM runners and their hooks can outlive a test in
            # reference cycles. Release their memory before the next worker.
            gc.collect()
            torch.cuda.empty_cache()

    @unittest.skipUnless(BACKEND == "vllm" and os.environ.get("AUTELLIX_TEST_QWEN3_KERNEL"),
                         "requires AUTELLIX_TEST_QWEN3_KERNEL=1 in the vLLM CUDA environment")
    def test_qwen3_fused_qk_rope_matches_native(self):
        import torch
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding import get_rope
        from integrations.vllm.qwen3_kernels import qk_rms_rope

        torch.manual_seed(9)
        dim, query_heads, kv_heads = 128, 64, 8
        sizes = (query_heads * dim, kv_heads * dim, kv_heads * dim)
        for dtype in (torch.bfloat16, torch.float16):
            q_norm, k_norm = (RMSNorm(dim, eps=1e-6).to(device="cuda", dtype=dtype) for _ in range(2))
            rope = get_rope(dim, dim, 2048, 1_000_000, dtype=dtype).to("cuda")
            with torch.no_grad():
                q_norm.weight.uniform_(.25, 1.5)
                k_norm.weight.uniform_(.25, 1.5)
                for tokens in (1, 8, 64, 2048):
                    qkv = torch.randn(tokens, sum(sizes), device="cuda", dtype=dtype)
                    positions = torch.randperm(2048, device="cuda")[:tokens]
                    q, k, v = qkv.split(sizes, dim=-1)
                    reference_q, reference_k = rope(
                        positions, q_norm(q.reshape(-1, dim).contiguous()).view_as(q),
                        k_norm(k.reshape(-1, dim).contiguous()).view_as(k))
                    packed = qkv.clone()
                    qk_rms_rope(packed, positions, rope.cos_sin_cache, q_norm.weight, k_norm.weight,
                                query_heads, kv_heads, 1e-6)
                    fused_q, fused_k, fused_v = packed.split(sizes, dim=-1)
                    torch.testing.assert_close(fused_q, reference_q, rtol=.008, atol=.008)
                    torch.testing.assert_close(fused_k, reference_k, rtol=.008, atol=.008)
                    self.assertTrue(torch.equal(fused_v, v))

    @unittest.skipUnless(BACKEND == "vllm" and os.environ.get("AUTELLIX_TEST_QWEN3_KERNEL"),
                         "requires the local CUDA Qwen3 model")
    def test_qwen3_reuses_only_executed_prefix_during_concurrent_prefill(self):
        from integrations.vllm.backend import VLLMBackend

        config = PolicyConfig(policy="fcfs", schedule_interval=1, overprovision=0)
        sampling = dict(temperature=0, max_tokens=4, ignore_eos=True)
        with tempfile.TemporaryDirectory(prefix="autellix-prefix-") as directory:
            trace = Path(directory, "schedule.jsonl")
            backend = VLLMBackend(MODEL, str(Path(directory, "state.sqlite")), config,
                                  trace=str(trace),
                                  dtype=VLLM_DTYPE, gpu_memory_utilization=VLLM_MEMORY,
                                  max_num_seqs=2, max_model_len=128, max_num_batched_tokens=32,
                                  block_size=16, num_gpu_blocks_override=8, swap_space=.25, enforce_eager=False,
                                  paper_chunked_prefill=True, refill_idle_slots=True,
                                  qwen3_fused_qk_rope=True)
            try:
                prompt = backend.tokenize("Remember: the answer is Paris.\n" * 10 + "The capital of France is")[:65]
                self.assertEqual(len(prompt), 65)
                for pid in ("first", "second", "serial"):
                    backend.table.open(pid)
                backend.add("a", "first", prompt, sampling)
                backend.step()
                backend.add("b", "second", prompt, sampling)
                next_group = backend.scheduler.waiting[0]
                next_seq = next_group.get_seqs()[0]
                self.assertEqual(backend.scheduler._autellix_paper.prefix_cache.reusable_tokens(next_seq), 32)
                self.assertEqual(backend.scheduler.block_manager.get_num_free_gpu_blocks(), 3)
                outputs = {}
                while backend.has_work():
                    outputs.update((row["request_id"], row["token_ids"]) for row in backend.step() if row["finished"])
                steps = (event for event in map(json.loads, trace.read_text().splitlines())
                         if event["event"] == "plan_step")
                self.assertTrue(any(set(step["requests"]) == {"a", "b"} for step in steps))
                backend.add("c", "serial", prompt, sampling)
                while backend.has_work():
                    outputs.update((row["request_id"], row["token_ids"]) for row in backend.step() if row["finished"])
                self.assertEqual(len(outputs["c"]), 4)
                self.assertEqual(outputs["a"], outputs["c"])
                self.assertEqual(outputs["b"], outputs["c"])
            finally:
                backend.close()

    @unittest.skipUnless(BACKEND == "vllm", "requires vllm")
    def test_chunked_prefill_matches_whole_prefill_with_graph_decode(self):
        tokens = [1 + i % 11 for i in range(84)]
        sampling = dict(temperature=0, max_tokens=8, ignore_eos=True)
        outputs = []
        policy = PolicyConfig(schedule_interval=4, overprovision=0)
        for chunked, budget in ((False, 128), (True, 32)):
            args = dict(max_model_len=512, max_num_batched_tokens=budget, max_num_seqs=1,
                        gpu_memory_utilization=VLLM_MEMORY, swap_space=.25, dtype=VLLM_DTYPE,
                        enforce_eager=False, paper_chunked_prefill=chunked)
            with InferenceEngine([ReplicaConfig("vllm", MODEL, device=DEVICE, engine_args=args)], policy=policy) as engine:
                pid = engine.start_session()
                result = engine.submit(pid, input_ids=tokens, sampling=sampling).result(180)
                outputs.append(result["token_ids"])
                self.assertEqual(len(outputs[-1]), 8)
                steps = [e for e in map(json.loads, (engine.state_dir / "replica-0.jsonl").read_text().splitlines())
                         if e["event"] == "plan_step" and not e["decode"]]
                self.assertEqual([step["tokens"] for step in steps], [32, 32, 20] if chunked else [84])
                engine.end_session(pid)
        self.assertEqual(*outputs)

    @unittest.skipUnless(BACKEND == "vllm", "requires vllm")
    def test_decode_window_uses_last_private_kv_page(self):
        args = dict(max_model_len=64, max_num_batched_tokens=64, max_num_seqs=1,
                    block_size=16, num_gpu_blocks_override=4, gpu_memory_utilization=VLLM_MEMORY,
                    swap_space=.25, dtype=VLLM_DTYPE, paper_chunked_prefill=True)
        config = PolicyConfig(policy="plas", schedule_interval=8, overprovision=0)
        with InferenceEngine([ReplicaConfig("vllm", MODEL, device=DEVICE, engine_args=args)],
                             policy=config, scheduler_trace=False) as engine:
            pid = engine.start_session()
            result = engine.submit(pid, input_ids=[21] * 61,
                                   sampling=dict(temperature=0, max_tokens=3, ignore_eos=True)).result(180)
            self.assertEqual(result["completion_tokens"], 3)
            result = engine.submit(pid, input_ids=[23] * 63,
                                   sampling=dict(temperature=0, max_tokens=4, ignore_eos=True)).result(180)
            self.assertEqual(result["completion_tokens"], 2)
            engine.end_session(pid)

    @unittest.skipUnless(BACKEND == "vllm", "requires vllm")
    def test_preempted_chunked_prefill_restores_initialized_kv(self):
        import torch
        from integrations.vllm.backend import VLLMBackend
        from vllm.sequence import Sequence, SequenceGroup

        with tempfile.TemporaryDirectory(prefix="autellix-prefill-swap-") as directory:
            backend = VLLMBackend(MODEL, str(Path(directory, "state.sqlite")),
                                  PolicyConfig(schedule_interval=4, overprovision=0),
                                  dtype=VLLM_DTYPE, gpu_memory_utilization=VLLM_MEMORY,
                                  max_num_seqs=1, max_model_len=128, max_num_batched_tokens=32,
                                  block_size=16, num_gpu_blocks_override=8, swap_space=.25,
                                  paper_chunked_prefill=True, recompute_on_swap_full=False)
            try:
                backend.table.open("program")
                prompt = [20 + i % 11 for i in range(65)]
                sampling = dict(temperature=0, max_tokens=4, ignore_eos=True)
                backend.add("preempted", "program", prompt, sampling)
                backend.step()
                scheduler = backend.scheduler
                group = scheduler.running[0]
                self.assertEqual(group.get_seqs()[0].data.get_num_computed_tokens(), 32)
                outgoing = []
                scheduler._preempt_by_swap(group, outgoing)
                scheduler.running.remove(group)
                scheduler.swapped.append(group)
                self.assertEqual(len(outgoing), 2)
                worker = backend.engine.model_executor.driver_worker
                worker.cache_engine[0].swap_out(torch.tensor(outgoing, dtype=torch.int64))

                manager = scheduler.block_manager

                def clear_gpu_prefix(token):
                    other = Sequence(900, {"prompt_token_ids": [token] * 128}, 16)
                    manager.allocate(SequenceGroup("other", [other], 0.))
                    manager.free(other)
                    for layer in worker.cache_engine[0].gpu_cache:
                        layer.zero_()

                # Reuse every GPU page and overwrite its contents. The paused
                # call has to get both pages back from the pinned host cache.
                clear_gpu_prefix(40)

                def finish():
                    while backend.has_work():
                        for result in backend.step():
                            if result["finished"]:
                                return result["token_ids"]
                    self.fail("no final output")

                restored = finish()
                clear_gpu_prefix(41)
                backend.add("uninterrupted", "program", prompt, sampling)
                self.assertEqual(restored, finish())
                self.assertEqual(len(restored), 4)
            finally:
                backend.close()

    @unittest.skipUnless(BACKEND == "vllm", "requires vllm")
    def test_full_prefill_chunk_preserves_resident_decode(self):
        args = dict(max_model_len=512, max_num_batched_tokens=32, max_num_seqs=2,
                    gpu_memory_utilization=VLLM_MEMORY, swap_space=.25, dtype=VLLM_DTYPE,
                    paper_chunked_prefill=True, keep_resident_kv=True)
        config = PolicyConfig(policy="plas", boundaries=(0, 1e-8, math.inf), quanta=(1e6, 1e6),
                              schedule_interval=4, overprovision=0, beta=1e12)
        sampling = dict(temperature=0, ignore_eos=True)
        with InferenceEngine([ReplicaConfig("vllm", MODEL, device=DEVICE, engine_args=args)], policy=config) as engine:
            resident_pid, new_pid = engine.start_session(), engine.start_session()
            engine.submit(resident_pid, input_ids=[21], sampling=dict(sampling, max_tokens=1)).result(180)
            resident = engine.submit(resident_pid, input_ids=[25] * 16, sampling=dict(sampling, max_tokens=256))
            resident.progress.get(timeout=180)
            new = engine.submit(new_pid, input_ids=[37] * 84, sampling=dict(sampling, max_tokens=4))
            resident_result, new_result = resident.result(180), new.result(180)
            self.assertEqual([resident_result["completion_tokens"], new_result["completion_tokens"]], [256, 4])
            old_rid, new_rid = resident_result["request_id"], new_result["request_id"]
            events = list(map(json.loads, (engine.state_dir / "replica-0.jsonl").read_text().splitlines()))
            start = next(i for i, event in enumerate(events) if event["event"] == "schedule_window"
                         and event["requests"] == [new_rid, old_rid])
            stop = next((i for i in range(start + 1, len(events)) if events[i]["event"] == "schedule_window"), len(events))
            steps = [e for e in events[start:stop] if e["event"] == "plan_step"]
            self.assertEqual([e["tokens"] for e in steps[:2]], [32, 32])
            self.assertEqual(steps[2]["requests"], [new_rid, old_rid])
            self.assertTrue(steps[2]["decode"])
            for pid in (resident_pid, new_pid):
                engine.end_session(pid)

    @unittest.skipUnless(BACKEND == "vllm", "requires vllm")
    def test_oversized_prefill_does_not_kill_replica(self):
        args = dict(max_model_len=512, max_num_batched_tokens=32, max_num_seqs=1,
                    gpu_memory_utilization=VLLM_MEMORY, swap_space=.25, dtype=VLLM_DTYPE)
        with InferenceEngine([ReplicaConfig("vllm", MODEL, device=DEVICE, engine_args=args)]) as engine:
            bad_pid, good_pid = engine.start_session(), engine.start_session()
            bad = engine.submit(bad_pid, input_ids=[1] * 64, sampling=dict(max_tokens=8))
            good = engine.submit(good_pid, input_ids=[1] * 8,
                                 sampling=dict(temperature=0, max_tokens=8, ignore_eos=True))
            with self.assertRaisesRegex(RuntimeError, "whole-prefill token budget"):
                bad.result(180)
            self.assertEqual(len(good.result(180)["token_ids"]), 8)
            self.assertEqual(engine.failed, {})
            engine.end_session(bad_pid)
            engine.end_session(good_pid)
            engine.wait_idle(30)
            self.assertEqual(engine.table.describe(), {})

    @unittest.skipUnless(BACKEND == "vllm" and os.environ.get("AUTELLIX_TEST_TP"), "requires vllm and AUTELLIX_TEST_TP=1 with two GPUs")
    def test_tensor_parallel_generation_and_program_accounting(self):
        import torch
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        devices = os.environ.get("AUTELLIX_TEST_TP_DEVICES", "0,1")
        args = dict(tensor_parallel_size=2, max_model_len=512, max_num_seqs=1,
                    gpu_memory_utilization=.35, swap_space=.25, dtype="half",
                    batched_swap=os.environ.get("AUTELLIX_TEST_TP_NATIVE_SWAP") != "1")
        policy = PolicyConfig(boundaries=(0, .001, .002, math.inf), quanta=(.0001, .0002, .0004),
                              schedule_interval=1, overprovision=0, beta=1e12)
        with InferenceEngine([ReplicaConfig("vllm", MODEL, device=devices, engine_args=args)],
                             policy=policy) as engine:
            pid = engine.start_session()
            tokens = engine.tokenize(prompt="The capital of France is")
            sampling = dict(temperature=0, max_tokens=24, ignore_eos=True)
            a = engine.submit(pid, input_ids=tokens, sampling=sampling)
            other = engine.start_session()
            b = engine.submit(other, input_ids=tokens, sampling=dict(temperature=0, max_tokens=8))
            result, _ = a.result(180), b.result(180)
            reference = engine.submit(pid, input_ids=tokens, sampling=sampling).result(180)
            self.assertEqual(result["token_ids"], reference["token_ids"])
            self.assertGreater(result["metrics"]["executed"], 0)
            with (engine.state_dir / "replica-0.jsonl").open(encoding="utf-8") as trace:
                events = Counter(json.loads(line)["event"] for line in trace)
            self.assertGreater(events["swap_out"], 0)
            self.assertGreater(events["resume"], 0)
            engine.end_session(pid)
            engine.end_session(other)
            engine.wait_idle(30)
            self.assertEqual(engine.table.describe(), {})

    def test_real_generation_preemption_and_session_inheritance(self):
        recompute = BACKEND == "vllm" and os.environ.get("AUTELLIX_TEST_RECOMPUTE") == "1"
        if BACKEND == "vllm":
            args = dict(max_model_len=512, max_num_seqs=1, gpu_memory_utilization=VLLM_MEMORY,
                        swap_space=.25, dtype=VLLM_DTYPE, enforce_eager=True,
                        fused_swap=os.environ.get("AUTELLIX_TEST_FUSED_SWAP") == "1")
            if recompute:
                args.update(swap_space=0, recompute_on_swap_full=True)
        else:
            args = dict(context_length=512, max_running_requests=1, mem_fraction_static=.35,
                        dtype="float16", random_seed=0)
            if os.environ.get("AUTELLIX_TEST_LEGACY_GPU") == "1":
                args.update(autellix_legacy_gpu=True, max_total_tokens=2048, dtype="float32")
        # Very short slices deliberately force real preemption with a tiny model.
        policy = PolicyConfig(boundaries=(0, .001, .002, math.inf),
                              quanta=(.0001, .0002, .0004), beta=1e12,
                              schedule_interval=int(os.environ.get("AUTELLIX_TEST_STEPS", "1")),
                              overprovision=int(os.environ.get("AUTELLIX_TEST_RESERVE", "0")),
                              implementation=os.environ.get("AUTELLIX_TEST_IMPLEMENTATION", "paper"))
        with tempfile.TemporaryDirectory(prefix="autellix-gpu-") as directory:
            engine = InferenceEngine([ReplicaConfig(BACKEND, MODEL, device=DEVICE, engine_args=args)],
                                     policy=policy, state_dir=directory)
            try:
                engine.wait_ready(600)
                first = engine.start_session("long")
                second = engine.start_session("short")
                tokens = engine.tokenize(prompt="The capital of France is")
                sampling = dict(temperature=0, max_tokens=24, ignore_eos=True)
                a = engine.submit(first, input_ids=tokens, sampling=sampling)
                b = engine.submit(second, input_ids=tokens, sampling=dict(temperature=0, max_tokens=8, ignore_eos=True))
                ra, rb = a.result(180), b.result(180)
                self.assertEqual(ra.get("completion_tokens", len(ra["token_ids"])), 24)
                self.assertEqual(rb.get("completion_tokens", len(rb["token_ids"])), 8)
                self.assertGreater(ra["metrics"]["executed"], 0)
                follow = engine.submit(first, input_ids=tokens, sampling=dict(temperature=0, max_tokens=4, ignore_eos=True)).result(120)
                self.assertGreater(follow["metrics"]["inherited"], 0)
                events = [json.loads(line) for line in Path(directory, "replica-0.jsonl").read_text().splitlines()]
                counts = Counter(event["event"] for event in events)
                self.assertGreater(counts["demote"], 0)
                if policy.implementation == "paper" and policy.schedule_interval > 1:
                    self.assertGreater(counts["continue_plan"], 0)
                    self.assertLess(counts["schedule_window"], counts["execute"])
                if recompute:
                    self.assertGreater(counts["recompute_retract"], 0)
                    self.assertFalse(counts["swap_out"])
                elif policy.overprovision and BACKEND == "vllm":
                    self.assertGreater(counts["reserve"], 0)
                    if policy.schedule_interval > 1:
                        self.assertGreater(counts["refill"], 0)
                else:
                    self.assertGreater(counts["swap_out" if BACKEND == "vllm" else "retract"], 0)
                if policy.overprovision and BACKEND == "sglang":
                    self.assertGreater(counts["reserve"], 0)
                    self.assertGreater(counts["resume_cached"], 0)
                    if policy.schedule_interval > 1:
                        self.assertGreater(counts["refill"], 0)
                if BACKEND == "sglang" and not policy.overprovision:
                    self.assertGreater(counts["swap_out"], 0)
                    self.assertGreater(counts["swap_in"], 0)
                # Compare interrupted greedy generation against uninterrupted
                # generation in the SAME loaded engine and prompt, not a mock.
                reference_pid = engine.start_session("reference")
                reference = engine.submit(reference_pid, input_ids=tokens, sampling=sampling).result(180)
                self.assertEqual(ra["text"], reference["text"])
                if ra["token_ids"]:
                    self.assertEqual(ra["token_ids"], reference["token_ids"])
                for pid in (first, second, reference_pid):
                    engine.end_session(pid)
                self.assertEqual(engine.table.snapshot(), {})
                print(json.dumps({"backend": BACKEND, "text": ra["text"], "trace_events": dict(counts),
                                  "generated_tokens": 24, "greedy_resume_matches": True}))
            finally:
                engine.shutdown()

    @unittest.skipUnless(os.environ.get("AUTELLIX_TEST_REPLICAS"), "requires AUTELLIX_TEST_REPLICAS=1")
    def test_two_real_replicas_cancel_and_cleanup(self):
        args = (dict(max_model_len=512, max_num_seqs=1, gpu_memory_utilization=.2,
                     swap_space=.1, dtype=VLLM_DTYPE, enforce_eager=True) if BACKEND == "vllm" else
                dict(context_length=512, max_running_requests=1, mem_fraction_static=.2,
                     dtype="float16", random_seed=0))
        if BACKEND == "sglang" and os.environ.get("AUTELLIX_TEST_LEGACY_GPU") == "1":
            args.update(autellix_legacy_gpu=True, max_total_tokens=2048, dtype="float32")
        with tempfile.TemporaryDirectory(prefix="autellix-replicas-") as directory:
            devices = os.environ.get("AUTELLIX_TEST_DEVICES", "0,0").split(",")
            if BACKEND == "vllm" and len(set(devices)) < len(devices):
                # Each vLLM worker profiles the same card. Fix its cache size
                # so peer allocations cannot change its startup result.
                args["num_gpu_blocks_override"] = 64
            engine = InferenceEngine([ReplicaConfig(BACKEND, MODEL, device=device, engine_args=args) for device in devices],
                                     state_dir=directory, locality_threshold=10)
            try:
                engine.wait_ready(600)
                pid = engine.start_session()
                tokens = engine.tokenize(prompt="The capital of France is")
                a = engine.submit(pid, input_ids=tokens, sampling=dict(temperature=0, max_tokens=64, ignore_eos=True))
                b = engine.submit(pid, input_ids=tokens, sampling=dict(temperature=0, max_tokens=8, ignore_eos=True))
                ra, rb = a.result(180), b.result(180)
                self.assertNotEqual(ra["engine_id"], rb["engine_id"])
                long_tokens = tokens * 3
                first = engine.submit(pid, input_ids=long_tokens, sampling=dict(temperature=0, max_tokens=8)).result(120)
                second = engine.submit(pid, input_ids=long_tokens, sampling=dict(temperature=0, max_tokens=8)).result(120)
                self.assertEqual(first["engine_id"], second["engine_id"])
                pending = engine.submit(pid, input_ids=tokens, sampling=dict(temperature=0, max_tokens=256, ignore_eos=True))
                self.assertTrue(pending.cancel())
                engine.end_session(pid)
                engine.wait_idle(30)
                self.assertFalse(engine.pending)
                self.assertEqual(engine.loads, [0, 0])
                self.assertEqual(engine.table.snapshot(), {})
                print(json.dumps({"backend": BACKEND, "real_replicas": 2,
                                  "short_engines": [ra["engine_id"], rb["engine_id"]],
                                  "long_engine": first["engine_id"], "cancel_cleaned": True}))
            finally:
                engine.shutdown()


@unittest.skipUnless(os.environ.get("AUTELLIX_TEST_CUDA_SWAP"), "requires AUTELLIX_TEST_CUDA_SWAP=1")
class CUDASwapTests(unittest.TestCase):
    @unittest.skipIf(BACKEND == "vllm", "SGLang attention check uses the SGLang PyTorch environment")
    def test_tiled_prefix_attention_cuda(self):
        import torch
        from integrations.sglang.attention import prefix_attention
        torch.manual_seed(19)
        for dtype in (torch.float16, torch.float32):
            total, count, heads = 4096, 1024, 32
            q = torch.randn(count, heads, 16, device="cuda", dtype=dtype)
            k, v = [torch.randn(total, 8, 16, device="cuda", dtype=dtype) for _ in range(2)]
            mask = torch.arange(total, device="cuda")[None] <= total-count+torch.arange(count, device="cuda")[:, None]
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                attn_mask=mask, enable_gqa=True)[0].transpose(0, 1)
            output = torch.empty_like(q)
            prefix_attention(None, q, output, k, v, torch.arange(total, device="cuda")[None],
                             torch.tensor([0], device="cuda"), torch.tensor([total], device="cuda"),
                             torch.tensor([total-count], device="cuda"), torch.tensor([count], device="cuda"),
                             enable_gqa=True, causal=True)
            torch.testing.assert_close(output, expected, atol=1e-3 if dtype==torch.float16 else 1e-5, rtol=1e-3)

    def test_real_kv_roundtrip_nontrivial_mapping(self):
        import torch
        from unittest.mock import patch
        from autellix.runtime.packed_kv import PackedKVLayout
        from autellix.runtime.swap import batched_swap, packed_host_swap
        self.assertTrue(torch.cuda.is_available())
        source = [torch.randn(2, 9, 4, 2, 8, dtype=torch.float16, device="cuda") for _ in range(3)]
        host = [torch.full((2, 11, 4, 2, 8), -10., dtype=torch.float16) for _ in source]
        target = [torch.zeros_like(layer) for layer in source]
        block_bytes = len(source) * source[0][:, 0].numel() * source[0].element_size()
        swap_out = [(0, 2), (7, 2), (0, 9), (4, 5), (4, 5)]
        swap_in = [(9, 7), (2, 7), (9, 0), (5, 4)]
        with patch("autellix.runtime.swap.MAX_TRANSFER_BYTES", block_bytes):
            batched_swap(source, host, swap_out)
            batched_swap(host, target, swap_in)
        for src, dst, cpu in zip(source, target, host):
            self.assertTrue(torch.equal(src[:, [7, 0, 4]], dst[:, [7, 0, 4]]))
            self.assertTrue(torch.equal(dst[:, 1], torch.zeros_like(dst[:, 1])))
            self.assertTrue(torch.all(cpu[:, 0] == -10).item())
        with self.assertRaises(IndexError):
            batched_swap(source, host, [(10, 0), (7, 0)])

        packer = PackedKVLayout(source, channels=2, token_major=True)
        unpacker = PackedKVLayout(target, channels=2, token_major=True)
        blocks = torch.empty(packer.shape(11), dtype=packer.dtype, pin_memory=True)
        for transfer_bytes, copies in ((block_bytes, 3), (3 * block_bytes, 1)):
            blocks.fill_(-10)
            for layer in target:
                layer.zero_()
            with patch("autellix.runtime.swap.MAX_TRANSFER_BYTES", transfer_bytes):
                self.assertEqual(packed_host_swap(packer, blocks, swap_out, to_host=True), copies)
                self.assertEqual(packed_host_swap(unpacker, blocks, swap_in, to_host=False), copies)
            for src, dst in zip(source, target):
                self.assertTrue(torch.equal(src[:, [7, 0, 4]], dst[:, [7, 0, 4]]))
                self.assertEqual(dst[:, 1].count_nonzero().item(), 0)
            self.assertTrue(torch.all(blocks[0] == -10).item())
        with self.assertRaises(IndexError):
            packed_host_swap(packer, blocks, [(10, 0), (7, 0)], to_host=True)

    def test_fused_kv_matches_selected_blocks_for_both_layouts(self):
        import torch
        from autellix.runtime.packed_kv import PackedKVLayout

        indices = torch.tensor([7, 0, 4], device="cuda")
        for channels in (1, 2):
            prefix = (2,) if channels == 2 else ()
            shape = (*prefix, 9, 3, 17)
            source = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            target = [torch.zeros_like(layer) for layer in source]
            packer = PackedKVLayout(source, channels=channels, token_major=channels == 1)
            unpacker = PackedKVLayout(target, channels=channels, token_major=channels == 1)
            payload = torch.empty(packer.shape(len(indices)), device="cuda", dtype=torch.bfloat16)
            packer.move(indices, payload, pack=True)
            unpacker.move(indices, payload, pack=False)
            axis = channels - 1
            for original, restored in zip(source, target):
                self.assertTrue(torch.equal(original.index_select(axis, indices), restored.index_select(axis, indices)))
                self.assertEqual(restored.select(axis, 1).count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
