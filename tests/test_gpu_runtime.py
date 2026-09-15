"""Opt-in integration checks; never report these as passed when skipped.

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


@unittest.skipUnless(BACKEND and MODEL, "requires AUTELLIX_GPU_BACKEND and AUTELLIX_TEST_MODEL")
class RealInferenceTests(unittest.TestCase):
    def test_real_generation_preemption_and_session_inheritance(self):
        if BACKEND == "vllm":
            args = dict(max_model_len=512, max_num_seqs=1, gpu_memory_utilization=.35,
                        swap_space=.25, dtype="half", enforce_eager=True)
        else:
            args = dict(context_length=512, max_running_requests=1, mem_fraction_static=.35,
                        dtype="float16", random_seed=0)
        # Very short slices deliberately force real preemption with a tiny model.
        policy = PolicyConfig(boundaries=(0, .001, .002, math.inf),
                              quanta=(.0001, .0002, .0004), beta=1e12,
                              schedule_interval=int(os.environ.get("AUTELLIX_TEST_STEPS", "1")),
                              overprovision=int(os.environ.get("AUTELLIX_TEST_RESERVE", "0")))
        with tempfile.TemporaryDirectory(prefix="autellix-gpu-") as directory:
            engine = InferenceEngine([ReplicaConfig(BACKEND, MODEL, engine_args=args)],
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
                if policy.overprovision and BACKEND == "vllm":
                    self.assertGreater(counts["reserve"], 0)
                else:
                    self.assertGreater(counts["swap_out" if BACKEND == "vllm" else "retract"], 0)
                if policy.overprovision and BACKEND == "sglang":
                    self.assertGreater(counts["reserve"], 0)
                    self.assertGreater(counts["resume_cached"], 0)
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
                     swap_space=.1, dtype="half", enforce_eager=True) if BACKEND == "vllm" else
                dict(context_length=512, max_running_requests=1, mem_fraction_static=.2,
                     dtype="float16", random_seed=0))
        with tempfile.TemporaryDirectory(prefix="autellix-replicas-") as directory:
            engine = InferenceEngine([ReplicaConfig(BACKEND, MODEL, engine_args=args) for _ in range(2)],
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
    def test_real_kv_roundtrip_nontrivial_mapping(self):
        import torch
        from autellix.runtime.swap import batched_swap
        self.assertTrue(torch.cuda.is_available())
        source = [torch.randn(2, 9, 4, 2, 8, dtype=torch.float16, device="cuda") for _ in range(3)]
        host = [torch.full((2, 11, 4, 2, 8), -10., dtype=torch.float16) for _ in source]
        target = [torch.zeros_like(layer) for layer in source]
        batched_swap(source, host, [(7, 2), (0, 9), (4, 5)])
        batched_swap(host, target, [(2, 7), (9, 0), (5, 4)])
        for src, dst, cpu in zip(source, target, host):
            self.assertTrue(torch.equal(src[:, [7, 0, 4]], dst[:, [7, 0, 4]]))
            self.assertTrue(torch.equal(dst[:, 1], torch.zeros_like(dst[:, 1])))
            self.assertTrue(torch.all(cpu[:, 0] == -10).item())
        with self.assertRaises(IndexError):
            batched_swap(source, host, [(10, 0)])


if __name__ == "__main__":
    unittest.main()
