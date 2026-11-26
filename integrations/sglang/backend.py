from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from functools import partial
from importlib.metadata import version

from autellix.runtime.policy import PolicyConfig, ProgramTable
from .scheduler import encode_request, run_scheduler_process


class SGLangBackend:
    """SGLang 0.4.9.post6 with hooks installed in its spawned GPU scheduler."""

    def __init__(self, model, table_path, config=None, trace=None, completion_only=False, **engine_args):
        if version("sglang") != "0.4.9.post6":
            raise RuntimeError("this integration requires sglang==0.4.9.post6")
        config = config or PolicyConfig()
        self.completion_only = completion_only
        self.native = engine_args.pop("autellix_native_baseline", False)
        native_defaults = engine_args.pop("autellix_native_defaults", False)
        pp = engine_args.get("pp_size", 1)
        if self.native and config.overprovision:
            raise ValueError("native SGLang baseline requires overprovision=0")
        legacy_gpu = engine_args.pop("autellix_legacy_gpu", False)
        if native_defaults and (not self.native or legacy_gpu):
            raise ValueError("upstream SGLang defaults require native mode on a modern GPU")
        prefix_attention = engine_args.pop("autellix_prefix_attention", True)
        score_elements = engine_args.pop("autellix_attention_score_elements", 2**26)
        mlp_chunk_size = engine_args.pop("autellix_mlp_chunk_size", 0)
        retain_running = engine_args.pop("autellix_retain_running", True)
        prefill_frontier = engine_args.pop("autellix_prefill_frontier", False)
        partial_retention = engine_args.pop("autellix_partial_retention", False)
        if partial_retention and (pp != 1 or config.implementation != "paper" or not retain_running):
            raise ValueError("autellix_partial_retention requires single-stage paper mode with decode retention")
        cache_paused_kv = engine_args.pop("autellix_cache_paused_kv", False)
        publish_activity_at_window = engine_args.pop("autellix_window_activity", False)
        publish_activity = engine_args.pop("autellix_publish_activity", True)
        refill_idle_slots = engine_args.pop("autellix_refill_idle_slots", False)
        resident_reserve_only = engine_args.pop("autellix_resident_reserve_only", False)
        mix_prefill_decode = engine_args.pop("autellix_mix_prefill_decode", True)
        mix_prefill_max_tokens = engine_args.pop("autellix_mix_prefill_max_tokens", 0)
        if prefill_frontier and (self.native or pp != 1 or legacy_gpu or config.implementation != "paper"
                                 or not retain_running or (mix_prefill_decode and not mix_prefill_max_tokens)
                                 or (engine_args.get("chunked_prefill_size") or 0) <= 0):
            raise ValueError("autellix_prefill_frontier requires single-stage paper mode with split chunked admission")
        if type(mix_prefill_max_tokens) is not int or mix_prefill_max_tokens < 0:
            raise ValueError("autellix_mix_prefill_max_tokens must be a nonnegative integer")
        if mix_prefill_max_tokens and (self.native or pp != 1 or legacy_gpu or config.implementation != "paper"
                                      or (engine_args.get("chunked_prefill_size") or 0) <= 0):
            raise ValueError("autellix_mix_prefill_max_tokens requires modern single-stage paper mode with chunked prefill")
        fill_chunk_tail = engine_args.pop("autellix_fill_chunk_tail", False)
        decode_cached_resumes = engine_args.pop("autellix_decode_cached_resumes", False)
        min_decode_steps = engine_args.pop("autellix_min_decode_steps", 0)
        if (type(min_decode_steps) is not int or min_decode_steps < 0
                or (min_decode_steps and (pp != 1 or config.implementation != "paper"))):
            raise ValueError("autellix_min_decode_steps requires single-stage paper mode and a nonnegative integer")
        defer_model_timing = engine_args.pop("autellix_defer_model_timing", True)
        fused_decode_metadata = engine_args.pop("autellix_fused_decode_metadata", False)
        if fused_decode_metadata and (self.native or pp != 1 or legacy_gpu or config.implementation != "paper"):
            raise ValueError("autellix_fused_decode_metadata requires modern single-stage paper mode")
        stable_decode_overlap = engine_args.pop("autellix_stable_decode_overlap", False)
        if stable_decode_overlap and (self.native or pp != 1 or legacy_gpu or config.implementation != "paper"
                                      or engine_args.get("disaggregation_mode", "null") != "null"):
            raise ValueError("autellix_stable_decode_overlap requires modern single-stage paper mode without disaggregation")
        pipeline_timing_metadata = engine_args.pop("autellix_pipeline_timing_metadata", True)
        fused_swap = engine_args.pop("autellix_fused_swap", False)
        measure_timing = engine_args.pop("autellix_measure_timing", False)
        if legacy_gpu:
            # Explicit compatibility mode: native PyTorch ops, no CUDA JIT.
            engine_args.setdefault("dtype", "float32")
            engine_args.update(attention_backend="torch_native", sampling_backend="pytorch",
                               disable_cuda_graph=True,
                               disable_custom_all_reduce=True)
        for key in ("tp_size", "dp_size"):
            if engine_args.get(key, 1) != 1 and (not self.native or key == "dp_size"):
                raise ValueError(f"{key} must be 1; use independent replicas")
        incompatible = ("speculative_algorithm", "enable_hierarchical_cache", "disable_radix_cache",
                        "enable_lora", "enable_dp_attention")
        if any(engine_args.get(k) for k in incompatible):
            raise ValueError("speculation, HiCache, LoRA, DP attention, and disabled radix cache are unsupported")
        from sglang.srt.entrypoints import engine as entry
        host_bytes = int(engine_args.pop("autellix_swap_space", 1) * 1024**3)
        host_kv_recompute = engine_args.pop("autellix_host_kv_recompute", False)
        if host_bytes <= 0:
            raise ValueError("autellix_swap_space must be positive GiB")
        if host_kv_recompute and pp > 1:
            raise ValueError("host KV overflow recomputation requires pp_size=1")
        options = dict(table_path=table_path, config=asdict(config), trace=trace,
                       host_bytes=host_bytes, host_kv_recompute=host_kv_recompute,
                       legacy_gpu=legacy_gpu, pipeline=pp > 1, prefix_attention=prefix_attention,
                       retain_running=retain_running, partial_retention=partial_retention,
                       prefill_frontier=prefill_frontier,
                       cache_paused_kv=cache_paused_kv and pp == 1,
                       publish_activity_at_window=publish_activity_at_window and config.implementation == "paper",
                       publish_activity=publish_activity,
                       refill_idle_slots=refill_idle_slots and config.implementation == "paper",
                       resident_reserve_only=resident_reserve_only and config.implementation == "paper",
                       mix_prefill_decode=mix_prefill_decode,
                       mix_prefill_max_tokens=mix_prefill_max_tokens,
                       fill_chunk_tail=fill_chunk_tail,
                       decode_cached_resumes=decode_cached_resumes and not legacy_gpu,
                       min_decode_steps=min_decode_steps,
                       defer_model_timing=defer_model_timing,
                       fused_decode_metadata=fused_decode_metadata,
                       stable_decode_overlap=stable_decode_overlap,
                       pipeline_timing_metadata=pipeline_timing_metadata,
                       fused_swap=fused_swap,
                       measure_timing=measure_timing,
                       attention_score_elements=score_elements,
                       mlp_chunk_size=mlp_chunk_size)
        target = scheduler_target = entry.run_scheduler_process
        if self.native:
            if legacy_gpu or pp > 1:
                from .legacy_gpu import run_native_scheduler_process
                scheduler_target = partial(run_native_scheduler_process, prefix_attention=prefix_attention,
                                           attention_score_elements=score_elements,
                                           mlp_chunk_size=mlp_chunk_size, legacy_gpu=legacy_gpu,
                                           pipeline=pp > 1)
        else:
            scheduler_target = partial(run_scheduler_process, autellix_options=options)
        args = dict(max_running_requests=8)
        if not native_defaults:
            args.update(disable_cuda_graph=True, schedule_policy="fcfs", attention_backend="triton")
        args.update(engine_args)
        if not native_defaults:
            args.update(disable_overlap_schedule=not stable_decode_overlap, page_size=1)
            args.setdefault("chunked_prefill_size", -1)
            if (args["chunked_prefill_size"] or 0) > 0 and (pp > 1 or legacy_gpu or config.implementation != "paper"):
                raise ValueError("Autellix chunked prefill requires a modern single-GPU paper scheduler")
        options["capacity"] = args["max_running_requests"]
        if pp > 1:
            if options["capacity"] % pp:
                raise ValueError("max_running_requests must be divisible by pp_size")
            args["max_micro_batch_size"] = options["capacity"] // pp + config.overprovision
        if config.overprovision:
            args["max_running_requests"] += config.overprovision * pp
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.table = ProgramTable(table_path)
        try:
            entry.run_scheduler_process = scheduler_target
            self.engine = entry.Engine(model_path=model, **args)
        finally:
            entry.run_scheduler_process = target
        self.pending = {}
        self.programs = {}
        self.updates = []

    def tokenize(self, prompt):
        return self.engine.tokenizer_manager.tokenizer.encode(prompt)

    def add(self, rid, pid, prompt, sampling):
        from sglang.srt.managers.io_struct import GenerateReqInput
        sampling = dict(sampling)
        if sampling.pop("n", 1) != 1:
            raise ValueError("one sequence per request is required")
        if "max_tokens" in sampling:
            sampling["max_new_tokens"] = sampling.pop("max_tokens")
        internal = encode_request(pid, rid)
        inputs = {"input_ids": prompt} if isinstance(prompt, list) else {"text": prompt}
        request = GenerateReqInput(rid=internal, sampling_params=sampling, stream=not self.completion_only, **inputs)

        async def generate():
            gen = self.engine.tokenizer_manager.generate_request(request, None)
            try:
                result = None
                async for part in gen:
                    result = part
                    if not self.completion_only:
                        self.updates.append({"request_id": rid, "finished": False, "text": part.get("text", "")})
                if result is None:
                    raise RuntimeError("SGLang returned no output")
                # Tokenizer output and GPU completion accounting cross separate IPC paths.
                metrics = {"native_sglang": True} if self.native else None
                for _ in range(0 if self.native else 1000):
                    metrics = self.table.take_metrics(rid)
                    if metrics is not None:
                        break
                    await asyncio.sleep(.001)
                if metrics is None:
                    raise RuntimeError("scheduler did not acknowledge completion accounting")
                meta = result.get("meta_info", {})
                return {"request_id": rid, "text": result.get("text", ""),
                        "token_ids": result.get("output_ids", []),
                        "completion_tokens": meta.get("completion_tokens", 0),
                        "prompt_tokens": meta.get("prompt_tokens", 0),
                        "sglang_cached_tokens": meta.get("cached_tokens"),
                        "finish_reason": meta.get("finish_reason", {}), "metrics": metrics}
            finally:
                await gen.aclose()
        if rid in self.pending:
            raise ValueError("duplicate request ID")
        self.programs[rid] = (pid, internal)
        self.pending[rid] = self.loop.create_task(generate())

    def step(self):
        self.loop.run_until_complete(asyncio.sleep(.001))
        results, self.updates = self.updates, []
        for rid, task in list(self.pending.items()):
            if task.done():
                del self.pending[rid]
                self.programs.pop(rid, None)
                try:
                    results.append(task.result())
                except Exception as exc:
                    results.append({"request_id": rid, "error": repr(exc)})
        return results

    def cancel(self, rid):
        task = self.pending.pop(rid, None)
        program = self.programs.pop(rid, None)
        if program:
            pid, internal = program
            self.engine.tokenizer_manager.abort_request(internal)
        if task:
            task.cancel()
            self.loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        if program and not self.native:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.table.take_metrics(rid) is not None:
                    break
                self.loop.run_until_complete(asyncio.sleep(.005))
            # Idempotent fallback for a cancellation before GPU admission.
            self.table.finish(pid, internal, 0, 0, 0, "atlas")

    def has_work(self):
        return bool(self.pending)

    def close(self):
        for rid in list(self.pending):
            self.cancel(rid)
        self.engine.shutdown()
        self.loop.close()
        self.table.close()
