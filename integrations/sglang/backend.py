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

    def __init__(self, model, table_path, config=None, trace=None, **engine_args):
        if version("sglang") != "0.4.9.post6":
            raise RuntimeError("this integration requires sglang==0.4.9.post6")
        config = config or PolicyConfig()
        for key in ("tp_size", "pp_size", "dp_size"):
            if engine_args.get(key, 1) != 1:
                raise ValueError(f"{key} must be 1; use independent replicas")
        incompatible = ("speculative_algorithm", "enable_hierarchical_cache", "disable_radix_cache",
                        "enable_lora", "enable_dp_attention")
        if any(engine_args.get(k) for k in incompatible):
            raise ValueError("speculation, HiCache, LoRA, DP attention, and disabled radix cache are unsupported")
        from sglang.srt.entrypoints import engine as entry
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.table = ProgramTable(table_path)
        options = dict(table_path=table_path, config=asdict(config), trace=trace)
        target = entry.run_scheduler_process
        entry.run_scheduler_process = partial(run_scheduler_process, autellix_options=options)
        args = dict(disable_overlap_schedule=True, disable_cuda_graph=True,
                    chunked_prefill_size=-1, page_size=1, max_running_requests=8,
                    schedule_policy="fcfs", attention_backend="triton")
        args.update(engine_args)
        args.update(disable_overlap_schedule=True, chunked_prefill_size=-1, page_size=1)
        try:
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
        request = GenerateReqInput(rid=internal, sampling_params=sampling, stream=True, **inputs)

        async def generate():
            gen = self.engine.tokenizer_manager.generate_request(request, None)
            try:
                result = None
                async for part in gen:
                    result = part
                    self.updates.append({"request_id": rid, "finished": False, "text": part.get("text", "")})
                if result is None:
                    raise RuntimeError("SGLang returned no output")
                # Tokenizer output and GPU completion accounting cross separate IPC paths.
                metrics = None
                for _ in range(1000):
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
        if program:
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
