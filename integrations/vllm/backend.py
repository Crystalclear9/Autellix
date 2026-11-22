from __future__ import annotations

from importlib.metadata import version
from time import perf_counter
from types import MethodType

from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
from autellix.runtime.swap import attach_cache_engine
from .scheduler import attach_scheduler


class VLLMBackend:
    """Pinned single-GPU or tensor-parallel replica with internal scheduling."""

    def __init__(self, model: str, table_path: str, config: PolicyConfig | None = None,
                 trace: str | None = None, batched_swap: bool = True, observe_native: bool = True,
                 measure_timing: bool = False, publish_waiting_each_step: bool = True,
                 keep_resident_kv: bool = False, publish_activity_at_window: bool = False,
                 refill_idle_slots: bool = False, paper_chunked_prefill: bool = False,
                 resident_reserve_only: bool = False, fused_swap: bool = False,
                 publish_activity: bool = True, recompute_on_swap_full: bool = False,
                 resident_decode_headroom: bool = False, qwen3_fused_qk_rope: bool = False,
                 exact_prefix_cache: bool | None = None, **engine_args):
        if version("vllm") != "0.6.1":
            raise RuntimeError("this backend requires vllm==0.6.1")
        config = config or PolicyConfig()
        mode = engine_args.pop("autellix_mode", "autellix")
        if mode not in {"autellix", "vllm", "vllm-opt", "vllm-opt-multistep"}:
            raise ValueError("invalid autellix_mode")
        native = mode != "autellix"
        paper = not native and config.implementation == "paper"
        if engine_args.get("pipeline_parallel_size", 1) != 1:
            raise ValueError("pipeline parallelism is not supported; use tensor parallelism")
        tp = engine_args.get("tensor_parallel_size", 1)
        if "num_scheduler_steps" in engine_args and engine_args["num_scheduler_steps"] != config.schedule_interval:
            raise ValueError("num_scheduler_steps must match PolicyConfig.schedule_interval")
        if engine_args.get("enable_chunked_prefill", False) and not native and not paper:
            raise ValueError("chunked prefill is not supported by this pinned integration")
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        import torch

        from .qwen3_config import register_local_qwen3
        register_local_qwen3(model, tp, qwen3_fused_qk_rope)

        args = dict(enable_prefix_caching=True, enforce_eager=True,
                    disable_async_output_proc=True, use_v2_block_manager=True,
                    preemption_mode="swap", swap_space=1, max_num_seqs=8)
        if native:
            args.update(enable_prefix_caching=mode != "vllm",
                        enable_chunked_prefill=mode == "vllm-opt")
        args.update(engine_args)
        args["disable_async_output_proc"] = True
        # Native cached multi-step metadata cannot accept replacements. Reserve
        # mode uses our policy window plus native per-step tensor/block updates.
        refill_mode = (paper or config.overprovision > 0 or tp > 1) and not native
        capacity = args["max_num_seqs"]
        args["num_scheduler_steps"] = 1 if refill_mode else config.schedule_interval
        if refill_mode:
            args["max_num_seqs"] = capacity + config.overprovision
        if paper:
            # Mixed prefill/decode kernels, with admission controlled entirely
            # by PaperPlan. An explicit chunk budget can be below max_model_len.
            args["enable_chunked_prefill"] = True
            args.setdefault("max_num_batched_tokens", max(args.get("max_model_len", 2048), capacity + config.overprovision))
        if tp > 1 and not native:
            from .executor import AutellixFusedSwapMPExecutor, AutellixMPExecutor, AutellixNativeSwapMPExecutor
            executor = AutellixFusedSwapMPExecutor if fused_swap else AutellixMPExecutor
            args["distributed_executor_backend"] = executor if batched_swap else AutellixNativeSwapMPExecutor
        self.table = ProgramTable(table_path)
        self.controller = None
        if not native or observe_native:
            self.controller = RuntimeScheduler(self.table, config, trace,
                                               publish_waiting_each_step=publish_waiting_each_step,
                                               publish_activity_at_window=publish_activity_at_window and paper,
                                               publish_activity=publish_activity)
        self.engine = LLMEngine.from_engine_args(EngineArgs(model=model, **args))
        self.scheduler = self.engine.scheduler[0]
        prefix_cache = None
        supports_exact_prefix = (args["enable_prefix_caching"] and
                                 self.engine.model_config.get_sliding_window() is None)
        if exact_prefix_cache is None:
            exact_prefix_cache = paper and paper_chunked_prefill and supports_exact_prefix
        if exact_prefix_cache:
            if not supports_exact_prefix:
                raise ValueError("exact_prefix_cache requires full-context prefix caching")
            from .prefix import ChunkedPrefixCache
            prefix_cache = ChunkedPrefixCache(self.scheduler.block_manager)
        if self.controller is None:
            return
        attach_scheduler(self.scheduler, self.controller, observe_only=native)
        if refill_mode:
            from autellix.runtime.window import SchedulingWindow
            self.scheduler._autellix_window = SchedulingWindow(self.controller, capacity)
        if paper:
            from .paper import PaperPlan
            self.scheduler._autellix_paper = PaperPlan(self.scheduler, self.controller, capacity, keep_resident_kv,
                                                      refill_idle_slots=refill_idle_slots,
                                                      chunked_prefill=paper_chunked_prefill,
                                                      resident_reserve_only=resident_reserve_only,
                                                      recompute_on_swap_full=recompute_on_swap_full,
                                                      resident_decode_headroom=resident_decode_headroom,
                                                      prefix_cache=prefix_cache)
            self.scheduler._autellix_window = self.scheduler._autellix_paper.window
        worker = self.engine.model_executor.driver_worker
        if tp == 1 and batched_swap and not native:
            for cache in worker.cache_engine:
                attach_cache_engine(cache, fused=fused_swap)
        if measure_timing:
            controller = self.controller
            schedule = self.scheduler.schedule

            def timed_schedule(*args, **kwargs):
                started = perf_counter()
                result = schedule(*args, **kwargs)
                outputs = result[1]
                groups = len(outputs.scheduled_seq_groups)
                prefill = outputs.num_prefill_groups
                self.scheduler._autellix_timed_phase = ("prefill" if groups == prefill else "mixed") if prefill else "decode"
                controller.emit("schedule_time", seconds=perf_counter() - started,
                                requests=[g.seq_group.request_id for g in outputs.scheduled_seq_groups])
                return result

            def timed_transfer(transfer, direction, layers):
                def execute(mapping):
                    if not len(mapping):
                        return transfer(mapping)
                    started = perf_counter()
                    copies = transfer(mapping)
                    torch.cuda.synchronize()
                    if copies is None:
                        copies = 2 * layers * len(mapping)
                    controller.emit("kv_transfer", direction=direction, blocks=len(mapping), copies=copies,
                                    seconds=perf_counter() - started)
                return execute

            self.scheduler.schedule = timed_schedule
            for cache in worker.cache_engine:
                layers = len(cache.gpu_cache)
                cache.swap_out = timed_transfer(cache.swap_out, "out", layers)
                cache.swap_in = timed_transfer(cache.swap_in, "in", layers)
        def transfer_out(blocks):
            mapping = torch.tensor(blocks, dtype=torch.int64, device="cpu")
            worker.cache_engine[0].swap_out(mapping)
            torch.cuda.synchronize()
        self.scheduler._autellix_transfer_out = transfer_out
        if args["num_scheduler_steps"] > 1:
            original_worker_execute = worker.execute_worker
            worker._autellix_transferred_input = None
            def execute_worker_once(_worker, worker_input):
                if _worker._autellix_transferred_input is worker_input:
                    return
                original_worker_execute(worker_input)
                # MultiStepWorker reuses WorkerInput across decode steps.
                # Replaying swap-in overwrites tokens computed since the
                # first step with stale CPU blocks. Transfers are one-shot.
                _worker._autellix_transferred_input = worker_input
            worker.execute_worker = MethodType(execute_worker_once, worker)
        runner = worker.model_runner
        original = runner.execute_model
        controller, scheduler = self.controller, self.scheduler
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def measured_execute(_runner, *a, **kw):
            rids = list(scheduler._autellix_execution)
            start.record()
            result = original(*a, **kw)
            end.record()
            end.synchronize()
            if rids:
                seconds = start.elapsed_time(end) / 1000.
                controller.executed(rids, seconds)
                if measure_timing:
                    controller.emit("model_phase_time", phase=scheduler._autellix_timed_phase,
                                    seconds=seconds, requests=rids)
                if scheduler._autellix_window is not None:
                    scheduler._autellix_window.executed(scheduler._autellix_is_prefill)
            return result

        runner.execute_model = MethodType(measured_execute, runner)

    def tokenize(self, prompt: str) -> list[int]:
        return self.engine.get_tokenizer().encode(prompt)

    def add(self, rid: str, pid: str, prompt: str | list[int], sampling: dict):
        from vllm import SamplingParams
        if sampling.get("n", 1) != 1 or sampling.get("best_of", 1) != 1:
            raise ValueError("only one output sequence per call is supported")
        params = SamplingParams(**sampling)
        tokens = prompt if isinstance(prompt, list) else self.tokenize(prompt)
        if len(tokens) >= self.engine.model_config.max_model_len:
            raise ValueError("prompt leaves no room for generation within max_model_len")
        plan = getattr(self.scheduler, "_autellix_paper", None)
        if (plan is not None and not plan.chunked_prefill and
                len(tokens) > self.scheduler.scheduler_config.max_num_batched_tokens):
            raise ValueError("prompt exceeds the paper-mode whole-prefill token budget; increase max_num_batched_tokens")
        if any(t < 0 or t >= self.engine.model_config.hf_config.vocab_size for t in tokens):
            raise ValueError("input token ID outside model vocabulary")
        metadata = self.table.take_context(rid)
        if self.controller is not None:
            self.controller.admit(rid, pid, metadata=metadata)
        try:
            inputs = {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
            self.engine.add_request(rid, inputs, params)
        except BaseException:
            if self.controller is not None:
                self.controller.finish(rid, "failed")
            raise

    def step(self):
        try:
            outputs = self.engine.step()
        except BaseException:
            if self.controller is not None:
                self.controller.fail_all()
            raise
        results = []
        for result in outputs:
            choice = result.outputs[0] if result.outputs else None
            metrics = {}
            if result.finished and self.controller is not None:
                metrics = self.controller.finish(result.request_id)
            results.append({"request_id": result.request_id, "finished": result.finished,
                                "text": choice.text if choice else "",
                                "token_ids": list(choice.token_ids) if choice else [],
                                "completion_tokens": len(choice.token_ids) if choice else 0,
                                "finish_reason": choice.finish_reason if choice else "error",
                                "prompt_tokens": len(result.prompt_token_ids or []),
                            "metrics": metrics})
        return results

    def cancel(self, rid):
        self.engine.abort_request(rid)
        if self.controller is not None:
            self.controller.finish(rid, "cancelled")

    def has_work(self):
        return self.engine.has_unfinished_requests()

    def close(self):
        if self.controller is not None:
            for rid in list(self.controller.calls):
                self.cancel(rid)
        shutdown = getattr(self.engine.model_executor, "shutdown", None)
        if shutdown is not None:
            shutdown()
        self.table.close()
