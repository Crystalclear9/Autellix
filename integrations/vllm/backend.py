from __future__ import annotations

from importlib.metadata import version
from types import MethodType

from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
from autellix.runtime.swap import attach_cache_engine
from .scheduler import attach_scheduler


class VLLMBackend:
    """Real, single-GPU vLLM 0.6.1 replica with internal Autellix scheduling.

    Scale through independent replicas. Pipeline/tensor parallel execution is
    rejected because timing and scheduler-state synchronization differ.
    """

    def __init__(self, model: str, table_path: str, config: PolicyConfig | None = None,
                 trace: str | None = None, batched_swap: bool = True, **engine_args):
        if version("vllm") != "0.6.1":
            raise RuntimeError("this backend requires vllm==0.6.1")
        config = config or PolicyConfig()
        for key in ("tensor_parallel_size", "pipeline_parallel_size"):
            if engine_args.get(key, 1) != 1:
                raise ValueError(f"{key} must be 1; Autellix manages replicas and scheduling intervals")
        if "num_scheduler_steps" in engine_args and engine_args["num_scheduler_steps"] != config.schedule_interval:
            raise ValueError("num_scheduler_steps must match PolicyConfig.schedule_interval")
        if engine_args.get("enable_chunked_prefill", False):
            raise ValueError("chunked prefill is not supported by this pinned integration")
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        import torch

        args = dict(enable_prefix_caching=True, enforce_eager=True,
                    disable_async_output_proc=True, use_v2_block_manager=True,
                    preemption_mode="swap", swap_space=1, max_num_seqs=8)
        args.update(engine_args)
        args["disable_async_output_proc"] = True
        args["num_scheduler_steps"] = config.schedule_interval
        self.table = ProgramTable(table_path)
        self.controller = RuntimeScheduler(self.table, config, trace)
        self.engine = LLMEngine.from_engine_args(EngineArgs(model=model, **args))
        self.scheduler = attach_scheduler(self.engine.scheduler[0], self.controller)
        worker = self.engine.model_executor.driver_worker
        if batched_swap:
            for cache in worker.cache_engine:
                attach_cache_engine(cache)
        def transfer_out(blocks):
            mapping = torch.tensor(blocks, dtype=torch.int64, device="cpu")
            worker.cache_engine[0].swap_out(mapping)
            torch.cuda.synchronize()
        self.scheduler._autellix_transfer_out = transfer_out
        if config.schedule_interval > 1:
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

        def measured_execute(_runner, *a, **kw):
            rids = list(scheduler._autellix_execution)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*a, **kw)
            end.record()
            end.synchronize()
            if rids:
                controller.executed(rids, start.elapsed_time(end) / 1000.)
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
        if any(t < 0 or t >= self.engine.model_config.hf_config.vocab_size for t in tokens):
            raise ValueError("input token ID outside model vocabulary")
        self.controller.admit(rid, pid, metadata=self.table.take_context(rid))
        try:
            inputs = {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
            self.engine.add_request(rid, inputs, params)
        except BaseException:
            self.controller.finish(rid, "failed")
            raise

    def step(self):
        try:
            outputs = self.engine.step()
        except BaseException:
            self.controller.fail_all()
            raise
        results = []
        for result in outputs:
            choice = result.outputs[0] if result.outputs else None
            metrics = {}
            if result.finished:
                metrics = self.controller.finish(result.request_id)
            results.append({"request_id": result.request_id, "finished": result.finished,
                                "text": choice.text if choice else "",
                                "token_ids": list(choice.token_ids) if choice else [],
                                "finish_reason": choice.finish_reason if choice else "error",
                                "prompt_tokens": len(result.prompt_token_ids or []),
                            "metrics": metrics})
        return results

    def cancel(self, rid):
        self.engine.abort_request(rid)
        self.controller.finish(rid, "cancelled")

    def has_work(self):
        return self.engine.has_unfinished_requests()

    def close(self):
        for rid in list(self.controller.calls):
            self.cancel(rid)
        self.table.close()
