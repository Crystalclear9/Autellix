from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .models import CallState, EngineState, ProcessEntry


@dataclass(frozen=True)
class ExecutionModel:
    """Approximate prefill/decode/cache/swap costs for paper simulations."""

    name: str = "autellix"
    prefill_tokens_per_step: int = 512
    decode_tokens_per_step: int = 64
    system_prompt_tokens: int = 256
    prefix_caching: bool = True
    locality_cache_bonus: float = 0.9
    remote_cache_bonus: float = 0.25
    swap_penalty_per_preemption: int = 0
    batched_swap: bool = True
    scheduler_overhead_per_call: int = 0
    fixed_model_time: bool = False

    def prepare_call(
        self,
        call: CallState,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
    ) -> None:
        entry = process_table[call.program_id]
        if self.fixed_model_time:
            call.cache_hit_rate = 0.0
            call.prefill_time = 0
            call.decode_time = call.spec.model_time
            call.scheduler_time = 0
            call.swap_time = 0
            call.model_time = call.spec.model_time
            call.remaining_time = call.model_time
            return
        cache_hit = self.cache_hit_rate(call, entry, engine.engine_id)
        effective_prefill = max(0, int(round(call.spec.prefill_tokens * (1.0 - cache_hit))))
        prefill_time = ceil(effective_prefill / max(1, self.prefill_tokens_per_step))
        decode_time = ceil(call.spec.generated_tokens / max(1, self.decode_tokens_per_step))
        call.cache_hit_rate = cache_hit
        call.prefill_time = prefill_time
        call.decode_time = max(1, decode_time)
        call.scheduler_time = self.scheduler_overhead_per_call
        call.swap_time = 0
        call.model_time = max(1, call.prefill_time + call.decode_time + call.scheduler_time)
        call.remaining_time = call.model_time

    def cache_hit_rate(
        self,
        call: CallState,
        entry: ProcessEntry,
        engine_id: int,
    ) -> float:
        if not self.prefix_caching or call.spec.prefill_tokens <= 0:
            return 0.0
        common = min(call.spec.prefill_tokens, self.system_prompt_tokens)
        common_hit = common / call.spec.prefill_tokens
        if entry.completed_call_ids and (entry.engine_id == engine_id or engine_id in entry.engine_ids):
            return min(0.98, max(common_hit, self.locality_cache_bonus))
        return min(0.75, max(common_hit, self.remote_cache_bonus))

    def preemption_penalty(self, active_calls: int) -> int:
        if self.swap_penalty_per_preemption <= 0:
            return 0
        if self.batched_swap:
            return max(1, ceil(self.swap_penalty_per_preemption / max(1, active_calls)))
        return self.swap_penalty_per_preemption


def make_execution_model(name: str | None = None) -> ExecutionModel:
    normalized = (name or "fixed").lower().replace("_", "-")
    if normalized in {"fixed", "trace", "model-time"}:
        return ExecutionModel(name="fixed", fixed_model_time=True)
    if normalized in {"vllm", "fcfs"}:
        return ExecutionModel(
            name="vllm",
            prefix_caching=False,
            swap_penalty_per_preemption=0,
            scheduler_overhead_per_call=0,
        )
    if normalized in {"vllm-opt", "vllmopt"}:
        return ExecutionModel(
            name="vllm-opt",
            prefix_caching=True,
            locality_cache_bonus=0.88,
            remote_cache_bonus=0.2,
            swap_penalty_per_preemption=2,
            batched_swap=False,
            scheduler_overhead_per_call=0,
        )
    if normalized in {"mlfq"}:
        return ExecutionModel(
            name="mlfq",
            prefix_caching=True,
            locality_cache_bonus=0.88,
            remote_cache_bonus=0.2,
            swap_penalty_per_preemption=2,
            batched_swap=False,
            scheduler_overhead_per_call=1,
        )
    if normalized in {"autellix", "plas", "atlas", "srpt", "round-robin", "roundrobin"}:
        return ExecutionModel(
            name="autellix",
            prefix_caching=True,
            locality_cache_bonus=0.92,
            remote_cache_bonus=0.25,
            swap_penalty_per_preemption=1,
            batched_swap=True,
            scheduler_overhead_per_call=1,
        )
    raise ValueError(f"unknown execution model: {name}")
