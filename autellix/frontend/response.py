from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core.models import CallState, SimulationResult


@dataclass(frozen=True)
class SimulatedChatResponse:
    """Lightweight response for simulated OpenAI/vLLM-style calls."""

    session_id: str
    program_id: str
    call_id: str
    engine_id: int | None
    submit_time: int | None
    finish_time: int | None
    status: str
    content: str
    usage: dict[str, int]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "program_id": self.program_id,
            "call_id": self.call_id,
            "engine_id": self.engine_id,
            "submit_time": self.submit_time,
            "finish_time": self.finish_time,
            "status": self.status,
            "content": self.content,
            "usage": dict(self.usage),
            "metrics": dict(self.metrics),
        }


def make_simulated_response(
    *,
    session_id: str,
    state: CallState,
    result: SimulationResult | None = None,
    content: str | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> SimulatedChatResponse:
    usage = {
        "prompt_tokens": state.spec.prefill_tokens,
        "completion_tokens": state.spec.decode_tokens or state.spec.generated_tokens,
        "total_tokens": state.spec.prefill_tokens
        + (state.spec.decode_tokens or state.spec.generated_tokens),
    }
    call_metrics: dict[str, Any] = {
        "thread_id": state.spec.thread_id,
        "call_metadata": dict(state.spec.metadata),
        "wait_time": state.wait_time,
        "executed_time": state.executed_time,
        "model_time": state.model_time,
        "prefill_time": state.prefill_time,
        "decode_time": state.decode_time,
        "cache_hit_rate": state.cache_hit_rate,
        "swap_time": state.swap_time,
        "scheduler_time": state.scheduler_time,
        "service_priority": state.service_priority,
        "critical_path_service": state.critical_path_service,
        "queue_index": state.queue_index,
    }
    if result is not None:
        call_metrics.update(
            {
                "scheduler_policy": result.policy,
                "load_balancer_policy": result.load_balancer,
            }
        )
        program_metric = result.program_metrics.get(state.program_id)
        if program_metric is not None:
            call_metrics.update(
                {
                    "program_response_time": program_metric.response_time,
                    "program_wait_time": program_metric.wait_time,
                    "program_execution_time": program_metric.execution_time,
                    "program_token_latency": program_metric.token_latency,
                    "critical_path_response_time": program_metric.critical_path_response_time,
                    "critical_path_token_latency": program_metric.critical_path_token_latency,
                }
            )
    if metrics:
        call_metrics.update(dict(metrics))

    status = state.status.value if hasattr(state.status, "value") else str(state.status)
    return SimulatedChatResponse(
        session_id=session_id,
        program_id=state.program_id,
        call_id=state.call_id,
        engine_id=state.engine_id,
        submit_time=state.spec.submit_time if state.spec.submit_time is not None else state.ready_time,
        finish_time=state.finish_time,
        status=status,
        content=content or f"[simulated completion for {state.program_id}:{state.call_id}]",
        usage=usage,
        metrics=call_metrics,
    )
