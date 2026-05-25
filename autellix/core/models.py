from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import Any, Deque, Iterable, Mapping


class CallStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PREFETCHED = "prefetched"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class CallSpec:
    """Static description of one LLM call in a program DAG."""

    call_id: str
    program_id: str
    model_time: int
    prefill_tokens: int = 0
    decode_tokens: int = 0
    parents: tuple[str, ...] = ()
    release_delay: int = 0
    submit_time: int | None = None
    thread_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_time <= 0:
            raise ValueError("model_time must be positive")
        if self.prefill_tokens < 0 or self.decode_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.release_delay < 0:
            raise ValueError("release_delay must be non-negative")
        if self.submit_time is not None and self.submit_time < 0:
            raise ValueError("submit_time must be non-negative")
        object.__setattr__(self, "parents", tuple(self.parents))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    @property
    def generated_tokens(self) -> int:
        return self.decode_tokens or self.model_time

    @property
    def key(self) -> tuple[str, str]:
        return (self.program_id, self.call_id)


@dataclass(frozen=True)
class ProgramSpec:
    """A dynamic agentic program represented as an LLM-call DAG."""

    program_id: str
    calls: tuple[CallSpec, ...]
    arrival_time: int = 0

    def __post_init__(self) -> None:
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        calls = tuple(self.calls)
        if not calls:
            raise ValueError(f"program {self.program_id} must contain at least one call")
        ids = {c.call_id for c in calls}
        if len(ids) != len(calls):
            raise ValueError(f"program {self.program_id} has duplicate call ids")
        children: dict[str, list[str]] = {call.call_id: [] for call in calls}
        indegree: dict[str, int] = {call.call_id: 0 for call in calls}
        for call in calls:
            if call.program_id != self.program_id:
                raise ValueError("call.program_id must match ProgramSpec.program_id")
            missing = set(call.parents) - ids
            if missing:
                raise ValueError(f"call {call.call_id} has missing parents: {missing}")
            indegree[call.call_id] = len(call.parents)
            for parent in call.parents:
                children[parent].append(call.call_id)
        ready = [call_id for call_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            call_id = ready.pop()
            visited += 1
            for child in children[call_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(calls):
            raise ValueError(f"program {self.program_id} contains a dependency cycle")
        object.__setattr__(self, "calls", calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def generated_tokens(self) -> int:
        return sum(c.generated_tokens for c in self.calls)


@dataclass
class CallState:
    """Mutable simulation state for one call."""

    spec: CallSpec
    remaining_time: int = field(init=False)
    model_time: int = field(init=False)
    prefill_time: int = 0
    decode_time: int = 0
    cache_hit_rate: float = 0.0
    swap_time: int = 0
    scheduler_time: int = 0
    status: CallStatus = CallStatus.PENDING
    ready_time: int | None = None
    start_time: int | None = None
    finish_time: int | None = None
    engine_id: int | None = None
    queue_index: int = 0
    max_queue_index: int = 0
    quantum_remaining: int | float = inf
    service_priority: float = 0.0
    critical_path_service: float = 0.0
    wait_time: int = 0
    wait_time_window: int = 0
    executed_time: int = 0
    run_time_window: int = 0

    def __post_init__(self) -> None:
        self.model_time = self.spec.model_time
        self.remaining_time = self.model_time

    @property
    def key(self) -> tuple[str, str]:
        return self.spec.key

    @property
    def program_id(self) -> str:
        return self.spec.program_id

    @property
    def call_id(self) -> str:
        return self.spec.call_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "call_id": self.call_id,
            "thread_id": self.spec.thread_id,
            "metadata": dict(self.spec.metadata),
            "status": self.status.value,
            "engine_id": self.engine_id,
            "ready_time": self.ready_time,
            "start_time": self.start_time,
            "finish_time": self.finish_time,
            "wait_time": self.wait_time,
            "executed_time": self.executed_time,
            "model_time": self.model_time,
            "prefill_time": self.prefill_time,
            "decode_time": self.decode_time,
            "cache_hit_rate": self.cache_hit_rate,
            "swap_time": self.swap_time,
            "scheduler_time": self.scheduler_time,
            "service_priority": self.service_priority,
            "critical_path_service": self.critical_path_service,
            "max_queue_index": self.max_queue_index,
        }


@dataclass
class ThreadMetadata:
    call_id: str
    arrival_time: int
    engine_id: int
    queue_index: int
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    wait_time: int = 0
    service_time: int = 0
    critical_path_service: float = 0.0
    finish_time: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "thread_id": self.thread_id,
            "arrival_time": self.arrival_time,
            "engine_id": self.engine_id,
            "queue_index": self.queue_index,
            "metadata": dict(self.metadata),
            "wait_time": self.wait_time,
            "service_time": self.service_time,
            "critical_path_service": self.critical_path_service,
            "finish_time": self.finish_time,
        }


@dataclass
class ProcessEntry:
    """Program-level state used by Autellix schedulers."""

    program_id: str
    arrival_time: int
    service_time: float = 0.0
    waiting_time: int = 0
    engine_id: int | None = None
    engine_ids: set[int] = field(default_factory=set)
    active_call_ids: set[str] = field(default_factory=set)
    completed_call_ids: set[str] = field(default_factory=set)
    thread_metadata: dict[str, ThreadMetadata] = field(default_factory=dict)
    last_arrival: int | None = None
    last_completion: int | None = None

    @property
    def active_count(self) -> int:
        return len(self.active_call_ids)


@dataclass
class EngineState:
    """One simulated LLM engine replica."""

    engine_id: int
    batch_size: int
    queue_count: int
    queues: list[Deque[CallState]] = field(init=False)
    running: list[CallState] = field(default_factory=list)
    prefetched: list[CallState] = field(default_factory=list)
    busy_slot_steps: int = 0
    total_slot_steps: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.queue_count <= 0:
            raise ValueError("queue_count must be positive")
        self.queues = [deque() for _ in range(self.queue_count)]

    @property
    def workload(self) -> int:
        return len(self.running) + len(self.prefetched) + sum(len(q) for q in self.queues)

    @property
    def available_slots(self) -> int:
        return self.batch_size - len(self.running)

    def iter_queued(self) -> Iterable[CallState]:
        for queue in self.queues:
            yield from queue

    def iter_waiting(self) -> Iterable[CallState]:
        yield from self.iter_queued()
        yield from self.prefetched


@dataclass(frozen=True)
class ProgramMetrics:
    program_id: str
    arrival_time: int
    finish_time: int
    response_time: int
    wait_time: int
    execution_time: int
    generated_tokens: int
    token_latency: float
    call_count: int
    critical_path_response_time: int = 0
    critical_path_token_latency: float = 0.0
    prefill_time: int = 0
    decode_time: int = 0
    swap_time: int = 0
    scheduler_time: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "arrival_time": self.arrival_time,
            "finish_time": self.finish_time,
            "response_time": self.response_time,
            "wait_time": self.wait_time,
            "execution_time": self.execution_time,
            "prefill_time": self.prefill_time,
            "decode_time": self.decode_time,
            "swap_time": self.swap_time,
            "scheduler_time": self.scheduler_time,
            "generated_tokens": self.generated_tokens,
            "token_latency": self.token_latency,
            "critical_path_response_time": self.critical_path_response_time,
            "critical_path_token_latency": self.critical_path_token_latency,
            "call_count": self.call_count,
        }


@dataclass
class SimulationResult:
    policy: str
    load_balancer: str
    makespan: int
    program_metrics: dict[str, ProgramMetrics]
    calls: dict[tuple[str, str], CallState]
    process_table: dict[str, ProcessEntry]
    engine_utilization: dict[int, float]
    gantt: list[dict[str, Any]]
    prefetched_calls: int = 0

    @property
    def total_wait_time(self) -> int:
        return sum(m.wait_time for m in self.program_metrics.values())

    @property
    def total_execution_time(self) -> int:
        return sum(m.execution_time for m in self.program_metrics.values())

    @property
    def total_prefill_time(self) -> int:
        return sum(m.prefill_time for m in self.program_metrics.values())

    @property
    def total_decode_time(self) -> int:
        return sum(m.decode_time for m in self.program_metrics.values())

    @property
    def total_swap_time(self) -> int:
        return sum(m.swap_time for m in self.program_metrics.values())

    @property
    def total_scheduler_time(self) -> int:
        return sum(m.scheduler_time for m in self.program_metrics.values())

    @property
    def total_response_time(self) -> int:
        return sum(m.response_time for m in self.program_metrics.values())

    @property
    def total_generated_tokens(self) -> int:
        return sum(m.generated_tokens for m in self.program_metrics.values())

    @property
    def avg_token_latency(self) -> float:
        metrics = list(self.program_metrics.values())
        if not metrics:
            return 0.0
        return sum(m.token_latency for m in metrics) / len(metrics)

    @property
    def avg_critical_path_token_latency(self) -> float:
        metrics = list(self.program_metrics.values())
        if not metrics:
            return 0.0
        return sum(m.critical_path_token_latency for m in metrics) / len(metrics)

    @property
    def avg_response_time(self) -> float:
        metrics = list(self.program_metrics.values())
        if not metrics:
            return 0.0
        return self.total_response_time / len(metrics)

    @property
    def avg_wait_time(self) -> float:
        metrics = list(self.program_metrics.values())
        if not metrics:
            return 0.0
        return self.total_wait_time / len(metrics)

    def summary(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "load_balancer": self.load_balancer,
            "scheduler_policy": self.policy,
            "load_balancer_policy": self.load_balancer,
            "programs": len(self.program_metrics),
            "calls": len(self.calls),
            "makespan": self.makespan,
            "prefetched_calls": self.prefetched_calls,
            "total_wait_time": self.total_wait_time,
            "avg_wait_time": self.avg_wait_time,
            "avg_response_time": self.avg_response_time,
            "avg_token_latency": self.avg_token_latency,
            "avg_critical_path_token_latency": self.avg_critical_path_token_latency,
            "total_generated_tokens": self.total_generated_tokens,
            "total_execution_time": self.total_execution_time,
            "total_prefill_time": self.total_prefill_time,
            "total_decode_time": self.total_decode_time,
            "total_swap_time": self.total_swap_time,
            "total_scheduler_time": self.total_scheduler_time,
            "engine_utilization": self.engine_utilization,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.summary()
        data["program_metrics"] = {
            pid: metric.to_dict() for pid, metric in self.program_metrics.items()
        }
        data["calls"] = {
            f"{pid}:{cid}": state.to_dict()
            for (pid, cid), state in sorted(self.calls.items())
        }
        return data
