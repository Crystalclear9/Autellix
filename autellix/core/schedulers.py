from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable, Sequence

from .models import CallState, CallStatus, EngineState, ProcessEntry, ThreadMetadata

DEFAULT_PRIORITY_BOUNDARIES: tuple[float, ...] = (0, 2, 4, 8, 16, 32, 64, inf)
DEFAULT_QUEUE_QUANTA: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class Scheduler:
    name: str
    priority_boundaries: tuple[float, ...] = DEFAULT_PRIORITY_BOUNDARIES
    queue_quanta: tuple[int, ...] = DEFAULT_QUEUE_QUANTA
    anti_starvation_beta: float = 8.0
    preemptive: bool = True

    def __post_init__(self) -> None:
        if len(self.priority_boundaries) != len(self.queue_quanta) + 1:
            raise ValueError("priority_boundaries must have one more item than queue_quanta")
        if self.priority_boundaries[0] != 0:
            raise ValueError("priority_boundaries must start at 0")
        if any(q <= 0 for q in self.queue_quanta):
            raise ValueError("queue quanta must be positive")
        if self.anti_starvation_beta <= 0:
            raise ValueError("anti_starvation_beta must be positive")

    @property
    def queue_count(self) -> int:
        return len(self.queue_quanta)

    def priority_for_call(self, call: CallState, entry: ProcessEntry) -> float:
        return entry.service_time

    def assign_queue_index(self, priority: float) -> int:
        for idx in range(self.queue_count):
            low = self.priority_boundaries[idx]
            high = self.priority_boundaries[idx + 1]
            if low <= priority < high:
                return idx
        return self.queue_count - 1

    def enqueue(
        self,
        call: CallState,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
        time: int,
    ) -> None:
        entry = process_table[call.program_id]
        call.ready_time = time
        call.engine_id = engine.engine_id
        call.status = CallStatus.QUEUED
        call.service_priority = self.priority_for_call(call, entry)
        call.queue_index = self.assign_queue_index(call.service_priority)
        call.max_queue_index = max(call.max_queue_index, call.queue_index)
        call.quantum_remaining = self.queue_quanta[call.queue_index]
        engine.queues[call.queue_index].append(call)
        entry.active_call_ids.add(call.call_id)
        entry.last_arrival = time
        entry.thread_metadata[call.call_id] = ThreadMetadata(
            call_id=call.call_id,
            arrival_time=time,
            engine_id=engine.engine_id,
            queue_index=call.queue_index,
            thread_id=call.spec.thread_id,
            metadata=dict(call.spec.metadata),
            critical_path_service=call.critical_path_service,
        )

    def schedule(
        self,
        engine: EngineState,
        time: int,
        target_slots: int | None = None,
    ) -> list[CallState]:
        scheduled: list[CallState] = []
        target = engine.batch_size if target_slots is None else max(0, target_slots)
        while len(engine.running) + len(engine.prefetched) < target:
            call = self._pop_next_queued(engine)
            if call is None:
                break
            if len(engine.running) < engine.batch_size:
                self.start_call(call, engine, time)
            else:
                call.status = CallStatus.PREFETCHED
                engine.prefetched.append(call)
            scheduled.append(call)
        return scheduled

    def fill_from_prefetch(self, engine: EngineState, time: int) -> list[CallState]:
        started: list[CallState] = []
        while engine.available_slots > 0 and engine.prefetched:
            call = engine.prefetched.pop(0)
            self.start_call(call, engine, time)
            started.append(call)
        return started

    def start_call(self, call: CallState, engine: EngineState, time: int) -> None:
        call.status = CallStatus.RUNNING
        if call.start_time is None:
            call.start_time = time
        engine.running.append(call)

    def _pop_next_queued(self, engine: EngineState) -> CallState | None:
        for queue in engine.queues:
            if queue:
                return queue.popleft()
        return None

    def should_preempt(self, call: CallState) -> bool:
        return self.preemptive and call.quantum_remaining <= 0

    def on_tick_executed(self, call: CallState, amount: int = 1) -> None:
        return

    def demote(self, call: CallState, engine: EngineState) -> None:
        call.status = CallStatus.QUEUED
        call.queue_index = min(call.queue_index + 1, self.queue_count - 1)
        call.max_queue_index = max(call.max_queue_index, call.queue_index)
        call.quantum_remaining = self.queue_quanta[call.queue_index]
        engine.queues[call.queue_index].append(call)

    def tick_queued_wait(
        self,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
    ) -> set[str]:
        promoted_programs: set[str] = set()
        for queue_index, queue in enumerate(engine.queues):
            for call in list(queue):
                call.wait_time += 1
                call.wait_time_window += 1
                if self._should_promote(call, queue_index, process_table):
                    self._promote_program(call.program_id, engine, process_table)
                    promoted_programs.add(call.program_id)
        for call in engine.prefetched:
            call.wait_time += 1
            call.wait_time_window += 1
        return promoted_programs

    def _should_promote(
        self,
        call: CallState,
        queue_index: int,
        process_table: dict[str, ProcessEntry],
    ) -> bool:
        if queue_index == 0:
            return False
        entry = process_table[call.program_id]
        wait = entry.waiting_time + call.wait_time_window
        service = max(1.0, entry.service_time + call.run_time_window)
        return wait / service >= self.anti_starvation_beta

    def _promote_program(
        self,
        program_id: str,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
    ) -> None:
        for queue_index, queue in enumerate(engine.queues):
            if queue_index == 0:
                continue
            for queued in list(queue):
                if queued.program_id != program_id:
                    continue
                try:
                    queue.remove(queued)
                except ValueError:
                    continue
                queued.queue_index = 0
                queued.quantum_remaining = self.queue_quanta[0]
                queued.wait_time_window = 0
                queued.run_time_window = 0
                engine.queues[0].append(queued)
                meta = process_table[program_id].thread_metadata.get(queued.call_id)
                if meta is not None:
                    meta.queue_index = 0

    def promote_program(
        self,
        program_id: str,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
    ) -> None:
        self._promote_program(program_id, engine, process_table)

    def complete_call(
        self,
        call: CallState,
        process_table: dict[str, ProcessEntry],
        time: int,
    ) -> None:
        entry = process_table[call.program_id]
        call.status = CallStatus.FINISHED
        call.finish_time = time
        entry.active_call_ids.discard(call.call_id)
        entry.completed_call_ids.add(call.call_id)
        entry.waiting_time += call.wait_time
        entry.last_completion = time
        self.update_service_time(entry, call)
        meta = entry.thread_metadata.get(call.call_id)
        if meta is not None:
            meta.wait_time = call.wait_time
            meta.service_time = call.model_time
            meta.finish_time = time
            meta.queue_index = call.queue_index

    def update_service_time(self, entry: ProcessEntry, call: CallState) -> None:
        entry.service_time += call.model_time


@dataclass
class FCFSScheduler(Scheduler):
    name: str = "fcfs"
    priority_boundaries: tuple[float, ...] = (0, inf)
    queue_quanta: tuple[int, ...] = (10**12,)
    preemptive: bool = False

    def assign_queue_index(self, priority: float) -> int:
        return 0

    def tick_queued_wait(
        self,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
    ) -> set[str]:
        for call in engine.iter_waiting():
            call.wait_time += 1
            call.wait_time_window += 1
        return set()


@dataclass
class MLFQScheduler(Scheduler):
    name: str = "mlfq"

    def priority_for_call(self, call: CallState, entry: ProcessEntry) -> float:
        return 0.0

    def update_service_time(self, entry: ProcessEntry, call: CallState) -> None:
        return


@dataclass
class PLASScheduler(Scheduler):
    name: str = "plas"

    def update_service_time(self, entry: ProcessEntry, call: CallState) -> None:
        entry.service_time += call.model_time


@dataclass
class ATLASScheduler(Scheduler):
    name: str = "atlas"

    def priority_for_call(self, call: CallState, entry: ProcessEntry) -> float:
        return call.critical_path_service

    def update_service_time(self, entry: ProcessEntry, call: CallState) -> None:
        entry.service_time = max(entry.service_time, call.critical_path_service + call.model_time)


@dataclass
class RoundRobinScheduler(Scheduler):
    name: str = "round-robin"
    priority_boundaries: tuple[float, ...] = (0, inf)
    queue_quanta: tuple[int, ...] = (1,)
    preemptive: bool = True

    def assign_queue_index(self, priority: float) -> int:
        return 0

    def demote(self, call: CallState, engine: EngineState) -> None:
        call.status = CallStatus.QUEUED
        call.queue_index = 0
        call.quantum_remaining = self.queue_quanta[0]
        engine.queues[0].append(call)

    def tick_queued_wait(
        self,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
    ) -> set[str]:
        for call in engine.iter_waiting():
            call.wait_time += 1
            call.wait_time_window += 1
        return set()


@dataclass
class SRPTScheduler(Scheduler):
    """Clairvoyant simulator-only baseline.

    This is not part of Autellix. It sorts queued calls by each program's
    known remaining model time to mimic shortest-remaining-processing-time.
    """

    name: str = "srpt"
    priority_boundaries: tuple[float, ...] = (0, inf)
    queue_quanta: tuple[int, ...] = (1,)
    preemptive: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self._remaining_by_program: dict[str, int] = {}

    def set_program_remaining(self, remaining: dict[str, int]) -> None:
        self._remaining_by_program = dict(remaining)

    def priority_for_call(self, call: CallState, entry: ProcessEntry) -> float:
        return float(self._remaining_by_program.get(call.program_id, call.remaining_time))

    def on_tick_executed(self, call: CallState, amount: int = 1) -> None:
        current = self._remaining_by_program.get(call.program_id, call.remaining_time)
        self._remaining_by_program[call.program_id] = max(0, current - amount)

    def update_service_time(self, entry: ProcessEntry, call: CallState) -> None:
        entry.service_time += call.model_time

    def enqueue(
        self,
        call: CallState,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
        time: int,
    ) -> None:
        super().enqueue(call, engine, process_table, time)
        self._sort_queue(engine)

    def schedule(
        self,
        engine: EngineState,
        time: int,
        target_slots: int | None = None,
    ) -> list[CallState]:
        self._sort_queue(engine)
        return super().schedule(engine, time, target_slots=target_slots)

    def demote(self, call: CallState, engine: EngineState) -> None:
        call.status = CallStatus.QUEUED
        call.queue_index = 0
        call.quantum_remaining = self.queue_quanta[0]
        engine.queues[0].append(call)
        self._sort_queue(engine)

    def _sort_queue(self, engine: EngineState) -> None:
        ordered = sorted(
            engine.queues[0],
            key=lambda call: (
                self._remaining_by_program.get(call.program_id, call.remaining_time),
                call.ready_time if call.ready_time is not None else 0,
                call.program_id,
                call.call_id,
            ),
        )
        engine.queues[0].clear()
        engine.queues[0].extend(ordered)

    def tick_queued_wait(
        self,
        engine: EngineState,
        process_table: dict[str, ProcessEntry],
    ) -> set[str]:
        for call in engine.iter_waiting():
            call.wait_time += 1
            call.wait_time_window += 1
        return set()


def parse_float_tuple(value: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if not value:
        return tuple(default)
    parsed: list[float] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        parsed.append(inf if item in {"inf", "infinity"} else float(item))
    return tuple(parsed)


def parse_int_tuple(value: str | None, default: Sequence[int]) -> tuple[int, ...]:
    if not value:
        return tuple(default)
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def make_scheduler(
    name: str,
    *,
    priority_boundaries: Iterable[float] = DEFAULT_PRIORITY_BOUNDARIES,
    queue_quanta: Iterable[int] = DEFAULT_QUEUE_QUANTA,
    anti_starvation_beta: float = 8.0,
) -> Scheduler:
    normalized = name.lower().replace("_", "-")
    kwargs = {
        "priority_boundaries": tuple(priority_boundaries),
        "queue_quanta": tuple(queue_quanta),
        "anti_starvation_beta": anti_starvation_beta,
    }
    if normalized == "fcfs":
        return FCFSScheduler(anti_starvation_beta=anti_starvation_beta)
    if normalized == "mlfq":
        return MLFQScheduler(**kwargs)
    if normalized in {"round-robin", "roundrobin", "rr"}:
        return RoundRobinScheduler(anti_starvation_beta=anti_starvation_beta)
    if normalized == "plas":
        return PLASScheduler(**kwargs)
    if normalized == "atlas":
        return ATLASScheduler(**kwargs)
    if normalized == "srpt":
        return SRPTScheduler(**kwargs)
    raise ValueError(f"unknown scheduler: {name}")
