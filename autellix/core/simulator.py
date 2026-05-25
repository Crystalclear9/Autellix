from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .execution import ExecutionModel, make_execution_model
from .load_balancer import LoadBalancer, make_load_balancer
from .models import (
    CallState,
    CallStatus,
    EngineState,
    ProcessEntry,
    ProgramMetrics,
    ProgramSpec,
    SimulationResult,
)
from .schedulers import Scheduler, SRPTScheduler, make_scheduler


def is_sequential_program(program: ProgramSpec) -> bool:
    """Return true for a single root chain with no fork or join."""

    if not program.calls:
        return False
    children: dict[str, list[str]] = {call.call_id: [] for call in program.calls}
    roots = 0
    for call in program.calls:
        if not call.parents:
            roots += 1
        if len(call.parents) > 1:
            return False
        for parent in call.parents:
            children[parent].append(call.call_id)
            if len(children[parent]) > 1:
                return False
    return roots == 1


class Simulator:
    """Discrete-time simulator for program-aware LLM serving."""

    def __init__(
        self,
        programs: Iterable[ProgramSpec],
        scheduler: Scheduler | str = "plas",
        load_balancer: LoadBalancer | str | None = None,
        *,
        num_engines: int = 1,
        batch_size: int = 2,
        schedule_interval: int = 1,
        overprovision: int = 0,
        execution_model: ExecutionModel | str | None = None,
        max_time: int = 1_000_000,
    ) -> None:
        self.programs = tuple(programs)
        if not self.programs:
            raise ValueError("at least one program is required")
        if num_engines <= 0:
            raise ValueError("num_engines must be positive")
        if schedule_interval <= 0:
            raise ValueError("schedule_interval must be positive")
        self.scheduler = make_scheduler(scheduler) if isinstance(scheduler, str) else scheduler
        self.load_balancer = (
            make_load_balancer(load_balancer)
            if isinstance(load_balancer, str) or load_balancer is None
            else load_balancer
        )
        self.schedule_interval = schedule_interval
        self.overprovision = max(0, overprovision)
        self.execution_model = (
            make_execution_model(execution_model)
            if isinstance(execution_model, str) or execution_model is None
            else execution_model
        )
        self.max_time = max_time
        self.engines = [
            EngineState(i, batch_size=batch_size, queue_count=self.scheduler.queue_count)
            for i in range(num_engines)
        ]
        self.process_table: dict[str, ProcessEntry] = {}
        self.calls: dict[tuple[str, str], CallState] = {}
        self.program_by_id: dict[str, ProgramSpec] = {}
        self.children: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._pending: set[tuple[str, str]] = set()
        self.gantt: list[dict[str, int | str]] = []
        self.prefetched_calls = 0
        self._build_state()
        self.time = min(program.arrival_time for program in self.programs)

    def _build_state(self) -> None:
        program_ids: set[str] = set()
        remaining_by_program: dict[str, int] = {}
        for program in self.programs:
            if program.program_id in program_ids:
                raise ValueError(f"duplicate program id: {program.program_id}")
            program_ids.add(program.program_id)
            self.program_by_id[program.program_id] = program
            remaining_by_program[program.program_id] = sum(c.model_time for c in program.calls)
            for spec in program.calls:
                key = spec.key
                self.calls[key] = CallState(spec)
                self._pending.add(key)
                for parent in spec.parents:
                    self.children[(program.program_id, parent)].append(spec.call_id)
        if isinstance(self.scheduler, SRPTScheduler):
            self.scheduler.set_program_remaining(remaining_by_program)

    def run(self) -> SimulationResult:
        while not self._finished():
            self._step_once(advance_idle=True)

        return self._result(self.time)

    def step(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        for _ in range(steps):
            if self._finished():
                self.time += 1
                continue
            self._step_once(advance_idle=False)

    def run_until_idle(self) -> SimulationResult:
        while not self._finished() and any(
            engine.running or engine.prefetched or any(engine.queues)
            for engine in self.engines
        ):
            self._step_once(advance_idle=False)
        return self._result(self.time)

    def add_call(
        self,
        spec: CallSpec,
        *,
        arrival_time: int | None = None,
        replace_existing: bool = False,
    ) -> None:
        if spec.key in self.calls:
            if not replace_existing:
                raise ValueError(f"duplicate call: {spec.program_id}:{spec.call_id}")
            return
        if arrival_time is None:
            arrival_time = spec.submit_time if spec.submit_time is not None else self.time
        if spec.program_id in self.program_by_id:
            current = self.program_by_id[spec.program_id]
            program = ProgramSpec(
                spec.program_id,
                current.calls + (spec,),
                arrival_time=current.arrival_time,
            )
            self.programs = tuple(
                program if item.program_id == spec.program_id else item
                for item in self.programs
            )
        else:
            program = ProgramSpec(spec.program_id, (spec,), arrival_time=arrival_time)
            self.programs = self.programs + (program,)
            if len(self.programs) == 1:
                self.time = min(self.time, program.arrival_time)
        self.program_by_id[spec.program_id] = program
        self.calls[spec.key] = CallState(spec)
        self._pending.add(spec.key)
        for parent in spec.parents:
            self.children[(spec.program_id, parent)].append(spec.call_id)
        if isinstance(self.scheduler, SRPTScheduler):
            remaining = dict(self.scheduler._remaining_by_program)
            remaining[spec.program_id] = remaining.get(spec.program_id, 0) + spec.model_time
            self.scheduler.set_program_remaining(remaining)

    def _step_once(self, *, advance_idle: bool) -> None:
        if self.time > self.max_time:
            raise RuntimeError(f"simulation exceeded max_time={self.max_time}")

        self._release_ready_calls(self.time)
        self._schedule(self.time)
        self._execute_one_tick(self.time)

        self.time += 1
        if advance_idle and self._idle_with_future_work(self.time):
            self.time = self._next_possible_release_time(self.time)

    def _finished(self) -> bool:
        return all(call.status == CallStatus.FINISHED for call in self.calls.values())

    def _idle_with_future_work(self, time: int) -> bool:
        if any(engine.running or engine.prefetched or any(engine.queues) for engine in self.engines):
            return False
        return any(self._earliest_ready_time(key) > time for key in self._pending)

    def _next_possible_release_time(self, time: int) -> int:
        if not self._pending:
            return time
        return max(time, min(self._earliest_ready_time(key) for key in self._pending))

    def _ensure_process_entry(self, program: ProgramSpec) -> ProcessEntry:
        entry = self.process_table.get(program.program_id)
        if entry is None:
            entry = ProcessEntry(program_id=program.program_id, arrival_time=program.arrival_time)
            self.process_table[program.program_id] = entry
        return entry

    def _release_ready_calls(self, time: int) -> None:
        for key in sorted(list(self._pending), key=lambda item: (self.program_by_id[item[0]].arrival_time, item)):
            if self._earliest_ready_time(key) > time:
                continue
            state = self.calls[key]
            program = self.program_by_id[state.program_id]
            self._ensure_process_entry(program)
            engine = self.load_balancer.assign(state.spec, self.engines, self.process_table)
            self.execution_model.prepare_call(state, engine, self.process_table)
            self._compute_critical_path_service(state)
            self.scheduler.enqueue(state, engine, self.process_table, time)
            self._pending.remove(key)

    def _earliest_ready_time(self, key: tuple[str, str]) -> int:
        state = self.calls[key]
        program = self.program_by_id[state.program_id]
        if state.spec.submit_time is not None:
            base_time = max(program.arrival_time, state.spec.submit_time)
        else:
            base_time = program.arrival_time
        if not state.spec.parents:
            return base_time + state.spec.release_delay
        parent_finish_times: list[int] = []
        for parent_id in state.spec.parents:
            parent = self.calls[(state.program_id, parent_id)]
            if parent.finish_time is None:
                return self.max_time + 1
            parent_finish_times.append(parent.finish_time)
        return max(base_time, max(parent_finish_times) + state.spec.release_delay)

    def _schedule(self, time: int) -> None:
        for engine in self.engines:
            self.scheduler.fill_from_prefetch(engine, time)
            if time % self.schedule_interval != 0:
                continue
            scheduled = self.scheduler.schedule(
                engine,
                time,
                target_slots=engine.batch_size + self.overprovision,
            )
            self.prefetched_calls += sum(1 for call in scheduled if call.status == CallStatus.PREFETCHED)

    def _execute_one_tick(self, time: int) -> None:
        for engine in self.engines:
            engine.total_slot_steps += engine.batch_size
            engine.busy_slot_steps += len(engine.running)
            finished: list[CallState] = []
            exhausted: list[CallState] = []
            for call in list(engine.running):
                call.remaining_time -= 1
                call.executed_time += 1
                call.run_time_window += 1
                self.scheduler.on_tick_executed(call)
                if call.quantum_remaining != float("inf"):
                    call.quantum_remaining -= 1
                self.gantt.append(
                    {
                        "time": time,
                        "engine_id": engine.engine_id,
                        "program_id": call.program_id,
                        "call_id": call.call_id,
                        "queue_index": call.queue_index,
                    }
                )
                if call.remaining_time <= 0:
                    finished.append(call)
                elif self.scheduler.should_preempt(call):
                    exhausted.append(call)

            for call in finished:
                if call in engine.running:
                    engine.running.remove(call)
                self.scheduler.complete_call(call, self.process_table, time + 1)

            promoted_programs = self.scheduler.tick_queued_wait(engine, self.process_table)
            for program_id in promoted_programs:
                for peer in self.engines:
                    if peer is not engine:
                        self.scheduler.promote_program(program_id, peer, self.process_table)

            for call in exhausted:
                if call.status == CallStatus.FINISHED:
                    continue
                if call in engine.running:
                    engine.running.remove(call)
                penalty = self.execution_model.preemption_penalty(len(exhausted))
                if penalty:
                    call.swap_time += penalty
                self.scheduler.demote(call, engine)

            self.scheduler.fill_from_prefetch(engine, time + 1)

    def _compute_critical_path_service(self, state: CallState) -> None:
        if not state.spec.parents:
            state.critical_path_service = 0.0
            return
        state.critical_path_service = max(
            self.calls[(state.program_id, parent_id)].critical_path_service
            + self.calls[(state.program_id, parent_id)].model_time
            for parent_id in state.spec.parents
        )

    def _critical_path_response_time(self, program: ProgramSpec) -> int:
        if is_sequential_program(program):
            states = [self.calls[spec.key] for spec in program.calls]
            return max(state.finish_time or 0 for state in states) - program.arrival_time
        by_id = {call.call_id: call for call in program.calls}
        memo: dict[str, int] = {}

        def visit(call_id: str) -> int:
            if call_id in memo:
                return memo[call_id]
            spec = by_id[call_id]
            state = self.calls[spec.key]
            ready_time = state.ready_time if state.ready_time is not None else program.arrival_time
            finish_time = state.finish_time if state.finish_time is not None else ready_time
            local_response = max(0, finish_time - ready_time)
            if not spec.parents:
                value = local_response
            else:
                value = max(visit(parent_id) for parent_id in spec.parents) + local_response
            memo[call_id] = value
            return value

        return max(visit(spec.call_id) for spec in program.calls)

    def _result(self, end_time: int) -> SimulationResult:
        metrics: dict[str, ProgramMetrics] = {}
        min_arrival = min(program.arrival_time for program in self.programs)
        max_finish = min_arrival
        for program in self.programs:
            states = [self.calls[spec.key] for spec in program.calls]
            finish_time = max(state.finish_time or 0 for state in states)
            max_finish = max(max_finish, finish_time)
            response_time = finish_time - program.arrival_time
            wait_time = sum(state.wait_time for state in states)
            execution_time = sum(state.model_time for state in states)
            prefill_time = sum(state.prefill_time for state in states)
            decode_time = sum(state.decode_time for state in states)
            swap_time = sum(state.swap_time for state in states)
            scheduler_time = sum(state.scheduler_time for state in states)
            generated_tokens = sum(spec.generated_tokens for spec in program.calls)
            critical_path_response_time = self._critical_path_response_time(program)
            critical_path_token_latency = critical_path_response_time / max(1, generated_tokens)
            if is_sequential_program(program):
                token_latency = response_time / max(1, generated_tokens)
            else:
                token_latency = critical_path_token_latency
            metrics[program.program_id] = ProgramMetrics(
                program_id=program.program_id,
                arrival_time=program.arrival_time,
                finish_time=finish_time,
                response_time=response_time,
                wait_time=wait_time,
                execution_time=execution_time,
                prefill_time=prefill_time,
                decode_time=decode_time,
                swap_time=swap_time,
                scheduler_time=scheduler_time,
                generated_tokens=generated_tokens,
                token_latency=token_latency,
                critical_path_response_time=critical_path_response_time,
                critical_path_token_latency=critical_path_token_latency,
                call_count=len(program.calls),
            )
        utilization = {
            engine.engine_id: (
                engine.busy_slot_steps / engine.total_slot_steps
                if engine.total_slot_steps
                else 0.0
            )
            for engine in self.engines
        }
        return SimulationResult(
            policy=self.scheduler.name,
            load_balancer=self.load_balancer.name,
            makespan=max_finish - min_arrival,
            program_metrics=metrics,
            calls=self.calls,
            process_table=self.process_table,
            engine_utilization=utilization,
            gantt=self.gantt,
            prefetched_calls=self.prefetched_calls,
        )
