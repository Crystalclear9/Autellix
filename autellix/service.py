from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from math import ceil
from typing import Any

from .execution import ExecutionModel, make_execution_model
from .load_balancer import LoadBalancer, make_load_balancer
from .models import CallSpec, ProgramSpec, SimulationResult
from .response import SimulatedChatResponse, make_simulated_response
from .schedulers import Scheduler, make_scheduler
from .simulator import Simulator


@dataclass
class Session:
    session_id: str
    program_id: str
    arrival_time: int = 0
    calls: list[CallSpec] = field(default_factory=list)
    completed: bool = False

    def to_program(self) -> ProgramSpec:
        return ProgramSpec(self.program_id, tuple(self.calls), arrival_time=self.arrival_time)


class AutellixService:
    """Stateful frontend-like API for building dynamic program traces."""

    def __init__(
        self,
        *,
        scheduler: Scheduler | str = "plas",
        load_balancer: LoadBalancer | str | None = None,
        num_engines: int = 1,
        batch_size: int = 2,
        schedule_interval: int = 1,
        overprovision: int = 0,
        execution_model: ExecutionModel | str | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.load_balancer = load_balancer
        self.num_engines = num_engines
        self.batch_size = batch_size
        self.schedule_interval = schedule_interval
        self.overprovision = overprovision
        self.execution_model = execution_model
        self._counter = count(1)
        self.sessions: dict[str, Session] = {}
        self.time = 0
        self.last_result: SimulationResult | None = None
        self._simulator: Simulator | None = None
        self._program_session_ids: dict[str, str] = {}
        self._call_counters: dict[str, count] = {}

    def start_session(
        self,
        program_id: str | None = None,
        *,
        arrival_time: int = 0,
    ) -> Session:
        sid = f"s{next(self._counter):06d}"
        session = Session(
            session_id=sid,
            program_id=program_id or sid,
            arrival_time=arrival_time,
        )
        self.sessions[sid] = session
        self._program_session_ids[session.program_id] = sid
        self._call_counters[sid] = count(1)
        return session

    def submit_call(
        self,
        session_id: str,
        call_id: str,
        *,
        model_time: int,
        prefill_tokens: int = 0,
        decode_tokens: int = 0,
        parents: tuple[str, ...] = (),
        release_delay: int = 0,
    ) -> CallSpec:
        session = self.sessions[session_id]
        if session.completed:
            raise ValueError(f"session {session_id} is already completed")
        if parents:
            known = {call.call_id for call in session.calls}
            missing = set(parents) - known
            if missing:
                raise ValueError(f"call {call_id} has missing parents: {missing}")
        call = CallSpec(
            call_id=call_id,
            program_id=session.program_id,
            model_time=model_time,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            parents=parents,
            release_delay=release_delay,
            submit_time=self.time,
        )
        session.calls.append(call)
        self._ensure_live_simulator(call)
        if self._simulator is not None:
            self._simulator.add_call(call, arrival_time=session.arrival_time, replace_existing=True)
        return call

    def complete_session(self, session_id: str) -> ProgramSpec:
        session = self.sessions[session_id]
        session.completed = True
        return session.to_program()

    def end_session(self, session_id: str) -> ProgramSpec:
        """Mark a session complete, mirroring the paper's frontend lifecycle."""

        return self.complete_session(session_id)

    def chat_completion(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        call_id: str | None = None,
        model_time: int | None = None,
        prefill_tokens: int | None = None,
        decode_tokens: int | None = None,
        parents: tuple[str, ...] = (),
        release_delay: int = 0,
        max_tokens: int | None = None,
        **metadata: Any,
    ) -> SimulatedChatResponse:
        """Submit a deterministic OpenAI Chat Completion-style simulated call."""

        if call_id is None:
            call_id = f"chat{next(self._call_counters[session_id]):06d}"
        prompt_tokens = (
            prefill_tokens
            if prefill_tokens is not None
            else self._estimate_prompt_tokens(messages)
        )
        completion_tokens = (
            decode_tokens
            if decode_tokens is not None
            else max(1, max_tokens if max_tokens is not None else min(256, max(16, prompt_tokens // 8)))
        )
        simulated_model_time = (
            model_time
            if model_time is not None
            else max(1, ceil(prompt_tokens / 512) + ceil(completion_tokens / 64))
        )
        call = self.submit_call(
            session_id,
            call_id,
            model_time=simulated_model_time,
            prefill_tokens=prompt_tokens,
            decode_tokens=completion_tokens,
            parents=parents,
            release_delay=release_delay,
        )
        state = self._call_state(call)
        response_metadata = {"request_metadata": metadata} if metadata else None
        return make_simulated_response(
            session_id=session_id,
            state=state,
            result=self.last_result,
            metrics=response_metadata,
        )

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
        total_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                total_chars += sum(len(str(part)) for part in content)
            else:
                total_chars += len(str(content))
            total_chars += len(str(message.get("role", "")))
        return max(1, ceil(total_chars / 4))

    def _call_state(self, call: CallSpec):
        if self._simulator is not None and call.key in self._simulator.calls:
            return self._simulator.calls[call.key]
        if self.last_result is not None and call.key in self.last_result.calls:
            return self.last_result.calls[call.key]
        from .models import CallState

        return CallState(call)

    def _programs(self, *, include_open: bool) -> list[ProgramSpec]:
        programs = [
            session.to_program()
            for session in self.sessions.values()
            if session.completed or include_open
        ]
        if not programs:
            detail = "submitted" if include_open else "completed"
            raise ValueError(f"no {detail} sessions to simulate")
        return programs

    def _make_simulator(self, programs: list[ProgramSpec]) -> Simulator:
        scheduler = self.scheduler if not isinstance(self.scheduler, str) else make_scheduler(self.scheduler)
        load_balancer = (
            self.load_balancer
            if not isinstance(self.load_balancer, str) and self.load_balancer is not None
            else make_load_balancer(self.load_balancer)
        )
        execution_model = (
            self.execution_model
            if not isinstance(self.execution_model, str) and self.execution_model is not None
            else make_execution_model(self.execution_model)
        )
        return Simulator(
            programs,
            scheduler=scheduler,
            load_balancer=load_balancer,
            num_engines=self.num_engines,
            batch_size=self.batch_size,
            schedule_interval=self.schedule_interval,
            overprovision=self.overprovision,
            execution_model=execution_model,
        )

    def _ensure_live_simulator(self, first_call: CallSpec | None = None) -> None:
        if self._simulator is not None:
            return
        if first_call is None:
            return
        scheduler = self.scheduler if not isinstance(self.scheduler, str) else make_scheduler(self.scheduler)
        load_balancer = (
            self.load_balancer
            if not isinstance(self.load_balancer, str) and self.load_balancer is not None
            else make_load_balancer(self.load_balancer)
        )
        execution_model = (
            self.execution_model
            if not isinstance(self.execution_model, str) and self.execution_model is not None
            else make_execution_model(self.execution_model)
        )
        session = self.sessions[self._program_session_ids[first_call.program_id]]
        program = ProgramSpec(
            first_call.program_id,
            (first_call,),
            arrival_time=session.arrival_time,
        )
        self._simulator = Simulator(
            [program],
            scheduler=scheduler,
            load_balancer=load_balancer,
            num_engines=self.num_engines,
            batch_size=self.batch_size,
            schedule_interval=self.schedule_interval,
            overprovision=self.overprovision,
            execution_model=execution_model,
        )
        if self.time > self._simulator.time:
            self._simulator.time = self.time

    def tick(self, steps: int = 1) -> SimulationResult | None:
        """Advance frontend time and admit already submitted calls.

        The CPU-only service keeps deterministic trace-replay semantics while
        exposing the paper's online session lifecycle. Calls submitted before a
        tick receive a concrete submit time; `run_until_idle`/`drain` replay the
        trace with those submit times, so dynamic and static submissions match.
        """

        if steps <= 0:
            raise ValueError("steps must be positive")
        if self._simulator is None:
            self.time += steps
            return None
        self._simulator.step(steps)
        self.time = self._simulator.time
        self.last_result = self._simulator._result(self._simulator.time)
        return self.last_result

    def run_until_idle(self) -> SimulationResult:
        if self._simulator is None:
            self.last_result = self._make_simulator(self._programs(include_open=True)).run()
            return self.last_result
        self.last_result = self._simulator.run_until_idle()
        self.time = self._simulator.time
        return self.last_result

    def drain(self) -> SimulationResult:
        if self._simulator is None:
            result = self.run_until_idle()
        else:
            result = self._simulator.run()
            self.time = self._simulator.time
            self.last_result = result
        finished = set(result.program_metrics)
        for session_id, session in list(self.sessions.items()):
            if session.program_id in finished:
                session.completed = True
                del self.sessions[session_id]
                self._program_session_ids.pop(session.program_id, None)
                self._call_counters.pop(session_id, None)
        self._simulator = None
        return result

    def run(self) -> SimulationResult:
        if self._simulator is not None:
            result = self._simulator.run()
        else:
            result = self._make_simulator(self._programs(include_open=False)).run()
        self.last_result = result
        return result
