from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pipe, Process, Queue
from typing import Any

from ..core.models import CallState, ProgramSpec, SimulationResult
from .response import SimulatedChatResponse, make_simulated_response
from .service import AutellixService, Session


@dataclass
class SimulatedRequestFuture:
    """Future-like handle for a simulated LLM request."""

    _engine: "AsyncMultiLLMEngine"
    session_id: str
    call_key: tuple[str, str]

    @property
    def engine_id(self) -> int | None:
        return self._state().engine_id

    def done(self) -> bool:
        return self._state().finish_time is not None

    def result(self) -> SimulatedChatResponse:
        if not self.done():
            program_id, call_id = self.call_key
            raise RuntimeError(
                f"simulated request {program_id}:{call_id} is not finished; "
                "call step(), run_until_idle(), or drain() first"
            )
        state = self._state()
        return make_simulated_response(
            session_id=self.session_id,
            state=state,
            result=self._engine.last_result,
        )

    def _state(self) -> CallState:
        return self._engine._state_for_key(self.call_key)


class AsyncMultiLLMEngine:
    """In-repository async facade over the deterministic multi-engine simulator."""

    def __init__(
        self,
        *,
        scheduler: Any = "plas",
        load_balancer: Any = None,
        num_engines: int = 1,
        batch_size: int = 2,
        schedule_interval: int = 1,
        overprovision: int = 0,
        execution_model: Any = None,
        process_mode: bool = False,
    ) -> None:
        self.process_mode = process_mode
        self.service = AutellixService(
            scheduler=scheduler,
            load_balancer=load_balancer,
            num_engines=num_engines,
            batch_size=batch_size,
            schedule_interval=schedule_interval,
            overprovision=overprovision,
            execution_model=execution_model,
        )
        self._program_sessions: dict[str, str] = {}
        self._futures: list[SimulatedRequestFuture] = []
        self._worker: _ProcessWorker | None = None
        if process_mode:
            self._worker = _ProcessWorker(
                {
                    "scheduler": scheduler,
                    "load_balancer": load_balancer,
                    "num_engines": num_engines,
                    "batch_size": batch_size,
                    "schedule_interval": schedule_interval,
                    "overprovision": overprovision,
                    "execution_model": execution_model,
                }
            )

    @property
    def last_result(self) -> SimulationResult | None:
        return self.service.last_result

    def start_session(
        self,
        program_id: str | None = None,
        *,
        arrival_time: int = 0,
    ) -> Session:
        session = self.service.start_session(program_id, arrival_time=arrival_time)
        self._program_sessions[session.program_id] = session.session_id
        return session

    def end_session(self, session_id: str) -> ProgramSpec:
        return self.service.end_session(session_id)

    def submit_call(
        self,
        program_id: str,
        call_id: str,
        *,
        model_time: int,
        prefill_tokens: int = 0,
        decode_tokens: int = 0,
        parents: tuple[str, ...] = (),
        release_delay: int = 0,
        session_id: str | None = None,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulatedRequestFuture:
        if self._worker is not None:
            self._worker.submit(
                program_id=program_id,
                call_id=call_id,
                model_time=model_time,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                parents=parents,
                release_delay=release_delay,
                thread_id=thread_id,
                metadata=metadata,
            )
        if session_id is None:
            session_id = self._program_sessions.get(program_id)
            if session_id is None:
                session = self.start_session(program_id, arrival_time=self.service.time)
                session_id = session.session_id
        call = self.service.submit_call(
            session_id,
            call_id,
            model_time=model_time,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            parents=parents,
            release_delay=release_delay,
            thread_id=thread_id,
            metadata=metadata,
        )
        future = SimulatedRequestFuture(self, session_id=session_id, call_key=call.key)
        self._futures.append(future)
        return future

    def step(self, steps: int = 1) -> SimulationResult | None:
        if self._worker is not None:
            self._worker.step(steps)
        return self.service.tick(steps)

    def run_until_idle(self) -> SimulationResult:
        if self._worker is not None:
            self._worker.run_until_idle()
        return self.service.run_until_idle()

    def drain(self) -> SimulationResult:
        if self._worker is not None:
            self._worker.drain()
        return self.service.drain()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def futures(self) -> tuple[SimulatedRequestFuture, ...]:
        return tuple(self._futures)

    def _state_for_key(self, key: tuple[str, str]) -> CallState:
        if self.service._simulator is not None and key in self.service._simulator.calls:
            return self.service._simulator.calls[key]
        if self.service.last_result is not None and key in self.service.last_result.calls:
            return self.service.last_result.calls[key]
        raise KeyError(f"unknown simulated request: {key[0]}:{key[1]}")


class _ProcessWorker:
    def __init__(self, service_kwargs: dict[str, Any]) -> None:
        self._commands: Queue = Queue()
        self._parent_conn, child_conn = Pipe()
        self._process = Process(
            target=_worker_main,
            args=(service_kwargs, self._commands, child_conn),
        )
        self._process.daemon = True
        self._process.start()

    def submit(self, **payload: Any) -> None:
        self._commands.put(("submit", payload))

    def step(self, steps: int) -> None:
        self._commands.put(("step", {"steps": steps}))
        self._drain_events()

    def run_until_idle(self) -> None:
        self._commands.put(("run_until_idle", {}))
        self._drain_events()

    def drain(self) -> None:
        self._commands.put(("drain", {}))
        self._drain_events()

    def close(self) -> None:
        if self._process.is_alive():
            self._commands.put(("close", {}))
            self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)

    def _drain_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while self._parent_conn.poll():
            events.append(self._parent_conn.recv())
        return events


def _worker_main(service_kwargs: dict[str, Any], commands: Queue, conn) -> None:
    service = AutellixService(**service_kwargs)
    sessions: dict[str, str] = {}
    try:
        while True:
            command, payload = commands.get()
            if command == "close":
                conn.send({"event": "closed"})
                return
            if command == "submit":
                program_id = payload["program_id"]
                session_id = sessions.get(program_id)
                if session_id is None:
                    session = service.start_session(program_id, arrival_time=service.time)
                    session_id = session.session_id
                    sessions[program_id] = session_id
                service.submit_call(
                    session_id,
                    payload["call_id"],
                    model_time=payload["model_time"],
                    prefill_tokens=payload["prefill_tokens"],
                    decode_tokens=payload["decode_tokens"],
                    parents=payload["parents"],
                    release_delay=payload["release_delay"],
                    thread_id=payload["thread_id"],
                    metadata=payload["metadata"],
                )
                conn.send({"event": "submitted", "program_id": program_id, "call_id": payload["call_id"]})
            elif command == "step":
                service.tick(payload["steps"])
                conn.send({"event": "stepped", "time": service.time})
            elif command == "run_until_idle":
                service.run_until_idle()
                conn.send({"event": "idle", "time": service.time})
            elif command == "drain":
                result = service.drain()
                conn.send({"event": "drained", "makespan": result.makespan})
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        conn.send({"event": "error", "error": repr(exc)})
