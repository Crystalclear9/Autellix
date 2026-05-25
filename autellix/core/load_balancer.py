from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import CallSpec, EngineState, ProcessEntry


class LoadBalancer:
    name = "base"

    def assign(
        self,
        call: CallSpec,
        engines: Sequence[EngineState],
        process_table: dict[str, ProcessEntry],
    ) -> EngineState:
        raise NotImplementedError

    @staticmethod
    def least_used(engines: Sequence[EngineState]) -> EngineState:
        return min(engines, key=lambda engine: (engine.workload, engine.engine_id))


@dataclass
class LocalityAwareLoadBalancer(LoadBalancer):
    """Autellix Algorithm 2.

    Short requests are load-balanced because their common system prompt
    usually dominates cache reuse. Long requests are pinned to a program's
    engine to model program-level KV locality.
    """

    token_threshold: int = 2048
    name: str = "autellix"

    def assign(
        self,
        call: CallSpec,
        engines: Sequence[EngineState],
        process_table: dict[str, ProcessEntry],
    ) -> EngineState:
        entry = process_table[call.program_id]
        if call.total_tokens <= self.token_threshold:
            engine = self.least_used(engines)
        elif entry.engine_id is not None:
            engine = engines[entry.engine_id]
        else:
            engine = self.least_used(engines)
            entry.engine_id = engine.engine_id
        entry.engine_ids.add(engine.engine_id)
        return engine


@dataclass
class LeastUsedLoadBalancer(LoadBalancer):
    name: str = "least-used"

    def assign(
        self,
        call: CallSpec,
        engines: Sequence[EngineState],
        process_table: dict[str, ProcessEntry],
    ) -> EngineState:
        engine = self.least_used(engines)
        process_table[call.program_id].engine_ids.add(engine.engine_id)
        return engine


@dataclass
class RoundRobinLoadBalancer(LoadBalancer):
    name: str = "round-robin"
    _next: int = 0

    def assign(
        self,
        call: CallSpec,
        engines: Sequence[EngineState],
        process_table: dict[str, ProcessEntry],
    ) -> EngineState:
        engine = engines[self._next % len(engines)]
        self._next += 1
        process_table[call.program_id].engine_ids.add(engine.engine_id)
        return engine


def make_load_balancer(
    name: str | None = None,
    *,
    token_threshold: int = 2048,
) -> LoadBalancer:
    normalized = (name or "autellix").lower().replace("_", "-")
    if normalized in {"autellix", "locality", "locality-aware"}:
        return LocalityAwareLoadBalancer(token_threshold=token_threshold)
    if normalized in {"least-used", "least", "leastused"}:
        return LeastUsedLoadBalancer()
    if normalized in {"round-robin", "rr", "roundrobin"}:
        return RoundRobinLoadBalancer()
    raise ValueError(f"unknown load balancer: {name}")
