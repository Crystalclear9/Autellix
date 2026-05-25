from __future__ import annotations

from dataclasses import dataclass

from ..core.execution import ExecutionModel, make_execution_model
from ..core.load_balancer import LoadBalancer, make_load_balancer
from ..core.models import ProgramSpec
from ..core.schedulers import Scheduler, make_scheduler
from ..core.simulator import is_sequential_program


@dataclass(frozen=True)
class Baseline:
    name: str
    scheduler: Scheduler
    load_balancer: LoadBalancer
    execution_model: ExecutionModel
    schedule_interval: int = 1
    overprovision: int = 0


def make_baseline(
    name: str,
    *,
    programs: list[ProgramSpec] | tuple[ProgramSpec, ...] | None = None,
    token_threshold: int = 2048,
    schedule_interval: int = 1,
    load_balancer_name: str | None = None,
) -> Baseline:
    normalized = name.lower().replace("_", "-")
    override_load_balancer = (
        make_load_balancer(load_balancer_name, token_threshold=token_threshold)
        if load_balancer_name is not None
        else None
    )
    if normalized == "vllm":
        return Baseline(
            name="vllm",
            scheduler=make_scheduler("fcfs"),
            load_balancer=override_load_balancer
            or make_load_balancer("least-used", token_threshold=token_threshold),
            execution_model=make_execution_model("vllm"),
            schedule_interval=1,
        )
    if normalized in {"vllm-opt", "vllmopt"}:
        return Baseline(
            name="vllm-opt",
            scheduler=make_scheduler("fcfs"),
            load_balancer=override_load_balancer
            or make_load_balancer("least-used", token_threshold=token_threshold),
            execution_model=make_execution_model("vllm-opt"),
            schedule_interval=schedule_interval,
        )
    if normalized == "mlfq":
        return Baseline(
            name="mlfq",
            scheduler=make_scheduler("mlfq"),
            load_balancer=override_load_balancer
            or make_load_balancer("least-used", token_threshold=token_threshold),
            execution_model=make_execution_model("mlfq"),
            schedule_interval=schedule_interval,
        )
    if normalized in {"autellix", "plas"}:
        scheduler_name = "plas"
        if normalized == "autellix" and programs is not None:
            scheduler_name = (
                "plas"
                if all(is_sequential_program(program) for program in programs)
                else "atlas"
            )
        return Baseline(
            name="autellix",
            scheduler=make_scheduler(scheduler_name),
            load_balancer=override_load_balancer
            or make_load_balancer("autellix", token_threshold=token_threshold),
            execution_model=make_execution_model("autellix"),
            schedule_interval=schedule_interval,
            overprovision=1,
        )
    if normalized == "atlas":
        return Baseline(
            name="atlas",
            scheduler=make_scheduler("atlas"),
            load_balancer=override_load_balancer
            or make_load_balancer("autellix", token_threshold=token_threshold),
            execution_model=make_execution_model("autellix"),
            schedule_interval=schedule_interval,
            overprovision=1,
        )
    if normalized in {"round-robin", "roundrobin", "rr"}:
        return Baseline(
            name="round-robin",
            scheduler=make_scheduler("round-robin"),
            load_balancer=override_load_balancer
            or make_load_balancer("round-robin", token_threshold=token_threshold),
            execution_model=make_execution_model("autellix"),
            schedule_interval=1,
        )
    if normalized in {"least-used", "leastused", "least"}:
        return Baseline(
            name="least-used",
            scheduler=make_scheduler("plas"),
            load_balancer=override_load_balancer
            or make_load_balancer("least-used", token_threshold=token_threshold),
            execution_model=make_execution_model("autellix"),
            schedule_interval=schedule_interval,
        )
    if normalized == "srpt":
        return Baseline(
            name="srpt",
            scheduler=make_scheduler("srpt"),
            load_balancer=override_load_balancer
            or make_load_balancer("autellix", token_threshold=token_threshold),
            execution_model=make_execution_model("autellix"),
            schedule_interval=1,
        )
    if normalized in {"fcfs"}:
        return Baseline(
            name="fcfs",
            scheduler=make_scheduler("fcfs"),
            load_balancer=override_load_balancer
            or make_load_balancer("least-used", token_threshold=token_threshold),
            execution_model=make_execution_model("fixed"),
            schedule_interval=1,
        )
    raise ValueError(f"unknown baseline: {name}")
