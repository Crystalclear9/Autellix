"""Autellix scheduling simulator.

This package is a lightweight, CPU-only reproduction of the scheduling
algorithms described in the Autellix paper. It models LLM calls as timed
jobs and focuses on program-aware scheduling rather than real inference.
"""

from .core import LocalityAwareLoadBalancer, Simulator, make_load_balancer, make_scheduler
from .experiments import (
    ExperimentRunner,
    load_programs_from_file,
    make_baseline,
    make_figure2_workload,
    make_paper_workload,
    make_synthetic_workload,
    programs_from_records,
    workload_analysis,
)
from .frontend import (
    AsyncMultiLLMEngine,
    AutellixClient,
    AutellixService,
    Session,
    SimulatedChatResponse,
    SimulatedRequestFuture,
)

__all__ = [
    "AsyncMultiLLMEngine",
    "AutellixClient",
    "AutellixService",
    "ExperimentRunner",
    "LocalityAwareLoadBalancer",
    "Session",
    "SimulatedChatResponse",
    "SimulatedRequestFuture",
    "Simulator",
    "make_baseline",
    "make_figure2_workload",
    "make_load_balancer",
    "make_paper_workload",
    "make_scheduler",
    "make_synthetic_workload",
    "load_programs_from_file",
    "programs_from_records",
    "workload_analysis",
]
