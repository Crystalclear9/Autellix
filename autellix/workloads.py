from __future__ import annotations

import random
from itertools import count

from .models import CallSpec, ProgramSpec


def _sequential_program(
    program_id: str,
    model_times: list[int],
    *,
    arrival_time: int = 0,
    prefill_tokens: list[int] | None = None,
    decode_tokens: list[int] | None = None,
) -> ProgramSpec:
    calls: list[CallSpec] = []
    prefill_tokens = prefill_tokens or [0] * len(model_times)
    decode_tokens = decode_tokens or model_times
    parent: tuple[str, ...] = ()
    for idx, model_time in enumerate(model_times, start=1):
        call_id = f"{program_id}{idx}"
        calls.append(
            CallSpec(
                call_id=call_id,
                program_id=program_id,
                model_time=model_time,
                prefill_tokens=prefill_tokens[idx - 1],
                decode_tokens=decode_tokens[idx - 1],
                parents=parent,
            )
        )
        parent = (call_id,)
    return ProgramSpec(program_id=program_id, calls=tuple(calls), arrival_time=arrival_time)


def make_figure2_workload() -> list[ProgramSpec]:
    """Toy workload from Figure 2 of the paper."""

    return [
        _sequential_program("A", [4, 3, 1, 1]),
        _sequential_program("B", [3, 3, 4]),
        _sequential_program("C", [1, 2]),
        _sequential_program("D", [4]),
    ]


def _bounded_lognormal(rng: random.Random, mean: float, sigma: float, cap: int) -> int:
    value = int(rng.lognormvariate(0, sigma) * mean)
    return max(1, min(cap, value))


def _arrival_times(rng: random.Random, count_: int, arrival_rate: float) -> list[int]:
    arrivals: list[int] = []
    current = 0.0
    for _ in range(count_):
        current += rng.expovariate(arrival_rate)
        arrivals.append(int(current))
    return arrivals


def _make_sharegpt_program(rng: random.Random, program_id: str, arrival: int) -> ProgramSpec:
    call_count = min(80, max(1, int(rng.expovariate(1 / 5.5)) + 1))
    prefill: list[int] = []
    decode: list[int] = []
    model_times: list[int] = []
    for _ in range(call_count):
        p = _bounded_lognormal(rng, 256, 0.9, 150_000)
        d = _bounded_lognormal(rng, 277, 0.75, 4_000)
        prefill.append(p)
        decode.append(d)
        model_times.append(max(1, d // 35 + p // 1200))
    return _sequential_program(
        program_id,
        model_times,
        arrival_time=arrival,
        prefill_tokens=prefill,
        decode_tokens=decode,
    )


def _make_bfcl_program(rng: random.Random, program_id: str, arrival: int) -> ProgramSpec:
    call_count = min(70, max(1, int(rng.gauss(11, 4))))
    prefill: list[int] = []
    decode: list[int] = []
    model_times: list[int] = []
    for _ in range(call_count):
        p = _bounded_lognormal(rng, 735, 0.65, 40_000)
        d = _bounded_lognormal(rng, 34, 0.55, 1_000)
        prefill.append(p)
        decode.append(d)
        model_times.append(max(1, d // 20 + p // 900))
    return _sequential_program(
        program_id,
        model_times,
        arrival_time=arrival,
        prefill_tokens=prefill,
        decode_tokens=decode,
    )


def _make_lats_program(rng: random.Random, program_id: str, arrival: int) -> ProgramSpec:
    target_calls = min(400, max(40, int(rng.gauss(160, 48))))
    calls: list[CallSpec] = []
    id_counter = count(1)
    root_id = f"{program_id}{next(id_counter)}"
    root_prefill = _bounded_lognormal(rng, 467, 0.6, 20_000)
    root_decode = _bounded_lognormal(rng, 73, 0.6, 2_000)
    calls.append(
        CallSpec(
            call_id=root_id,
            program_id=program_id,
            model_time=max(1, root_decode // 25 + root_prefill // 1000),
            prefill_tokens=root_prefill,
            decode_tokens=root_decode,
        )
    )
    frontier = [root_id]
    while len(calls) < target_calls:
        width = min(target_calls - len(calls), rng.randint(2, 6))
        next_frontier: list[str] = []
        for _ in range(width):
            call_id = f"{program_id}{next(id_counter)}"
            parent = rng.choice(frontier)
            p = _bounded_lognormal(rng, 467, 0.6, 20_000)
            d = _bounded_lognormal(rng, 73, 0.6, 2_000)
            calls.append(
                CallSpec(
                    call_id=call_id,
                    program_id=program_id,
                    model_time=max(1, d // 25 + p // 1000),
                    prefill_tokens=p,
                    decode_tokens=d,
                    parents=(parent,),
                )
            )
            next_frontier.append(call_id)
        frontier = next_frontier or frontier
    return ProgramSpec(program_id=program_id, calls=tuple(calls), arrival_time=arrival)


def make_synthetic_workload(
    kind: str = "mixed",
    *,
    seed: int = 0,
    num_programs: int = 24,
    arrival_rate: float = 1.0,
) -> list[ProgramSpec]:
    """Generate deterministic synthetic workloads inspired by Section 6."""

    normalized = kind.lower()
    if normalized not in {"sharegpt", "bfcl", "lats", "mixed"}:
        raise ValueError("kind must be one of: sharegpt, bfcl, lats, mixed")
    rng = random.Random(seed)
    arrivals = _arrival_times(rng, num_programs, arrival_rate)
    programs: list[ProgramSpec] = []
    for idx, arrival in enumerate(arrivals):
        choice = normalized
        if normalized == "mixed":
            choice = rng.choice(["sharegpt", "bfcl", "lats"])
        program_id = f"P{idx:04d}"
        if choice == "sharegpt":
            programs.append(_make_sharegpt_program(rng, program_id, arrival))
        elif choice == "bfcl":
            programs.append(_make_bfcl_program(rng, program_id, arrival))
        else:
            programs.append(_make_lats_program(rng, program_id, arrival))
    return programs


def make_paper_workload(
    kind: str,
    *,
    seed: int = 0,
    num_programs: int = 24,
    arrival_rate: float = 1.0,
) -> list[ProgramSpec]:
    return make_synthetic_workload(
        kind,
        seed=seed,
        num_programs=num_programs,
        arrival_rate=arrival_rate,
    )
