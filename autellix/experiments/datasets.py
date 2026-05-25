from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..core.models import CallSpec, ProgramSpec


def load_programs_from_file(path: str | Path, *, default_kind: str = "dataset") -> list[ProgramSpec]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("programs", data.get("calls", []))
    elif suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported dataset format: {source.suffix}")
    return programs_from_records(rows, default_kind=default_kind)


def programs_from_records(
    records: Iterable[dict[str, Any]],
    *,
    default_kind: str = "dataset",
) -> list[ProgramSpec]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(records):
        if "calls" in row and isinstance(row["calls"], list):
            program_id = str(row.get("program_id") or row.get("id") or f"P{idx:04d}")
            for call in row["calls"]:
                item = dict(call)
                item.setdefault("program_id", program_id)
                grouped[program_id].append(item)
            continue
        program_id = str(row.get("program_id") or row.get("conversation_id") or row.get("id") or f"P{idx:04d}")
        grouped[program_id].append(dict(row))

    programs: list[ProgramSpec] = []
    for program_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(_get(row, "index", "turn", "call_index", default=0)))
        arrival = int(_get(rows[0], "arrival_time", "arrival", default=0))
        calls: list[CallSpec] = []
        previous_call_id: str | None = None
        for idx, row in enumerate(rows, start=1):
            call_id = str(_get(row, "call_id", "request_id", "id", default=f"{program_id}_{idx}"))
            parents = _parents(row, previous_call_id)
            prefill = int(_get(row, "prefill_tokens", "prompt_tokens", "input_tokens", default=0))
            decode = int(_get(row, "decode_tokens", "completion_tokens", "output_tokens", default=0))
            model_time = int(
                _get(
                    row,
                    "model_time",
                    "decode_steps",
                    "duration",
                    default=max(1, decode or prefill // 512 or 1),
                )
            )
            metadata = {
                "source_kind": row.get("kind", default_kind),
                **{k: v for k, v in row.items() if k not in _CALL_FIELDS},
            }
            calls.append(
                CallSpec(
                    call_id=call_id,
                    program_id=program_id,
                    model_time=max(1, model_time),
                    prefill_tokens=max(0, prefill),
                    decode_tokens=max(0, decode),
                    parents=parents,
                    release_delay=int(_get(row, "release_delay", default=0)),
                    submit_time=_optional_int(_get(row, "submit_time", default=None)),
                    thread_id=_optional_str(_get(row, "thread_id", "thread", default=None)),
                    metadata=metadata,
                )
            )
            previous_call_id = call_id
        programs.append(ProgramSpec(program_id, tuple(calls), arrival_time=arrival))
    return programs


def workload_analysis(programs: Iterable[ProgramSpec]) -> dict[str, float | int]:
    programs = list(programs)
    calls = [call for program in programs for call in program.calls]
    prefill = [call.prefill_tokens for call in calls]
    decode = [call.decode_tokens for call in calls]
    call_counts = [len(program.calls) for program in programs]
    return {
        "programs": len(programs),
        "calls": len(calls),
        "mean_prefill_tokens": _mean(prefill),
        "mean_decode_tokens": _mean(decode),
        "mean_calls_per_program": _mean(call_counts),
        "max_calls_per_program": max(call_counts) if call_counts else 0,
    }


_CALL_FIELDS = {
    "program_id",
    "conversation_id",
    "call_id",
    "request_id",
    "id",
    "index",
    "turn",
    "call_index",
    "arrival_time",
    "arrival",
    "prefill_tokens",
    "prompt_tokens",
    "input_tokens",
    "decode_tokens",
    "completion_tokens",
    "output_tokens",
    "model_time",
    "decode_steps",
    "duration",
    "parents",
    "parent_id",
    "parent",
    "release_delay",
    "submit_time",
    "thread_id",
    "thread",
}


def _get(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def _parents(row: dict[str, Any], previous_call_id: str | None) -> tuple[str, ...]:
    raw = _get(row, "parents", default=None)
    if raw:
        if isinstance(raw, str):
            return tuple(part.strip() for part in raw.split("|") if part.strip())
        return tuple(str(part) for part in raw)
    parent = _get(row, "parent_id", "parent", default=None)
    if parent:
        return (str(parent),)
    mode = str(row.get("dependency", "chain")).lower()
    if mode == "root" or previous_call_id is None:
        return ()
    return (previous_call_id,)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
