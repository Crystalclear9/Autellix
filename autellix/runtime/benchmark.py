"""Measured real-backend DAG benchmark. No simulated service times are used."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

from .engine import InferenceEngine, ReplicaConfig
from .policy import PolicyConfig


def validate_workload(programs):
    names = set()
    for program in programs:
        pid = program["program_id"]
        if pid in names:
            raise ValueError("duplicate program ID")
        names.add(pid)
        calls = program["calls"]
        ids = {c["call_id"] for c in calls}
        if not calls or len(ids) != len(calls):
            raise ValueError("calls must be nonempty and uniquely named")
        ready = set()
        for call in calls:
            if not isinstance(call["prompt"], str):
                raise ValueError("every call must supply a real text prompt")
            if not set(call.get("parents", [])) <= ids:
                raise ValueError("unknown parent")
        while len(ready) < len(calls):
            new = {c["call_id"] for c in calls if set(c.get("parents", [])) <= ready} - ready
            if not new:
                raise ValueError("cyclic call dependencies")
            ready.update(new)
    if not programs:
        raise ValueError("workload must contain programs")


def run_workload(engine, programs):
    validate_workload(programs)
    results, pending, submitted = {}, {}, set()
    program_times = {}
    sessions = {p["program_id"]: engine.start_session() for p in programs}
    start = time.monotonic()
    try:
        while len(results) < sum(len(p["calls"]) for p in programs):
            for program in programs:
                pid = program["program_id"]
                for call in program["calls"]:
                    key = pid, call["call_id"]
                    parents = [(pid, parent) for parent in call.get("parents", [])]
                    if key in submitted or not all(parent in results for parent in parents):
                        continue
                    prompt = call["prompt"]
                    if parents:
                        prompt += "\nPrevious results:\n" + "\n".join(results[parent]["text"] for parent in parents)
                    future = engine.submit(sessions[pid], prompt=prompt, call_id=call["call_id"],
                                           sampling=dict(temperature=0, max_tokens=call.get("max_tokens", 32)))
                    pending[future] = key
                    submitted.add(key)
            done, _ = concurrent.futures.wait(pending, timeout=300,
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                raise TimeoutError("no backend completion in 300 seconds")
            for future in done:
                key = pending.pop(future)
                results[key] = future.result()
            for program in programs:
                pid = program["program_id"]
                if pid not in program_times and all((pid, c["call_id"]) in results for c in program["calls"]):
                    program_times[pid] = time.monotonic() - start
        elapsed = time.monotonic() - start
        tokens = sum(r.get("completion_tokens", len(r["token_ids"])) for r in results.values())
        latencies = sorted(program_times.values())
        percentile = lambda fraction: latencies[min(len(latencies)-1, int((len(latencies)-1)*fraction+.5))]
        return dict(measurement="real_backend", policy=engine.policy.policy,
                    backend=[r.backend for r in engine.replicas], model=engine.model,
                    elapsed_seconds=elapsed, completion_tokens=tokens,
                    output_tokens_per_second=tokens/elapsed,
                    program_latency_seconds=dict(mean=statistics.mean(latencies), p95=percentile(.95), p99=percentile(.99)),
                    programs=program_times,
                    calls=[dict(workload_program=pid, workload_call=cid, **r) for (pid, cid), r in results.items()])
    finally:
        for future in pending:
            future.cancel()
        for sid in sessions.values():
            engine.end_session(sid)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, help="JSON list of programs with prompt-bearing DAG calls")
    parser.add_argument("--policies", default="fcfs,mlfq,plas,atlas")
    parser.add_argument("--engine-args", default="{}")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", default="outputs/real/results.json")
    args = parser.parse_args(argv)
    programs = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    validate_workload(programs)
    records = []
    for policy in args.policies.split(","):
        replicas = [ReplicaConfig(args.backend, args.model, d.strip(), json.loads(args.engine_args))
                    for d in args.devices.split(",")]
        with InferenceEngine(replicas, policy=PolicyConfig(policy=policy.strip())) as engine:
            # Warm up with real inference, outside timed measurements.
            sid = engine.start_session()
            engine.submit(sid, prompt="Hello", sampling=dict(max_tokens=2, temperature=0)).result(120)
            engine.end_session(sid)
            records.append(run_workload(engine, programs))
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k != "calls"} for r in records], indent=2))


if __name__ == "__main__":
    main()
