"""Measured real-backend DAG benchmark. No simulated service times are used."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import random
import copy
import math
import hashlib
from importlib.metadata import version, PackageNotFoundError
import time
from pathlib import Path

from .engine import InferenceEngine, ReplicaConfig
from .policy import PolicyConfig


def validate_workload(programs):
    names = set()
    for program in programs:
        pid = program["program_id"]
        arrival = program.get("arrival_time", 0)
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("arrival_time must be finite and nonnegative")
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
            delay = call.get("think_time", 0)
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("think_time must be finite and nonnegative")
            if not set(call.get("parents", [])) <= ids:
                raise ValueError("unknown parent")
        while len(ready) < len(calls):
            new = {c["call_id"] for c in calls if set(c.get("parents", [])) <= ready} - ready
            if not new:
                raise ValueError("cyclic call dependencies")
            ready.update(new)
    if not programs:
        raise ValueError("workload must contain programs")


def poisson_trace(programs, count, rate, seed=0):
    validate_workload(programs)
    if count < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("count and arrival rate must be positive")
    rng, now, trace = random.Random(seed), 0., []
    datasets = {}
    for program in programs:
        datasets.setdefault(program.get("dataset", "default"), []).append(program)
    for i in range(count):
        if i:
            now += rng.expovariate(rate)
        # Mixed workloads sample datasets equally, then programs (not calls).
        program = copy.deepcopy(rng.choice(rng.choice(list(datasets.values()))))
        program["program_id"] = f"{program['program_id']}-{i}"
        program["arrival_time"] = now
        trace.append(program)
    return trace


def load_workloads(paths):
    """Load prompt-bearing program traces; never fabricate prompts from lengths."""
    programs = []
    for index, path in enumerate(map(Path, paths)):
        content = path.read_text(encoding="utf-8")
        data = ([json.loads(line) for line in content.splitlines() if line.strip()]
                if path.suffix.lower() == ".jsonl" else json.loads(content))
        if isinstance(data, dict):
            data = data["programs"]
        validate_workload(data)
        for program in data:
            program = copy.deepcopy(program)
            program.setdefault("dataset", path.stem)
            if len(paths) > 1:
                program["program_id"] = f"dataset-{index}:{program['program_id']}"
            programs.append(program)
    return programs


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(mean=None, p95=None, p99=None, count=0)
    percentile = lambda p: values[min(len(values)-1, math.ceil(len(values)*p)-1)]
    return dict(mean=statistics.mean(values), p95=percentile(.95), p99=percentile(.99), count=len(values))


def run_workload(engine, programs):
    validate_workload(programs)
    results, pending, submitted = {}, {}, set()
    program_times = {}
    sessions, completions, call_starts, critical = {}, {}, {}, {}
    start = time.monotonic()
    try:
        while len(results) < sum(len(p["calls"]) for p in programs):
            for program in programs:
                pid = program["program_id"]
                arrival = start + program.get("arrival_time", 0)
                if time.monotonic() < arrival:
                    continue
                if pid not in sessions:
                    sessions[pid] = engine.start_session()
                for call in program["calls"]:
                    key = pid, call["call_id"]
                    parents = [(pid, parent) for parent in call.get("parents", [])]
                    if key in submitted or not all(parent in results for parent in parents):
                        continue
                    ready_at = max((completions[parent] for parent in parents), default=arrival) + call.get("think_time", 0)
                    if time.monotonic() < ready_at:
                        continue
                    prompt = call["prompt"]
                    if parents and call.get("append_parent_outputs", True):
                        prompt += "\nPrevious results:\n" + "\n".join(results[parent]["text"] for parent in parents)
                    call_starts[key] = time.monotonic()
                    future = engine.submit(sessions[pid], prompt=prompt, call_id=call["call_id"],
                                           sampling={"temperature": 0, "max_tokens": call.get("max_tokens", 32),
                                                     **call.get("sampling", {})})
                    pending[future] = key
                    future.add_done_callback(lambda _, k=key: completions.setdefault(k, time.monotonic()))
                    submitted.add(key)
            # Short bounded waits keep scheduled arrivals and external-tool
            # delays independent of the duration of outstanding generations.
            done, _ = concurrent.futures.wait(pending, timeout=.01,
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                if any(time.monotonic() - call_starts[key] > 600 for key in pending.values()):
                    raise TimeoutError("backend request exceeded 600 seconds")
                if not pending:
                    time.sleep(.001)
                continue
            for future in done:
                key = pending.pop(future)
                results[key] = future.result()
                completions.setdefault(key, time.monotonic())
            for program in programs:
                pid = program["program_id"]
                if pid not in program_times and all((pid, c["call_id"]) in results for c in program["calls"]):
                    program_times[pid] = max(completions[(pid, c["call_id"])] for c in program["calls"]) - start - program.get("arrival_time", 0)
                    unresolved = list(program["calls"])
                    while unresolved:
                        for call in unresolved[:]:
                            key = pid, call["call_id"]
                            parents = [(pid, parent) for parent in call.get("parents", [])]
                            if all(parent in critical for parent in parents):
                                critical[key] = max((critical[p] for p in parents), default=0) + completions[key] - call_starts[key] + call.get("think_time", 0)
                                unresolved.remove(call)
                    engine.end_session(sessions[pid])
        elapsed = time.monotonic() - start
        tokens = sum(r.get("completion_tokens", len(r["token_ids"])) for r in results.values())
        token_latencies, critical_times = {}, {}
        for program in programs:
            pid = program["program_id"]
            critical_times[pid] = max(critical[(pid, c["call_id"])] for c in program["calls"])
            count = sum(results[(pid, c["call_id"])].get("completion_tokens", len(results[(pid, c["call_id"])] ["token_ids"])) for c in program["calls"])
            token_latencies[pid] = critical_times[pid] / count if count else None
        return dict(measurement="real_backend", policy=engine.policy.policy,
                    backend=[r.backend for r in engine.replicas], model=engine.model,
                    elapsed_seconds=elapsed, completion_tokens=tokens,
                    output_tokens_per_second=tokens/elapsed,
                    program_latency_seconds=distribution(program_times.values()),
                    program_token_latency_seconds=distribution(token_latencies.values()),
                    program_critical_path_seconds=critical_times,
                    program_token_latencies=token_latencies,
                    arrival_times={p["program_id"]: p.get("arrival_time", 0) for p in programs},
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
    parser.add_argument("--workload", required=True, nargs="+", help="JSON/JSONL prompt-bearing program traces; multiple files create a mixed workload")
    parser.add_argument("--policies", default="fcfs,mlfq,plas,atlas")
    parser.add_argument("--engine-args", default="{}")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--device-groups", help="semicolon-separated replica GPU groups, e.g. 0,1;2,3")
    parser.add_argument("--output", default="outputs/real/results.json")
    parser.add_argument("--arrival-rates", help="comma-separated Poisson program arrival rates per second")
    parser.add_argument("--programs", type=int, default=100, help="sampled programs per rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule-interval", type=int, default=8)
    parser.add_argument("--overprovision", type=int, default=1)
    parser.add_argument("--implementation", choices=("paper", "compat"), default="paper")
    parser.add_argument("--native-opt-steps", type=int, default=8)
    args = parser.parse_args(argv)
    programs = load_workloads(args.workload)
    validate_workload(programs)
    records = []
    rates = [float(x) for x in args.arrival_rates.split(",")] if args.arrival_rates else [None]
    for rate in rates:
        trace = poisson_trace(programs, args.programs, rate, args.seed) if rate is not None else programs
        for policy in map(str.strip, args.policies.split(",")):
            engine_args = json.loads(args.engine_args)
            native = policy in {"vllm", "vllm-opt", "vllm-opt-multistep"}
            if native:
                if args.backend != "vllm":
                    raise ValueError("native vLLM baselines require --backend vllm")
                engine_args["autellix_mode"] = policy
            groups = args.device_groups.split(";") if args.device_groups else args.devices.split(",")
            replicas = [ReplicaConfig(args.backend, args.model, d.strip(), engine_args)
                        for d in groups if d.strip()]
            interval = args.native_opt_steps if policy == "vllm-opt-multistep" else (1 if native else args.schedule_interval)
            config = PolicyConfig(policy="fcfs" if native else policy, schedule_interval=interval,
                                  overprovision=0 if native else args.overprovision,
                                  implementation=args.implementation)
            with InferenceEngine(replicas, policy=config) as engine:
                sid = engine.start_session()
                engine.submit(sid, prompt="Hello", sampling=dict(max_tokens=2, temperature=0)).result(120)
                engine.end_session(sid)
                record = run_workload(engine, trace)
                record.update(policy=policy, arrival_rate=rate, seed=args.seed,
                              engine_args=engine_args, schedule_interval=interval,
                              overprovision=config.overprovision,
                              implementation=config.implementation,
                              workload_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest())
                record["versions"] = {}
                for package in ("autellix", args.backend, "torch", "transformers"):
                    try:
                        record["versions"][package] = version(package)
                    except PackageNotFoundError:
                        record["versions"][package] = None
                records.append(record)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k != "calls"} for r in records], indent=2))


if __name__ == "__main__":
    main()
