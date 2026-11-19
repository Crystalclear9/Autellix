"""Measured real-backend DAG benchmark. No simulated service times are used."""
from __future__ import annotations

import argparse
import json
import statistics
import random
import copy
import math
import hashlib
import heapq
from importlib.metadata import version, PackageNotFoundError
from queue import Empty, SimpleQueue
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


def sample_trace(programs, count, rate, seed=0, *, equal_dataset_counts=False):
    """Sample programs with Poisson arrivals, or submit them all at time zero if rate is None."""
    validate_workload(programs)
    if count < 1 or (rate is not None and (not math.isfinite(rate) or rate <= 0)):
        raise ValueError("count and arrival rate must be positive")
    rng, now, trace = random.Random(seed), 0., []
    datasets = {}
    for program in programs:
        datasets.setdefault(program.get("dataset", "default"), []).append(program)
    pools = list(datasets.values())
    if equal_dataset_counts:
        if count % len(pools):
            raise ValueError("an equally mixed trace needs a multiple of the dataset count")
        dataset_order = pools * (count // len(pools))
        rng.shuffle(dataset_order)
    for i in range(count):
        if i and rate is not None:
            now += rng.expovariate(rate)
        pool = dataset_order[i] if equal_dataset_counts else rng.choice(pools)
        program = copy.deepcopy(rng.choice(pool))
        program["program_id"] = f"{program['program_id']}-{i}"
        program["arrival_time"] = now
        trace.append(program)
    return trace


def poisson_trace(programs, count, rate, seed=0, *, equal_dataset_counts=False):
    if rate is None:
        raise ValueError("Poisson arrivals require a positive rate")
    return sample_trace(programs, count, rate, seed, equal_dataset_counts=equal_dataset_counts)


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


def run_workload(engine, programs, request_timeout=600, progress_interval=0):
    validate_workload(programs)
    arrivals = {p["program_id"]: p.get("arrival_time", 0) for p in programs}
    remaining_calls = {p["program_id"]: len(p["calls"]) for p in programs}
    calls, parents, dependents, order = {}, {}, {}, {}
    for program in programs:
        pid = program["program_id"]
        for call in program["calls"]:
            key = pid, call["call_id"]
            order[key] = len(calls)
            calls[key] = call
            parents[key] = [(pid, parent) for parent in call.get("parents", [])]
            for parent in parents[key]:
                dependents.setdefault(parent, []).append(key)
    remaining_parents = {key: len(keys) for key, keys in parents.items()}
    results, pending, sessions = {}, {}, {}
    program_times, first_submissions = {}, {}
    completions, call_starts, critical = {}, {}, {}
    last_completion, critical_times, output_counts = {}, {}, {}
    completed_futures = SimpleQueue()
    deadlines = []
    start = time.monotonic()
    ready = [(start + arrivals[key[0]] + call.get("think_time", 0), order[key], key)
             for key, call in calls.items() if not parents[key]]
    heapq.heapify(ready)
    next_progress = start + progress_interval if progress_interval else math.inf
    try:
        while len(results) < len(calls):
            now = time.monotonic()
            if now >= next_progress:
                print(f"PROGRESS elapsed={now-start:.1f}s calls={len(results)} "
                      f"programs={len(program_times)}/{len(programs)} pending={len(pending)}", flush=True)
                next_progress = now + progress_interval
            while ready and ready[0][0] <= time.monotonic():
                _, _, key = heapq.heappop(ready)
                pid, call_id = key
                if pid not in sessions:
                    sessions[pid] = engine.start_session()
                call = calls[key]
                append_parents = parents[key] and call.get("append_parent_outputs", True)
                if "input_ids" in call and not append_parents:
                    inputs = {"input_ids": call["input_ids"]}
                else:
                    prompt = call["prompt"]
                    if append_parents:
                        prompt += "\nPrevious results:\n" + "\n".join(results[parent]["text"] for parent in parents[key])
                    inputs = {"prompt": prompt}
                call_starts[key] = time.monotonic()
                first_submissions.setdefault(pid, call_starts[key] - start)
                future = engine.submit(sessions[pid], **inputs, call_id=call_id,
                                       sampling={"temperature": 0, "max_tokens": call.get("max_tokens", 32),
                                                 **call.get("sampling", {})})
                pending[future] = key
                heapq.heappush(deadlines, (call_starts[key] + request_timeout, order[key], future))
                future.add_done_callback(lambda done: completed_futures.put((done, time.monotonic())))
            while deadlines and deadlines[0][2] not in pending:
                heapq.heappop(deadlines)
            deadline = deadlines[0][0] if deadlines else math.inf
            wake_at = min(deadline, next_progress, ready[0][0] if ready else math.inf)
            try:
                timeout = None if math.isinf(wake_at) else max(0., wake_at - time.monotonic())
                future, completed = completed_futures.get(timeout=timeout)
            except Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"backend request exceeded {request_timeout} seconds")
                continue
            key = pending.pop(future)
            pid = key[0]
            result = results[key] = future.result()
            completions[key] = completed
            critical[key] = (max((critical[parent] for parent in parents[key]), default=0.)
                             + completed - call_starts[key] + calls[key].get("think_time", 0))
            critical_times[pid] = max(critical_times.get(pid, 0.), critical[key])
            output_counts[pid] = output_counts.get(pid, 0) + result.get("completion_tokens", len(result["token_ids"]))
            last_completion[pid] = max(last_completion.get(pid, start), completed)
            for child in dependents.get(key, ()):
                remaining_parents[child] -= 1
                if remaining_parents[child] == 0:
                    at = max(completions[parent] for parent in parents[child]) + calls[child].get("think_time", 0)
                    heapq.heappush(ready, (at, order[child], child))
            remaining_calls[pid] -= 1
            if remaining_calls[pid] == 0:
                program_times[pid] = last_completion[pid] - start - arrivals[pid]
                engine.end_session(sessions.pop(pid))
        elapsed = time.monotonic() - start
        tokens = sum(output_counts.values())
        token_latencies = {pid: critical_times[pid] / count if count else None for pid, count in output_counts.items()}
        arrival_end = max(arrivals.values())
        completion_times = {pid: completed - start for pid, completed in last_completion.items()}
        return dict(measurement="real_backend", policy=engine.policy.policy,
                    backend=[r.backend for r in engine.replicas], model=engine.model,
                    elapsed_seconds=elapsed, completion_tokens=tokens,
                    output_tokens_per_second=tokens/elapsed,
                    completed_programs_per_second=len(programs)/elapsed,
                    arrival_horizon_seconds=arrival_end,
                    drain_seconds=max(0., elapsed-arrival_end),
                    completions_during_arrivals=sum(t <= arrival_end for t in completion_times.values()),
                    program_completion_times=completion_times,
                    first_submission_lag_seconds={pid: submitted - arrivals[pid] for pid, submitted in first_submissions.items()},
                    program_latency_seconds=distribution(program_times.values()),
                    program_token_latency_seconds=distribution(token_latencies.values()),
                    program_critical_path_seconds=critical_times,
                    program_token_latencies=token_latencies,
                    arrival_times=arrivals,
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
    parser.add_argument("--routing", choices=("program", "least-used", "round-robin"), default="program")
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
            native_sglang = policy == "sglang-native"
            if native_sglang:
                if args.backend != "sglang":
                    raise ValueError("native SGLang baseline requires --backend sglang")
                engine_args["autellix_native_baseline"] = True
            if native:
                if args.backend != "vllm":
                    raise ValueError("native vLLM baselines require --backend vllm")
                engine_args["autellix_mode"] = policy
            groups = args.device_groups.split(";") if args.device_groups else args.devices.split(",")
            replicas = [ReplicaConfig(args.backend, args.model, d.strip(), engine_args)
                        for d in groups if d.strip()]
            interval = args.native_opt_steps if policy == "vllm-opt-multistep" else (1 if native else args.schedule_interval)
            config = PolicyConfig(policy="fcfs" if native or native_sglang else policy, schedule_interval=interval,
                                  overprovision=0 if native or native_sglang else args.overprovision,
                                  implementation=args.implementation)
            with InferenceEngine(replicas, policy=config, routing=args.routing) as engine:
                sid = engine.start_session()
                engine.submit(sid, prompt="Hello", sampling=dict(max_tokens=2, temperature=0)).result(120)
                engine.end_session(sid)
                record = run_workload(engine, trace)
                record.update(policy=policy, arrival_rate=rate, seed=args.seed,
                              engine_args=engine_args, schedule_interval=interval, routing=args.routing,
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
