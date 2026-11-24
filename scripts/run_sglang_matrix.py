"""Run a recorded load/ablation plan, with identical traces for every case."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
import traceback
from importlib.metadata import version
from pathlib import Path

from compare_sglang import trial
from summarize_paper_suite import paper_token_latencies
from autellix.runtime.benchmark import distribution, load_workloads, sample_trace


def hash_trace(trace):
    # Match json.dumps(trace, sort_keys=True) without buffering the entire trace.
    digest = hashlib.sha256(b"[")
    for index, program in enumerate(trace):
        if index:
            digest.update(b", ")
        digest.update(json.dumps(program, sort_keys=True).encode())
    digest.update(b"]")
    return digest.hexdigest()


def unique_trace(programs, count, rate, seed, *, equal_dataset_counts=False):
    rng = random.Random(seed)
    ordered = list(programs)
    rng.shuffle(ordered)
    if count:
        if count > len(ordered):
            raise ValueError(f"requested {count} distinct programs from {len(ordered)}")
        if equal_dataset_counts:
            pools = {}
            for program in ordered:
                pools.setdefault(program.get("dataset", "default"), []).append(program)
            per_dataset, remainder = divmod(count, len(pools))
            if remainder or any(len(pool) < per_dataset for pool in pools.values()):
                raise ValueError("equal sampling requires enough distinct programs in every dataset")
            selected = {p["program_id"] for pool in pools.values() for p in pool[:per_dataset]}
            ordered = [program for program in ordered if program["program_id"] in selected]
        else:
            ordered = ordered[:count]
    trace = copy.deepcopy(ordered)
    arrival = 0.
    for i, program in enumerate(trace):
        if i and rate is not None:
            arrival += rng.expovariate(rate)
        program["arrival_time"] = arrival
    return trace


def timed_trace(programs, rate, seed, seconds, *, equal_dataset_counts=False):
    """Sample distinct programs for a fixed-duration Poisson arrival window."""
    if rate is None or not math.isfinite(rate) or rate <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Poisson arrival rate and duration must be finite and positive")
    # The arrival process is independent of program selection.
    rng = random.Random(seed)
    arrivals = []
    arrival = rng.expovariate(rate)
    while arrival < seconds:
        arrivals.append(arrival)
        arrival += rng.expovariate(rate)
    if not arrivals:
        raise ValueError("the arrival window contains no programs")
    count = len(arrivals)
    datasets = {p.get("dataset", "default") for p in programs}
    sampled_count = math.ceil(count / len(datasets)) * len(datasets) if equal_dataset_counts else count
    trace = unique_trace(programs, sampled_count, None, f"{seed}:programs",
                         equal_dataset_counts=equal_dataset_counts)
    # Poisson counts need not be divisible by the number of datasets. Remove
    # at most one sample per dataset and keep all of the arrival timestamps.
    removed = set()
    for index in range(len(trace) - 1, -1, -1):
        if len(trace) == count:
            break
        dataset = trace[index].get("dataset", "default")
        if dataset not in removed:
            del trace[index]
            removed.add(dataset)
    for program, at in zip(trace, arrivals):
        program["arrival_time"] = at
    return trace


def completed_trials(records, plan, versions):
    completed = {}
    for row in records:
        if row["versions"] != versions:
            raise ValueError("resume requires the same backend versions")
        if row["plan"] != plan:
            def common(p):
                return {k: v for k, v in p.items() if k not in {"cases", "rates"}}
            def case(p):
                selected = [c for c in p["cases"] if c["name"] == row["condition"]
                            and row["rate"] in c.get("rates", p["rates"])]
                return [{k: v for k, v in c.items() if k != "rates"} for c in selected]
            if (common(row["plan"]) != common(plan) or row["rate"] not in plan.get("rates", [])
                    or len(case(plan)) != 1 or case(row["plan"]) != case(plan)):
                raise ValueError("resume requires unchanged resources, workload and completed case settings")
        key = row["condition"], row["rate"], row["repeat"]
        if key in completed:
            raise ValueError(f"duplicate completed trial: {key}")
        completed[key] = row
    return completed


def write_checkpoint(path, records):
    checkpoint = path.with_suffix(path.suffix + ".tmp")
    checkpoint.write_text(json.dumps(records, indent=2), encoding="utf-8")
    checkpoint.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="reuse completed trials with unchanged inputs and settings")
    parser.add_argument("--keep-going", action="store_true", help="record failed trials and continue on the assigned GPU")
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    backend = plan.get("backend", "sglang")
    device = str(plan.get("device", "0"))
    if args.keep_going and not device.isdecimal():
        parser.error("--keep-going requires one physical GPU index")
    if "layer_partition" in plan:
        os.environ["SGLANG_PP_LAYER_PARTITION"] = ",".join(map(str, plan["layer_partition"]))
    elif plan["engine_args"].get("pp_size", 1) == 1:
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
    programs = load_workloads([plan["workload"]])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    versions = {name: version(name) for name in (backend, "torch", "transformers")}
    records = json.loads(destination.read_text()) if args.resume and destination.exists() else []
    completed = completed_trials(records, plan, versions)
    failures_path = destination.with_name(f"{destination.stem}-failures.json")
    failures = json.loads(failures_path.read_text()) if args.resume and failures_path.exists() else []
    failed = False
    for rate in plan["rates"]:
        for repeat in range(plan.get("repeats", 1)):
            seed = plan.get("seed", 100) + repeat
            arrival_seconds = plan.get("arrival_seconds")
            if arrival_seconds is not None:
                trace = timed_trace(programs, rate, seed, arrival_seconds,
                                    equal_dataset_counts=plan.get("equal_dataset_counts", False))
            elif plan.get("programs") and not plan.get("unique_programs"):
                trace = sample_trace(programs, plan["programs"], rate, seed,
                                      equal_dataset_counts=plan.get("equal_dataset_counts", False))
            else:
                trace = unique_trace(programs, plan.get("programs"), rate, seed,
                                     equal_dataset_counts=plan.get("equal_dataset_counts", False))
            cases = plan["cases"]
            trace_hash = hash_trace(trace)
            offset = repeat % len(cases)
            cases = cases[offset:] + cases[:offset]
            for case in cases:
                if "rates" in case and rate not in case["rates"]:
                    continue
                previous = completed.get((case["name"], rate, repeat))
                if previous is not None:
                    if previous["trace_sha256"] != trace_hash:
                        raise ValueError("completed trial has a different input trace")
                    print(f"REUSE rate={rate} repeat={repeat} case={case['name']}", flush=True)
                    continue
                print(f"START rate={rate} repeat={repeat} case={case['name']}", flush=True)
                started = time.time()
                try:
                    row = trial(plan["model"], device, {**plan["engine_args"], **case.get("engine_args", {})},
                                trace, case["policy"], reserve=case.get("reserve", 0),
                                interval=case.get("interval", 8), quantum_scale=case.get("quantum_scale", 1.),
                                boundary_scale=case.get("boundary_scale"), beta=case.get("beta", 8.),
                                request_timeout=plan.get("request_timeout", 3600),
                                progress_interval=plan.get("progress_interval", 30),
                                routing=case.get("routing", "program"), backend=backend,
                                scheduler_trace=plan.get("scheduler_trace", True),
                                queue_levels=case.get("queue_levels"))
                except Exception as exc:
                    if not args.keep_going:
                        raise
                    failed = True
                    failures.append(dict(condition=case["name"], rate=rate, repeat=repeat, seed=seed,
                                         wall_start=started, trace_sha256=trace_hash, versions=versions,
                                         plan=plan, error=f"{type(exc).__name__}: {exc}"))
                    write_checkpoint(failures_path, failures)
                    traceback.print_exc()
                    print(f"FAILED rate={rate} repeat={repeat} case={case['name']}", flush=True)
                    pids = subprocess.check_output(["nvidia-smi", "-i", device, "--query-compute-apps=pid",
                                                    "--format=csv,noheader,nounits"], text=True)
                    if any(pid.strip().isdecimal() for pid in pids.splitlines()):
                        raise RuntimeError(f"GPU {device} is still occupied after failed trial") from exc
                    continue
                row.update(condition=case["name"], rate=rate, repeat=repeat, seed=seed,
                           wall_start=started, plan=plan,
                           trace_sha256=trace_hash, versions=versions)
                if arrival_seconds is not None:
                    row.update(arrival_horizon_seconds=arrival_seconds,
                               drain_seconds=max(0., row["elapsed_seconds"] - arrival_seconds),
                               completions_during_arrivals=sum(at <= arrival_seconds for at in
                                                               row["program_completion_times"].values()))
                if plan.get("compact_calls"):
                    for call in row["calls"]:
                        call.pop("text", None)
                        call.pop("token_ids", None)
                records.append(row)
                write_checkpoint(destination, records)
                token_times, _ = paper_token_latencies(row)
                mean = distribution(token_times.values())["mean"]
                token_latency = f"{mean:.6f}" if mean is not None else "n/a"
                print(f"DONE case={case['name']} tok/s={row['output_tokens_per_second']:.3f} "
                      f"latency={row['program_latency_seconds']['mean']:.3f} "
                      f"s/token={token_latency}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
