"""Run a recorded load/ablation plan, with identical traces for every case."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from importlib.metadata import version
from pathlib import Path

from compare_sglang import trial
from autellix.runtime.benchmark import load_workloads, poisson_trace


def completed_trials(records, plan, versions):
    completed = {}
    for row in records:
        if row["plan"] != plan or row["versions"] != versions:
            raise ValueError("resume requires the same experiment plan and backend versions")
        key = row["condition"], row["rate"], row["repeat"]
        if key in completed:
            raise ValueError(f"duplicate completed trial: {key}")
        completed[key] = row
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="reuse completed trials from this exact plan")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if "layer_partition" in plan:
        os.environ["SGLANG_PP_LAYER_PARTITION"] = ",".join(map(str, plan["layer_partition"]))
    elif plan["engine_args"].get("pp_size", 1) == 1:
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
    programs = load_workloads([plan["workload"]])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    versions = {name: version(name) for name in ["sglang", "torch", "transformers"]}
    records = json.loads(destination.read_text()) if args.resume and destination.exists() else []
    completed = completed_trials(records, plan, versions)
    for rate in plan["rates"]:
        for repeat in range(plan.get("repeats", 1)):
            seed = plan.get("seed", 100) + repeat
            if plan.get("programs"):
                trace = poisson_trace(programs, plan["programs"], rate, seed)
            else:
                trace = copy.deepcopy(programs)
                rng = random.Random(seed)
                rng.shuffle(trace)
                arrival = 0.
                for i, program in enumerate(trace):
                    if i:
                        arrival += rng.expovariate(rate)
                    program["arrival_time"] = arrival
            cases = plan["cases"]
            trace_hash = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
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
                row = trial(plan["model"], plan.get("device", "0"),
                            {**plan["engine_args"], **case.get("engine_args", {})},
                            trace, case["policy"], reserve=case.get("reserve", 0),
                            interval=case.get("interval", 8), quantum_scale=case.get("quantum_scale", 1.),
                            boundary_scale=case.get("boundary_scale"), beta=case.get("beta", 8.),
                            request_timeout=plan.get("request_timeout", 3600), progress_interval=30,
                            routing=case.get("routing", "program"))
                row.update(condition=case["name"], rate=rate, repeat=repeat, seed=seed,
                           wall_start=started, plan=plan,
                           trace_sha256=trace_hash, versions=versions)
                records.append(row)
                destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
                print(f"DONE case={case['name']} tok/s={row['output_tokens_per_second']:.3f} "
                      f"latency={row['program_latency_seconds']['mean']:.3f} "
                      f"s/token={row['paper_sequential_token_latency']['mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
