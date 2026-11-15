"""Paired, alternated native-SGLang/Autellix trials on one GPU."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from importlib.metadata import version

from autellix.runtime import InferenceEngine, ReplicaConfig, PolicyConfig
from autellix.runtime.benchmark import load_workloads, run_workload, distribution


def trial(model, device, engine_args, trace, arm, reserve=0):
    options = dict(engine_args)
    options["autellix_native_baseline"] = arm == "sglang-native"
    policy = PolicyConfig(policy="fcfs" if arm == "sglang-native" else arm,
                          schedule_interval=8, overprovision=0 if arm == "sglang-native" else reserve)
    with InferenceEngine([ReplicaConfig("sglang", model, device, options)], policy=policy) as engine:
        sid = engine.start_session()
        engine.submit(sid, prompt="Warmup.", sampling={"temperature": 0, "max_tokens": 4}).result(120)
        engine.end_session(sid); engine.wait_idle(30)
        result = run_workload(engine, trace)
        assert all(bool(c["metrics"].get("native_sglang")) == (arm == "sglang-native")
                   for c in result["calls"]), "unexpected scheduler accounting path"
        engine.wait_idle(30)
        assert engine.table.describe() == {}, "session cleanup incomplete"
        result["arm"] = arm
        result["state_dir"] = str(engine.state_dir)
        result["effective_engine_args"] = options
        result["hooks_installed"] = arm != "sglang-native"
        result["overprovision"] = policy.overprovision
        result["schedule_interval"] = policy.schedule_interval
        # For these sequential programs the paper metric uses entire response
        # time, including initial admission/driver delay, divided by output count.
        counts = {p["program_id"]: 0 for p in trace}
        for call in result["calls"]:
            counts[call["workload_program"]] += call["completion_tokens"]
        result["response_seconds_per_output_token"] = {
            pid: latency/counts[pid] for pid, latency in result["programs"].items()}
        result["paper_sequential_token_latency"] = distribution(result["response_seconds_per_output_token"].values())
        result["output_hashes"] = {f'{c["workload_program"]}/{c["workload_call"]}':
            hashlib.sha256(c["text"].encode()).hexdigest() for c in result["calls"]}
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--rates", default="0.1,1.0")
    p.add_argument("--arms", default="sglang-native,atlas")
    p.add_argument("--reserve", type=int, default=0, help="Autellix reserves; native keeps zero, KV budget unchanged")
    p.add_argument("--engine-args", default='{"autellix_legacy_gpu":true,"context_length":2048,"max_running_requests":2,"max_total_tokens":4096,"mem_fraction_static":0.5,"random_seed":0}')
    args = p.parse_args()
    if args.repeats < 1 or args.reserve < 0:
        p.error("repeats must be positive and reserve must be nonnegative")
    programs = load_workloads([args.workload])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for rate in map(float, args.rates.split(",")):
        if rate <= 0:
            raise ValueError("arrival rate must be positive")
        for repeat in range(args.repeats):
            trace = copy.deepcopy(programs)
            rng = random.Random(100+repeat); rng.shuffle(trace)
            arrival = 0.
            for i, program in enumerate(trace):
                if i: arrival += rng.expovariate(rate)
                program["arrival_time"] = arrival
            arms = args.arms.split(",")
            if repeat % 2: arms.reverse()
            for arm in arms:
                print(f"START rate={rate} repeat={repeat} arm={arm}", flush=True)
                start = time.time()
                result = trial(args.model, args.device, json.loads(args.engine_args), trace, arm, args.reserve)
                result.update(rate=rate, repeat=repeat, wall_start=start,
                              trace_sha256=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest(),
                              versions={p:version(p) for p in ["sglang","torch","transformers"]})
                records.append(result)
                output.write_text(json.dumps(records,indent=2),encoding="utf-8")
                print(f"DONE tokens/s={result['output_tokens_per_second']:.3f} "
                      f"mean_program_s={result['program_latency_seconds']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
