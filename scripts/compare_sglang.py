"""Paired native-SGLang/Autellix trials with matching GPU replica groups."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from importlib.metadata import version

from autellix.runtime import InferenceEngine, ReplicaConfig, PolicyConfig
from autellix.runtime.benchmark import load_workloads, run_workload, distribution, poisson_trace
from autellix.runtime.telemetry import GPUMonitor


def trial(model, device, engine_args, trace, arm, reserve=0, interval=8, quantum_scale=1., request_timeout=600,
          boundary_scale=None, beta=8., progress_interval=0, routing="program", backend="sglang",
          scheduler_trace=True, queue_levels=None, group_ready_calls=False, last_queue_fifo=False,
          max_active_programs=None):
    options = dict(engine_args)
    native = arm in {"sglang-native", "vllm", "vllm-opt", "vllm-opt-multistep"}
    if backend == "sglang":
        options["autellix_native_baseline"] = native
    elif native:
        options["autellix_mode"] = arm
        options.setdefault("observe_native", options.get("measure_timing", False))
    defaults = PolicyConfig()
    if queue_levels is not None:
        defaults = defaults.extend_queues(queue_levels)
    boundary_scale = quantum_scale if boundary_scale is None else boundary_scale
    policy = PolicyConfig(policy="fcfs" if native else arm,
                          schedule_interval=interval, overprovision=0 if native else reserve,
                          boundaries=tuple(x * boundary_scale for x in defaults.boundaries),
                          quanta=tuple(x * quantum_scale for x in defaults.quanta), beta=beta,
                          last_queue_fifo=last_queue_fifo)
    groups = device.split(";")
    with InferenceEngine([ReplicaConfig(backend, model, group, options) for group in groups],
                         policy=policy, routing=routing, scheduler_trace=scheduler_trace) as engine:
        sessions = [engine.start_session() for _ in groups]
        futures = [engine.submit(sid, prompt="Warmup.", sampling={"temperature": 0, "max_tokens": 4})
                   for sid in sessions]
        for sid, future in zip(sessions, futures):
            future.result(120)
            engine.end_session(sid)
        engine.wait_idle(30)
        devices = sorted({device.strip() for group in groups for device in group.split(",")})
        with GPUMonitor(devices=devices) as monitor:
            result = run_workload(engine, trace, request_timeout=request_timeout, progress_interval=progress_interval,
                                  group_ready_calls=group_ready_calls, max_active_programs=max_active_programs)
        if max_active_programs is not None:
            result["max_active_programs"] = max_active_programs
        result["gpu_samples"] = monitor.samples
        result["gpu_sampling_errors"] = monitor.errors
        result["device_groups"] = groups
        result["routing"] = routing
        if backend == "sglang":
            assert all(bool(c["metrics"].get("native_sglang")) == native
                       for c in result["calls"]), "unexpected scheduler accounting path"
        engine.wait_idle(30)
        assert engine.table.describe() == {}, "session cleanup incomplete"
        result["arm"] = arm
        result["state_dir"] = str(engine.state_dir)
        result["effective_engine_args"] = options
        result["hooks_installed"] = not native or (backend == "vllm" and bool(options["observe_native"]))
        result["overprovision"] = policy.overprovision
        result["schedule_interval"] = policy.schedule_interval
        result["quantum_scale"] = quantum_scale
        result["boundary_scale"] = boundary_scale
        result["beta"] = beta
        if queue_levels is not None:
            result["queue_levels"] = queue_levels
        if last_queue_fifo:
            result["last_queue_fifo"] = True
        # For these sequential programs the paper metric uses entire response
        # time, including initial admission/driver delay, divided by output count.
        counts = {p["program_id"]: 0 for p in trace}
        for call in result["calls"]:
            counts[call["workload_program"]] += call.get("completion_tokens", len(call["token_ids"]))
        result["response_seconds_per_output_token"] = {
            pid: latency/counts[pid] if counts[pid] else None for pid, latency in result["programs"].items()}
        result["paper_sequential_token_latency"] = distribution(result["response_seconds_per_output_token"].values())
        result["program_datasets"] = {p["program_id"]: p.get("dataset") for p in trace}
        result["output_hashes"] = {f'{c["workload_program"]}/{c["workload_call"]}':
            hashlib.sha256(c["text"].encode()).hexdigest() for c in result["calls"]}
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="0", help="one TP/PP group, or semicolon-separated replica groups")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--rates", default="0.1,1.0")
    p.add_argument("--arms", default="sglang-native,atlas")
    p.add_argument("--routing", choices=("program", "least-used", "round-robin"), default="program")
    p.add_argument("--reserve", type=int, default=0, help="Autellix reserves; native keeps zero, KV budget unchanged")
    p.add_argument("--interval", type=int, default=8)
    p.add_argument("--quantum-scale", type=float, default=1.)
    p.add_argument("--boundary-scale", type=float, help="service-history bins; defaults to quantum scale")
    p.add_argument("--beta", type=float, default=8.)
    p.add_argument("--programs", type=int, default=0, help="sample this many whole programs with replacement; 0 replays all")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--request-timeout", type=float, default=600)
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
            if args.programs:
                trace = poisson_trace(programs, args.programs, rate, args.seed + repeat)
            else:
                trace = copy.deepcopy(programs)
                rng = random.Random(args.seed+repeat); rng.shuffle(trace)
                arrival = 0.
                for i, program in enumerate(trace):
                    if i: arrival += rng.expovariate(rate)
                    program["arrival_time"] = arrival
            arms = args.arms.split(",")
            if repeat % 2: arms.reverse()
            for arm in arms:
                print(f"START rate={rate} repeat={repeat} arm={arm}", flush=True)
                start = time.time()
                result = trial(args.model, args.device, json.loads(args.engine_args), trace, arm,
                               args.reserve, args.interval, args.quantum_scale, args.request_timeout,
                               args.boundary_scale, args.beta, routing=args.routing)
                result.update(rate=rate, repeat=repeat, wall_start=start,
                              trace_seed=args.seed + repeat, sampled_programs=args.programs,
                              trace_sha256=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest(),
                              versions={p:version(p) for p in ["sglang","torch","transformers"]})
                records.append(result)
                output.write_text(json.dumps(records,indent=2),encoding="utf-8")
                print(f"DONE tokens/s={result['output_tokens_per_second']:.3f} "
                      f"mean_program_s={result['program_latency_seconds']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
