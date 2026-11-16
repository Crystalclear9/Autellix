"""Load one model across a GPU group and save generation and memory evidence."""
import argparse
import json
import time
from pathlib import Path

from autellix.runtime import InferenceEngine, PolicyConfig, ReplicaConfig
from autellix.runtime.telemetry import GPUMonitor, gpu_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--devices", default="0,1,2")
    p.add_argument("--engine-args", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--policy", default="fcfs")
    p.add_argument("--reserve", type=int, default=0)
    args = p.parse_args()
    defaults = PolicyConfig()
    policy = PolicyConfig(policy=args.policy, overprovision=args.reserve,
                          quanta=tuple(q * 64 for q in defaults.quanta),
                          boundaries=tuple(b * 1024 for b in defaults.boundaries))
    start = time.time()
    with InferenceEngine([ReplicaConfig("sglang", args.model, args.devices,
                                       json.loads(args.engine_args))], policy=policy) as engine:
        engine.wait_ready(1200)
        ready = time.time()
        memory = gpu_snapshot()
        sessions = [engine.start_session() for _ in range(6)]
        prompts = ["The capital of France is", "Explain what a GPU does in one sentence.",
                   "Write a Python function to add two numbers."] * 2
        futures = [engine.submit(s, prompt=q, sampling={"temperature": 0, "max_tokens": 48})
                   for s, q in zip(sessions, prompts)]
        with GPUMonitor() as monitor:
            results = [f.result(600) for f in futures]
        for session in sessions:
            engine.end_session(session)
        engine.wait_idle(30)
        Path(args.output).write_text(json.dumps(dict(
            model=args.model, devices=args.devices, engine_args=json.loads(args.engine_args),
            policy=args.policy, load_seconds=ready-start, generation_seconds=time.time()-ready,
            gpu_memory=memory, final_gpu_memory=gpu_snapshot(), results=results,
            gpu_samples=monitor.samples, gpu_sampling_errors=monitor.errors,
            state_dir=str(engine.state_dir), sessions_after=engine.table.describe()), indent=2))


if __name__ == "__main__":
    main()
