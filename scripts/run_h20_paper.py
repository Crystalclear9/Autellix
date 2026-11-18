"""Run paired SGLang and Autellix replays with H20 kernels and local weights."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


WORKLOADS = {
    "sharegpt": ("sharegpt-whole128.json", 48, (.12, .24, .48), "plas"),
    "bfcl": ("bfcl-captured64.json", 48, (.12, .24, .48), "plas"),
    "lats": ("lats-captured6.json", 6, (.035, .07), "atlas"),
    "mixed": ("mixed-paper.json", 12, (.06, .12), "atlas"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=WORKLOADS)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, help="comma-separated physical GPUs per replica; separate replicas with semicolons")
    parser.add_argument("--programs", type=int)
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kv-tokens", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--quantum-scale", type=float, default=2)
    parser.add_argument("--boundary-scale", type=float, default=128)
    parser.add_argument("--native-default", action="store_true", help="also measure upstream SGLang kernel and scheduler defaults")
    parser.add_argument("--routing-ablation", action="store_true", help="compare least-used and round-robin routing on multiple replicas")
    parser.add_argument("--label", help="separate a model or resource configuration in the summary")
    args = parser.parse_args()

    default_file, default_count, default_rates, policy = WORKLOADS[args.workload]
    model = args.model.resolve(strict=True)
    workload = (args.datasets / default_file).resolve(strict=True)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    groups = [group.split(",") for group in args.device.split(";")]
    stages = len(groups[0])
    devices = [device for group in groups for device in group]
    if (any(len(group) != stages for group in groups) or len(set(devices)) != len(devices)
            or not all(device.isdecimal() for device in devices)):
        parser.error("use distinct physical GPU indices and the same stage count for each replica")
    if args.routing_ablation and len(groups) < 2:
        parser.error("routing ablation requires multiple replicas")
    name = args.label or args.workload
    candidate = dict(name=policy, policy=policy, reserve=1, interval=8,
                     quantum_scale=args.quantum_scale, boundary_scale=args.boundary_scale)
    cases = [dict(name="native", policy="sglang-native", reserve=0), candidate]
    if args.native_default:
        cases.append(dict(name="native-default", policy="sglang-native", reserve=0, engine_args=dict(
            autellix_native_defaults=True, attention_backend=None, sampling_backend=None)))
    if args.routing_ablation:
        cases.extend(dict(candidate, name=f"{policy}-{routing}", routing=routing)
                     for routing in ("least-used", "round-robin"))
    plan = dict(model=str(model), device=args.device, workload=str(workload), workload_kind=name,
                workload_family=args.workload,
                programs=args.programs or default_count, rates=list(args.rates or default_rates),
                seed=args.seed, repeats=args.repeats, request_timeout=7200,
                engine_args=dict(dtype="bfloat16", context_length=32768, pp_size=stages,
                                 max_running_requests=args.batch_size, max_total_tokens=args.kv_tokens,
                                 max_prefill_tokens=32768, mem_fraction_static=.9, random_seed=0,
                                 attention_backend="triton", sampling_backend="pytorch",
                                 autellix_swap_space=16),
                cases=cases)
    if args.workload == "mixed":
        plan["equal_dataset_counts"] = True
    plan_file, results = run_dir / f"{name}-plan.json", run_dir / f"{name}.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    script_dir = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(script_dir / "run_sglang_matrix.py"), "--plan", str(plan_file),
                    "--output", str(results), "--resume"], check=True)
    subprocess.run([sys.executable, str(script_dir / "analyze_scheduler_trace.py"), str(results),
                    "--output", str(run_dir / f"{name}-scheduler.json")], check=True)
    for row in json.loads(results.read_text(encoding="utf-8")):
        state = Path(row["state_dir"])
        shutil.copytree(state, run_dir / "scheduler-state" / state.name, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
