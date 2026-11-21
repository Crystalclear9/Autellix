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
    parser.add_argument("--host-kv-gib", type=float, help="Autellix host KV budget per engine or pipeline stage")
    parser.add_argument("--host-kv-recompute", action="store_true", help="recompute KV when the single-stage host budget fills")
    parser.add_argument("--cache-paused-kv", action="store_true", help="cache paused prefixes in the GPU radix tree")
    parser.add_argument("--publish-activity-at-window", action="store_true", help="publish the shared activity view at window boundaries")
    parser.add_argument("--refill-idle-slots", action="store_true", help="fill vacant batch slots after the prepared reserve is exhausted")
    parser.add_argument("--resident-reserve-only", action="store_true", help="choose reserve requests from KV already on GPU")
    parser.add_argument("--optimized-kernels", action="store_true", help="use SGLang's selected attention kernels and CUDA graphs")
    parser.add_argument("--native-default", type=float, nargs="*", metavar="RATE",
                        help="also measure upstream SGLang defaults at the listed rates, or at all rates")
    parser.add_argument("--fcfs-control", type=float, nargs="+", metavar="RATE",
                        help="measure Autellix hooks with FCFS at the listed replay rates")
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
    candidate_options = {}
    if args.host_kv_gib is not None:
        candidate_options["autellix_swap_space"] = args.host_kv_gib
    if args.host_kv_recompute:
        candidate_options["autellix_host_kv_recompute"] = True
    if args.cache_paused_kv:
        candidate_options["autellix_cache_paused_kv"] = True
    if args.publish_activity_at_window:
        candidate_options["autellix_window_activity"] = True
    if args.refill_idle_slots:
        candidate_options["autellix_refill_idle_slots"] = True
    if args.resident_reserve_only:
        candidate_options["autellix_resident_reserve_only"] = True
    if args.optimized_kernels:
        candidate_options.update(attention_backend=None, sampling_backend=None, disable_cuda_graph=False)
    if candidate_options:
        candidate["engine_args"] = candidate_options
    cases = [dict(name="native", policy="sglang-native", reserve=0), candidate]
    rates = list(args.rates or default_rates)
    if args.native_default is not None:
        upstream = dict(name="native-default", policy="sglang-native", reserve=0, engine_args=dict(
            autellix_native_defaults=True, attention_backend=None, sampling_backend=None))
        if args.native_default:
            if not set(args.native_default) <= set(rates):
                parser.error("upstream SGLang rates must be included in the replay rates")
            upstream["rates"] = args.native_default
        cases.append(upstream)
    if args.fcfs_control:
        if not set(args.fcfs_control) <= set(rates):
            parser.error("FCFS control rates must be included in the replay rates")
        cases.append(dict(candidate, name="autellix-fcfs", policy="fcfs", rates=args.fcfs_control))
    if args.routing_ablation:
        cases.extend(dict(candidate, name=f"{policy}-{routing}", routing=routing)
                     for routing in ("least-used", "round-robin"))
    plan = dict(model=str(model), device=args.device, workload=str(workload), workload_kind=name,
                workload_family=args.workload,
                programs=args.programs or default_count, rates=rates,
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
