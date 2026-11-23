"""Run a real-GPU Autellix paper replay on the pinned vLLM backend."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from run_h20_paper import WORKLOADS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("online", "offline", "routing", "scaling", "swap", "timing"))
    parser.add_argument("workload", choices=WORKLOADS)
    parser.add_argument("--model", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--datasets", type=Path, help="directory with the standard workload files")
    source.add_argument("--trace", type=Path, help="use a prepared program trace")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, help="comma-separated TP ranks; semicolon-separated replicas")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--programs", type=int)
    selection.add_argument("--all-programs", action="store_true", help="replay each offline program once")
    parser.add_argument("--unique-programs", action="store_true", help="sample each source program at most once per trace")
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--kv-blocks", type=int, help="set identical GPU KV capacity in 16-token blocks")
    parser.add_argument("--swap-gib", type=float, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=14400,
                        help="seconds allowed for each call, including queueing")
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--quantum-scale", type=float, default=2)
    parser.add_argument("--boundary-scale", type=float, default=128)
    parser.add_argument("--window-activity", action="store_true",
                        help="publish unscheduled calls' waiting time at scheduling windows")
    parser.add_argument("--keep-resident-kv", action="store_true",
                        help="keep preempted KV on GPU until another request needs its blocks")
    parser.add_argument("--publish-activity-at-window", action="store_true", help="publish the shared activity view at window boundaries")
    parser.add_argument("--refill-idle-slots", action="store_true", help="fill vacant batch slots after the prepared reserve is exhausted")
    parser.add_argument("--resident-reserve-only", action="store_true", help="choose reserve requests from KV already on GPU")
    parser.add_argument("--cuda-graphs", action="store_true", help="enable the native decode CUDA graphs for all cases")
    parser.add_argument("--graph-context", type=int, help="maximum context length for CUDA graph decode")
    parser.add_argument("--paper-chunked-prefill", action="store_true", help="use budgeted prefill along with Autellix windows")
    parser.add_argument("--resident-decode-headroom", action="store_true", help="budget one token per selected resident decode")
    parser.add_argument("--exact-prefix-cache", action="store_true", help="match computed prefix accounting in the chunked vLLM controls")
    parser.add_argument("--deduplicate-prefill", action="store_true",
                        help="let identical prompts in the same window reuse the first prefetched prefix")
    parser.add_argument("--recompute-on-swap-full", action="store_true", help="recompute paused KV when the host cache fills")
    parser.add_argument("--completion-only", action="store_true", help="return only final outputs for all cases")
    parser.add_argument("--no-publish-activity", action="store_true", help="skip intermediate activity writes for performance replay")
    parser.add_argument("--fused-swap", action="store_true", help="pack and unpack all CUDA KV layers with the fused kernel")
    parser.add_argument("--no-scheduler-trace", action="store_true", help="measure without per-step scheduler JSONL events")
    parser.add_argument("--cases", nargs="+", help="run only the listed cases")
    parser.add_argument("--label")
    args = parser.parse_args()

    filename, count, rates, policy = WORKLOADS[args.workload]
    groups = [group.split(",") for group in args.device.split(";")]
    devices = [device for group in groups for device in group]
    tp_size = len(groups[0])
    if (any(len(group) != tp_size for group in groups) or not all(device.isdecimal() for device in devices)
            or len(devices) != len(set(devices))):
        parser.error("use distinct GPU indices and equal TP sizes across replicas")
    if args.experiment == "routing" and len(groups) < 2:
        parser.error("routing requires at least two replicas")
    if args.experiment == "offline" and args.rates:
        parser.error("offline programs all arrive at time zero")
    if args.all_programs and args.experiment != "offline":
        parser.error("--all-programs requires an offline replay")
    count = args.programs if args.programs is not None else count
    if count < 1 or (args.workload == "mixed" and count % 3):
        parser.error("use a positive program count divisible by three for Mixed")

    candidate = dict(name="autellix", policy=policy, interval=args.interval, reserve=1,
                     quantum_scale=args.quantum_scale, boundary_scale=args.boundary_scale)
    options = {name: True for name in ("keep_resident_kv", "publish_activity_at_window", "refill_idle_slots",
                                       "paper_chunked_prefill", "resident_reserve_only", "fused_swap",
                                       "resident_decode_headroom", "exact_prefix_cache", "recompute_on_swap_full",
                                       "deduplicate_prefill")
               if getattr(args, name)}
    if options:
        candidate["engine_args"] = options
    baselines = [dict(name=name, policy=name, interval=steps, engine_args={"preemption_mode": None}) for name, steps in (
        ("vllm", 1), ("vllm-opt", 1), ("vllm-opt-multistep", args.interval))]
    for case in baselines:
        if case["policy"] == "vllm-opt":
            if args.exact_prefix_cache:
                case["engine_args"]["exact_prefix_cache"] = True
        elif args.batch_tokens < 32768:
            case["engine_args"]["max_num_batched_tokens"] = 32768
    mlfq = dict(candidate, name="mlfq", policy="mlfq")
    cases = [*baselines, mlfq, candidate]
    if args.experiment == "routing":
        cases = [dict(candidate, name=routing, routing=routing)
                 for routing in ("program", "least-used", "round-robin")]
    elif args.experiment == "scaling":
        cases = [candidate]
    elif args.experiment == "swap":
        native_swap = {**candidate.get("engine_args", {}), "batched_swap": False}
        cases = [dict(candidate, name="native-swap", engine_args=native_swap), candidate]
        if args.fused_swap:
            cases.insert(1, dict(candidate, name="torch-swap", engine_args={**options, "fused_swap": False}))
    if args.cases:
        if set(args.cases) - {case["name"] for case in cases}:
            parser.error("unknown case for this experiment")
        cases = [case for case in cases if case["name"] in args.cases]

    workload = (args.trace if args.trace else args.datasets / filename).resolve(strict=True)
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    label = args.label or f"{args.workload}-{args.experiment}"
    plan = dict(backend="vllm", model=str(args.model.resolve(strict=True)), device=args.device,
                experiment=args.experiment, workload=str(workload), workload_kind=label, workload_family=args.workload,
                programs=0 if args.all_programs else count,
                rates=[None] if args.experiment == "offline" else list(args.rates or rates),
                seed=args.seed, repeats=args.repeats, request_timeout=args.request_timeout, compact_calls=True,
                scheduler_trace=not args.no_scheduler_trace,
                engine_args=dict(dtype="bfloat16", max_model_len=32768, tensor_parallel_size=tp_size,
                                 max_num_seqs=args.batch_size, max_num_batched_tokens=args.batch_tokens,
                                 gpu_memory_utilization=.88, swap_space=args.swap_gib, seed=0,
                                 enforce_eager=not args.cuda_graphs,
                                 observe_native=args.experiment == "timing", measure_timing=args.experiment == "timing"),
                cases=cases)
    if args.unique_programs:
        plan["unique_programs"] = True
    if args.completion_only:
        plan["engine_args"]["completion_only"] = True
    if args.graph_context is not None:
        plan["engine_args"]["max_seq_len_to_capture"] = args.graph_context
    if args.no_publish_activity:
        plan["engine_args"]["publish_activity"] = False
    if args.window_activity:
        plan["engine_args"]["publish_waiting_each_step"] = False
    if args.kv_blocks is not None:
        plan["engine_args"]["num_gpu_blocks_override"] = args.kv_blocks
    if args.workload == "mixed" and not args.all_programs:
        plan["equal_dataset_counts"] = True
    plan_file, results = output / f"{label}-plan.json", output / f"{label}.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).with_name("run_sglang_matrix.py")),
                    "--plan", str(plan_file), "--output", str(results), "--resume"], check=True)
    for row in json.loads(results.read_text(encoding="utf-8")):
        state = Path(row["state_dir"])
        if state.is_dir():
            shutil.copytree(state, output / "scheduler-state" / state.name, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
