"""Run the paper evaluation in one physical-GPU queue, with per-case checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from run_h20_paper import WORKLOADS
from summarize_paper_suite import paper_matrix


FIGURES = (4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19)
RATES = {
    "sharegpt": (.04, .10, .20, .30, .48),
    "bfcl": (.12, .24, .48, .72, 1.0),
    "lats": (.004, .01, .02, .03, .05),
    "mixed": (.01, .025, .05, .075, .12),
}


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def cases_for(backend, family):
    policy = WORKLOADS[family][3]
    if backend == "vllm":
        native = [dict(name=name, policy=name, interval=steps, engine_args={"preemption_mode": None})
                  for name, steps in (("vllm", 1), ("vllm-opt", 1), ("vllm-opt-multistep", 8))]
        options = dict(keep_resident_kv=True, publish_activity_at_window=True, paper_chunked_prefill=True,
                       refill_idle_slots=True, resident_reserve_only=True, fused_swap=True)
    else:
        native = [dict(name="sglang-native", policy="sglang-native", interval=1,
                       engine_args={"autellix_native_defaults": True})]
        options = dict(autellix_cache_paused_kv=True, autellix_window_activity=True,
                       autellix_refill_idle_slots=True, autellix_resident_reserve_only=True,
                       autellix_host_kv_recompute=True, autellix_fused_swap=True, disable_cuda_graph=False,
                       chunked_prefill_size=8192)
    common = dict(interval=8, reserve=1, quantum_scale=8, boundary_scale=16, engine_args=options)
    return [*native, dict(common, name="fcfs-combined", policy="fcfs"),
            dict(common, name="mlfq", policy="mlfq"), dict(common, name="autellix", policy=policy)]


def build_plans(args):
    selected = set(args.figures)
    figures = tuple(sorted(selected & {12, 13}))
    device = str(args.gpu)
    engine_args = {
        "vllm": dict(dtype="bfloat16", max_model_len=32768, tensor_parallel_size=1,
                     max_num_seqs=args.batch_size, max_num_batched_tokens=32768,
                     block_size=16, num_gpu_blocks_override=args.kv_tokens // 16,
                     gpu_memory_utilization=.88, swap_space=args.swap_gib, enforce_eager=args.eager,
                     observe_native=False, seed=0),
        "sglang": dict(dtype="bfloat16", context_length=32768, pp_size=1,
                       max_running_requests=args.batch_size, max_total_tokens=args.kv_tokens,
                       max_prefill_tokens=32768, mem_fraction_static=.9,
                       attention_backend=None, sampling_backend=None, random_seed=0,
                       autellix_swap_space=args.swap_gib),
    }

    def plan(backend, family, name, selected_figures, cases, rates, count, *, offline=False, timing=False):
        args_for_engine = dict(engine_args[backend])
        if timing and backend == "vllm":
            args_for_engine.update(observe_native=True, measure_timing=True)
        elif timing:
            args_for_engine["autellix_measure_timing"] = True
        if offline or timing:
            args_for_engine["publish_activity" if backend == "vllm" else "autellix_publish_activity"] = False
        filename = "sharegpt-whole4000.json" if offline or 4 in selected_figures else WORKLOADS[family][0]
        return dict(backend=backend, model=str(args.model.resolve()), device=device,
                    experiment="offline" if offline else "online", paper_figures=list(selected_figures),
                    workload=str((args.datasets / filename).resolve()), workload_family=family,
                    workload_kind=name, programs=count, rates=list(rates), seed=args.seed,
                    repeats=1 if offline or timing else args.repeats, unique_programs=offline,
                    request_timeout=args.request_timeout, progress_interval=60,
                    equal_dataset_counts=family == "mixed", scheduler_trace=timing,
                    compact_calls=True, engine_args=args_for_engine, cases=cases)

    for backend in args.backends:
        if backend == "vllm" and 4 in selected:
            family, rate = "sharegpt", .20
            cases = [case for case in cases_for(backend, family) if case["name"] in {"vllm", "autellix"}]
            yield plan(backend, family, "fig4-concurrency-vllm", (4,), cases, (rate,),
                       math.ceil(rate * args.diagnostic_seconds), timing=True)
        if backend == "vllm" and selected & {5, 6}:
            for family in ("sharegpt", "bfcl", "lats"):
                cases = [case for case in cases_for(backend, family) if case["name"] in {"vllm", "mlfq", "autellix"}]
                yield plan(backend, family, f"fig5-6-{family}-vllm", sorted(selected & {5, 6}), cases,
                           args.rates or (RATES[family][1], RATES[family][3]), WORKLOADS[family][1], timing=True)
        if figures:
            for family in args.workloads:
                name = f"fig12-13-{family}-{backend}"
                yield plan(backend, family, name, figures, cases_for(backend, family),
                           args.rates or RATES[family], WORKLOADS[family][1])
        if 16 in selected:
            for count in args.offline_sizes:
                for case in cases_for(backend, "sharegpt"):
                    if case["name"] == "vllm-opt-multistep":
                        continue
                    name = f"fig16-{count}-{backend}-{case['name']}"
                    yield plan(backend, "sharegpt", name, (16,), [case], (None,), count, offline=True)
        if backend == "vllm" and 17 in selected:
            for family in ("sharegpt", "lats"):
                cases = [case for case in cases_for(backend, family)
                         if case["name"] in {"vllm-opt", "mlfq", "autellix"}]
                name = f"fig17-{family}-{backend}"
                yield plan(backend, family, name, (17,), cases, (RATES[family][-1],),
                           WORKLOADS[family][1], timing=True)
        if 19 in selected:
            for family in ("sharegpt", "lats"):
                candidate = cases_for(backend, family)[-1]
                mode = dict(candidate["engine_args"])
                if backend == "vllm":
                    mode["keep_resident_kv"] = False
                    torch_swap = {**mode, "fused_swap": False}
                    cases = [dict(candidate, name="native-swap", engine_args={**torch_swap, "batched_swap": False})]
                else:
                    mode["autellix_cache_paused_kv"] = False
                    torch_swap = {**mode, "autellix_fused_swap": False}
                    cases = []
                cases.extend((dict(candidate, name="torch-swap", engine_args=torch_swap),
                              dict(candidate, name="fused-swap", engine_args=mode),
                              dict(candidate, name="fused-swap-N1", interval=1, engine_args=mode)))
                name = f"fig19-{family}-{backend}"
                pressure_rate = 1.92 if family == "sharegpt" else .28
                yield plan(backend, family, name, (19,), cases, (pressure_rate,),
                           WORKLOADS[family][1], timing=True)


def simulate(args):
    from autellix.experiments.datasets import load_programs_from_file, workload_analysis
    from autellix.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(seed=args.seed, batch_size=args.batch_size)
    if 7 in args.figures:
        from analyze_paper_motivation import prefix_opportunities
        from autellix.runtime.benchmark import load_workloads

        rows = {family: prefix_opportunities(load_workloads([args.datasets / WORKLOADS[family][0]]))
                for family in ("sharegpt", "lats")}
        write_json(args.run_dir / "fig7-prefix-opportunities.json", rows)
    for figure in sorted(set(args.figures) & {11, 14, 15, 18}):
        families = tuple(WORKLOADS) if figure == 11 else ("sharegpt", "lats") if figure in {14, 18} else ("sharegpt",)
        rows = []
        for family in families:
            data = str(args.datasets / ("sharegpt-whole4000.json" if figure == 11 and family == "sharegpt"
                                        else WORKLOADS[family][0]))
            count = 4000 if figure == 11 else WORKLOADS[family][1]
            if figure == 11:
                from analyze_paper_motivation import workload_histograms

                programs = load_programs_from_file(data, default_kind=family)[:count]
                rows.append(dict(workload_analysis(programs), workload=family, status="ok",
                                 histograms=workload_histograms(programs)))
            elif figure == 14:
                rates = (.04, .12, .24, .40, .70) if family == "sharegpt" else (.002, .004, .008, .016, .032)
                for engines in (2, 4):
                    rows.extend(runner.sweep(workload=family, baseline_names=["autellix"],
                                             arrival_rates=rates, engines=engines,
                                             load_balancers=("round-robin", "least-used", "autellix"),
                                             num_programs=count, dataset_path=data))
            elif figure == 15:
                for engines in (1, 2, 4, 8):
                    rows.extend(runner.sweep(workload=family, baseline_names=["autellix"],
                                             arrival_rates=(.08, .16, .32, .64, 1., 1.5, 2.5, 4., 6., 10.), engines=engines,
                                             num_programs=512, dataset_path=data))
            else:
                rates = (.0008, .0016, .0032, .0064) if family == "lats" else (.005, .01, .02, .03, .04)
                rows.extend(runner.sweep(workload=family, baseline_names=("fcfs", "round-robin", "mlfq", "autellix", "srpt"),
                                         arrival_rates=rates, engines=1, num_programs=count,
                                         dataset_path=data, fixed_steps=True))
        measurement = "workload_trace" if figure == 11 else "simulation"
        write_json(args.run_dir / f"fig{figure}-{measurement.replace('_', '-')}.json",
                   [dict(row, measurement=measurement, paper_figure=figure) for row in rows])
        if figure == 15:
            rates_by_slo = []
            for engines in (1, 2, 4, 8):
                for target in (.045, .05, .08, .1, .2, .4):
                    eligible = [row for row in rows if row["status"] == "ok" and row["engines"] == engines
                                and row["avg_token_latency"] <= target]
                    rates_by_slo.append(dict(engines=engines, seconds_per_token_target=target,
                                             maximum_tested_rate=max((r["arrival_rate"] for r in eligible), default=None),
                                             maximum_completed_rate=max((r["arrival_window_throughput"] for r in eligible),
                                                                        default=None)))
            write_json(args.run_dir / "fig15-slo-simulation.json", rates_by_slo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int, help="one physical GPU index")
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--sglang-python", default=sys.executable)
    parser.add_argument("--vllm-preload", help="LD_PRELOAD for the pinned vLLM environment")
    parser.add_argument("--figures", nargs="+", type=int, choices=FIGURES, default=FIGURES)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=tuple(WORKLOADS))
    parser.add_argument("--backends", nargs="+", choices=("vllm", "sglang"), default=("vllm", "sglang"))
    parser.add_argument("--cases", nargs="+", help="select named inference cases from the figure plans")
    parser.add_argument("--offline-sizes", nargs="+", type=int, default=(1000, 2000, 3000, 4000))
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kv-tokens", type=int, default=262144)
    parser.add_argument("--swap-gib", type=float, default=128)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=79200)
    parser.add_argument("--diagnostic-seconds", type=float, default=3600, help="nominal arrival window for Figure 4")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or args.kv_tokens < 32768 or args.kv_tokens % 16:
        parser.error("use a physical GPU index and a KV capacity divisible by 16 of at least 32768 tokens")
    if min(args.offline_sizes) < 1 or max(args.offline_sizes) > 4000:
        parser.error("offline sizes must be between 1 and 4000")

    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    plans = list(build_plans(args))
    if args.cases:
        allowed = {case["name"] for plan in plans for case in plan["cases"]}
        if set(args.cases) - allowed:
            parser.error("every selected case must exist in at least one requested figure")
        plans = [dict(plan, cases=[case for case in plan["cases"] if case["name"] in args.cases])
                 for plan in plans if any(case["name"] in args.cases for case in plan["cases"])]
    manifest_path = out / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists()
                else dict(device=args.gpu, gpu_plans={}, cpu_figures=[]))
    if manifest["device"] != args.gpu:
        parser.error("resume a results directory on its original physical GPU")
    for plan in plans:
        write_json(out / (plan["workload_kind"] + "-plan.json"), plan)
    manifest["gpu_plans"].update((p["workload_kind"], p["paper_figures"]) for p in plans)
    manifest["cpu_figures"] = sorted(set(manifest.get("cpu_figures", ())) | (set(args.figures) & {7, 11, 14, 15, 18}))
    write_json(manifest_path, manifest)
    if args.plan_only:
        return 0

    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    venv = dict(env, LD_PRELOAD=args.vllm_preload) if args.vllm_preload else env
    interpreters = {"vllm": (args.vllm_python, venv), "sglang": (args.sglang_python, env)}
    statuses = {}
    statuses_path = out / "exit-codes.json"
    previous_statuses = (json.loads(statuses_path.read_text(encoding="utf-8")) if statuses_path.exists() else {})

    def gpu_run(name, cmd, run_env):
        pids = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
                                        "--format=csv,noheader,nounits"], text=True)
        if any(line.strip().isdecimal() for line in pids.splitlines()):
            raise RuntimeError(f"GPU {args.gpu} is occupied before {name}")
        print(f"START {name}", flush=True)
        with (out / (name + ".log")).open("w", encoding="utf-8") as log:
            statuses[name] = subprocess.run(cmd, cwd=root, env=run_env, stdout=log, stderr=subprocess.STDOUT).returncode
        write_json(statuses_path, previous_statuses | statuses)

    simulate(args)
    if 19 in args.figures and "vllm" in args.backends:
        exe, run_env = interpreters["vllm"]
        for blocks in (8, 32, 128, 512):
            name = f"fig19-kernel-{blocks}"
            gpu_run(name, [exe, str(root / "cuda/batched_swap_benchmark.py"), "--blocks", str(blocks),
                           "--layers", "32", "--block-bytes", "32768", "--fused", "--output",
                           str(out / (name + ".json"))], run_env)
    for plan in plans:
        name, backend = plan["workload_kind"], plan["backend"]
        exe, run_env = interpreters[backend]
        gpu_run(name, [exe, str(root / "scripts/run_sglang_matrix.py"), "--plan", str(out / (name + "-plan.json")),
                       "--output", str(out / (name + ".json")), "--resume"], run_env)
        if plan["scheduler_trace"] and (out / (name + ".json")).exists():
            with (out / (name + "-timing.log")).open("w", encoding="utf-8") as log:
                subprocess.run([exe, str(root / "scripts/analyze_scheduler_trace.py"), str(out / (name + ".json")),
                                "--output", str(out / (name + "-timing.json"))],
                               cwd=root, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if set(plan["paper_figures"]) & {4, 5, 6} and (out / (name + ".json")).exists():
            from analyze_paper_motivation import motivation

            rows = json.loads((out / (name + ".json")).read_text(encoding="utf-8"))
            write_json(out / (name + "-motivation.json"), motivation(rows))
    results = [out / (name + ".json") for name, figures in manifest["gpu_plans"].items()
               if set(figures) & {12, 13, 16} and (out / (name + ".json")).exists()]
    if results:
        rows = (row for path in results for row in json.loads(path.read_text(encoding="utf-8")))
        write_json(out / "measured-comparisons.json", paper_matrix(rows))
    return int(any(statuses.values()))


if __name__ == "__main__":
    raise SystemExit(main())
