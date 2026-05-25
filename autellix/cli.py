from __future__ import annotations

import argparse
import json
from typing import Iterable

from .experiments import ExperimentRunner, make_baseline, plot_records, write_records
from .core.load_balancer import make_load_balancer
from .core.schedulers import (
    DEFAULT_PRIORITY_BOUNDARIES,
    DEFAULT_QUEUE_QUANTA,
    make_scheduler,
    parse_float_tuple,
    parse_int_tuple,
)
from .core.simulator import Simulator
from .experiments import make_figure2_workload, make_synthetic_workload


def _make_workload(name: str, seed: int, programs: int, arrival_rate: float):
    if name.lower() == "figure2":
        return make_figure2_workload()
    return make_synthetic_workload(
        name,
        seed=seed,
        num_programs=programs,
        arrival_rate=arrival_rate,
    )


def _make_scheduler(args: argparse.Namespace, policy: str):
    boundaries = parse_float_tuple(args.boundaries, DEFAULT_PRIORITY_BOUNDARIES)
    quanta = parse_int_tuple(args.quanta, DEFAULT_QUEUE_QUANTA)
    return make_scheduler(
        policy,
        priority_boundaries=boundaries,
        queue_quanta=quanta,
        anti_starvation_beta=args.beta,
    )


def _run_once(args: argparse.Namespace, policy: str):
    workload = _make_workload(args.workload, args.seed, args.programs, args.arrival_rate)
    if policy in {"vllm", "vllm-opt", "autellix", "least-used"}:
        baseline = make_baseline(
            policy,
            programs=workload,
            token_threshold=args.token_threshold,
            schedule_interval=args.schedule_interval,
        )
        scheduler = baseline.scheduler
        load_balancer = baseline.load_balancer
        execution_model = baseline.execution_model
        overprovision = baseline.overprovision
        schedule_interval = baseline.schedule_interval
    else:
        scheduler = _make_scheduler(args, policy)
        load_balancer = make_load_balancer(args.load_balancer, token_threshold=args.token_threshold)
        execution_model = args.execution_model
        overprovision = args.overprovision
        schedule_interval = args.schedule_interval
    return Simulator(
        workload,
        scheduler=scheduler,
        load_balancer=load_balancer,
        num_engines=args.engines,
        batch_size=args.batch_size,
        schedule_interval=schedule_interval,
        overprovision=overprovision,
        execution_model=execution_model,
    ).run()


def _print_summary(result, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return
    summary = result.summary()
    print(f"policy={summary['policy']} load_balancer={summary['load_balancer']}")
    print(
        "programs={programs} calls={calls} makespan={makespan} "
        "total_wait={total_wait_time} avg_wait={avg_wait_time:.2f} "
        "avg_response={avg_response_time:.2f} avg_token_latency={avg_token_latency:.4f}".format(
            **summary
        )
    )
    util = ", ".join(
        f"E{engine_id}={value:.2%}"
        for engine_id, value in sorted(summary["engine_utilization"].items())
    )
    print(f"engine_utilization: {util}")


def _print_compare(results: Iterable, *, as_json: bool) -> None:
    results = list(results)
    if as_json:
        print(json.dumps([result.summary() for result in results], indent=2, sort_keys=True))
        return
    header = (
        f"{'policy':<8} {'makespan':>8} {'total_wait':>11} "
        f"{'avg_wait':>9} {'avg_resp':>9} {'tok_lat':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        summary = result.summary()
        print(
            f"{summary['policy']:<8} {summary['makespan']:>8} "
            f"{summary['total_wait_time']:>11} {summary['avg_wait_time']:>9.2f} "
            f"{summary['avg_response_time']:>9.2f} {summary['avg_token_latency']:>9.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autellix scheduling simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--workload", default="figure2", choices=["figure2", "sharegpt", "bfcl", "lats", "mixed"])
        subparser.add_argument("--engines", type=int, default=1)
        subparser.add_argument("--batch-size", type=int, default=2)
        subparser.add_argument("--seed", type=int, default=0)
        subparser.add_argument("--programs", type=int, default=24)
        subparser.add_argument("--arrival-rate", type=float, default=1.0)
        subparser.add_argument("--load-balancer", default="autellix", choices=["autellix", "least-used", "round-robin"])
        subparser.add_argument("--token-threshold", type=int, default=2048)
        subparser.add_argument("--schedule-interval", type=int, default=1)
        subparser.add_argument("--overprovision", type=int, default=0)
        subparser.add_argument("--execution-model", default="fixed", choices=["fixed", "vllm", "vllm-opt", "mlfq", "autellix"])
        subparser.add_argument("--boundaries", default=None, help="comma-separated priority boundaries, e.g. 0,2,4,8,inf")
        subparser.add_argument("--quanta", default=None, help="comma-separated queue quanta, e.g. 1,2,4,8")
        subparser.add_argument("--beta", type=float, default=8.0)
        subparser.add_argument("--json", action="store_true", help="emit JSON output")

    run_parser = subparsers.add_parser("run", help="run one scheduler")
    add_common(run_parser)
    run_parser.add_argument("--policy", default="plas", choices=["fcfs", "round-robin", "mlfq", "plas", "atlas", "srpt", "vllm", "vllm-opt", "autellix", "least-used"])

    compare_parser = subparsers.add_parser("compare", help="compare schedulers")
    add_common(compare_parser)
    compare_parser.add_argument("--policies", default="fcfs,mlfq,plas", help="comma-separated scheduler names")

    sweep_parser = subparsers.add_parser("sweep", help="run an arrival-rate sweep")
    add_common(sweep_parser)
    sweep_parser.add_argument("--policies", default="vllm,vllm-opt,mlfq,autellix")
    sweep_parser.add_argument("--arrival-rates", default="0.1,0.2,0.4")
    sweep_parser.add_argument("--output", default=None)

    suite_parser = subparsers.add_parser("paper-suite", help="run paper-style simulation suite")
    suite_parser.add_argument("--output", default="outputs/paper_sim")
    suite_parser.add_argument("--seed", type=int, default=0)
    suite_parser.add_argument("--batch-size", type=int, default=8)
    suite_parser.add_argument("--schedule-interval", type=int, default=1)
    suite_parser.add_argument("--quick", action="store_true")

    preset_parser = subparsers.add_parser("paper-preset", help="run one paper-style experiment preset")
    preset_parser.add_argument(
        "--preset",
        required=True,
        choices=[
            "workload-analysis",
            "latency-throughput",
            "load-balancer",
            "offline-makespan",
            "timing-breakdown",
        ],
    )
    preset_parser.add_argument("--workload", default="sharegpt", choices=["sharegpt", "bfcl", "lats", "mixed"])
    preset_parser.add_argument("--dataset", default=None, help="optional JSON/JSONL/CSV dataset path")
    preset_parser.add_argument("--output", default=None)
    preset_parser.add_argument("--seed", type=int, default=0)
    preset_parser.add_argument("--batch-size", type=int, default=8)
    preset_parser.add_argument("--schedule-interval", type=int, default=1)
    preset_parser.add_argument("--programs", type=int, default=24)
    preset_parser.add_argument("--engines", type=int, default=1)

    plot_parser = subparsers.add_parser("plot", help="plot records from a JSON result file")
    plot_parser.add_argument("--input", required=True)
    plot_parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        result = _run_once(args, args.policy)
        _print_summary(result, as_json=args.json)
        return 0
    if args.command == "compare":
        policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
        results = [_run_once(args, policy) for policy in policies]
        _print_compare(results, as_json=args.json)
        return 0
    if args.command == "sweep":
        runner = ExperimentRunner(
            seed=args.seed,
            batch_size=args.batch_size,
            schedule_interval=args.schedule_interval,
            token_threshold=args.token_threshold,
        )
        records = runner.sweep(
            workload=args.workload,
            baseline_names=[p.strip() for p in args.policies.split(",") if p.strip()],
            arrival_rates=[float(v.strip()) for v in args.arrival_rates.split(",") if v.strip()],
            engines=args.engines,
            num_programs=args.programs,
        )
        if args.output:
            json_path, csv_path = write_records(records, args.output)
            print(f"wrote {json_path}")
            print(f"wrote {csv_path}")
        else:
            print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    if args.command == "paper-suite":
        runner = ExperimentRunner(
            seed=args.seed,
            batch_size=args.batch_size,
            schedule_interval=args.schedule_interval,
        )
        records = runner.paper_suite(quick=args.quick)
        json_path, csv_path = write_records(records, args.output)
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")
        return 0
    if args.command == "paper-preset":
        runner = ExperimentRunner(
            seed=args.seed,
            batch_size=args.batch_size,
            schedule_interval=args.schedule_interval,
        )
        records = runner.paper_preset(
            args.preset,
            workload=args.workload,
            num_programs=args.programs,
            engines=args.engines,
            dataset_path=args.dataset,
        )
        if args.output:
            json_path, csv_path = write_records(records, args.output)
            print(f"wrote {json_path}")
            print(f"wrote {csv_path}")
        else:
            print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    if args.command == "plot":
        figures = plot_records(args.input, args.output)
        for figure in figures:
            print(f"wrote {figure}")
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
