from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .baselines import make_baseline
from .models import SimulationResult
from .simulator import Simulator
from .workloads import make_paper_workload


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def result_record(
    result: SimulationResult,
    *,
    workload: str,
    arrival_rate: float,
    engines: int,
    baseline: str,
) -> dict[str, float | int | str]:
    latencies = [m.token_latency for m in result.program_metrics.values()]
    responses = [m.response_time for m in result.program_metrics.values()]
    makespan = max(1, result.makespan)
    return {
        "workload": workload,
        "baseline": baseline,
        "policy": result.policy,
        "scheduler_policy": result.policy,
        "load_balancer": result.load_balancer,
        "load_balancer_policy": result.load_balancer,
        "arrival_rate": arrival_rate,
        "engines": engines,
        "programs": len(result.program_metrics),
        "calls": len(result.calls),
        "makespan": result.makespan,
        "prefetched_calls": result.prefetched_calls,
        "throughput": len(result.program_metrics) / makespan,
        "avg_token_latency": mean(latencies) if latencies else 0.0,
        "avg_critical_path_token_latency": mean(
            [m.critical_path_token_latency for m in result.program_metrics.values()]
        )
        if result.program_metrics
        else 0.0,
        "p95_token_latency": percentile(latencies, 95),
        "p99_token_latency": percentile(latencies, 99),
        "avg_response_time": mean(responses) if responses else 0.0,
        "p95_response_time": percentile(responses, 95),
        "p99_response_time": percentile(responses, 99),
        "total_wait_time": result.total_wait_time,
        "total_execution_time": result.total_execution_time,
        "total_prefill_time": result.total_prefill_time,
        "total_decode_time": result.total_decode_time,
        "total_swap_time": result.total_swap_time,
        "total_scheduler_time": result.total_scheduler_time,
        "avg_engine_utilization": mean(result.engine_utilization.values()),
    }


@dataclass
class ExperimentRunner:
    seed: int = 0
    batch_size: int = 8
    schedule_interval: int = 1
    token_threshold: int = 2048

    def run_one(
        self,
        *,
        workload: str,
        baseline_name: str,
        num_programs: int,
        arrival_rate: float,
        engines: int,
    ) -> tuple[SimulationResult, dict[str, float | int | str]]:
        programs = make_paper_workload(
            workload,
            seed=self.seed,
            num_programs=num_programs,
            arrival_rate=arrival_rate,
        )
        baseline = make_baseline(
            baseline_name,
            programs=programs,
            token_threshold=self.token_threshold,
            schedule_interval=self.schedule_interval,
        )
        result = Simulator(
            programs,
            scheduler=baseline.scheduler,
            load_balancer=baseline.load_balancer,
            execution_model=baseline.execution_model,
            num_engines=engines,
            batch_size=self.batch_size,
            schedule_interval=baseline.schedule_interval,
            overprovision=baseline.overprovision,
            max_time=200_000,
        ).run()
        return result, result_record(
            result,
            workload=workload,
            arrival_rate=arrival_rate,
            engines=engines,
            baseline=baseline.name,
        )

    def sweep(
        self,
        *,
        workload: str,
        baseline_names: Iterable[str],
        arrival_rates: Iterable[float],
        engines: int,
        num_programs: int,
        load_balancers: Iterable[str] | None = None,
    ) -> list[dict[str, float | int | str]]:
        if load_balancers is not None:
            return self.sweep_load_balancers(
                workload=workload,
                load_balancers=load_balancers,
                arrival_rates=arrival_rates,
                engines=engines,
                num_programs=num_programs,
            )
        records: list[dict[str, float | int | str]] = []
        for rate in arrival_rates:
            for baseline_name in baseline_names:
                try:
                    _, record = self.run_one(
                        workload=workload,
                        baseline_name=baseline_name,
                        num_programs=num_programs,
                        arrival_rate=rate,
                        engines=engines,
                    )
                    record["status"] = "ok"
                except RuntimeError as exc:
                    record = {
                        "workload": workload,
                        "baseline": baseline_name,
                        "policy": baseline_name,
                        "scheduler_policy": baseline_name,
                        "load_balancer": "",
                        "load_balancer_policy": "",
                        "arrival_rate": rate,
                        "engines": engines,
                        "programs": num_programs,
                        "calls": 0,
                        "makespan": 0,
                        "prefetched_calls": 0,
                        "throughput": 0.0,
                        "avg_token_latency": 0.0,
                        "avg_critical_path_token_latency": 0.0,
                        "p95_token_latency": 0.0,
                        "p99_token_latency": 0.0,
                        "avg_response_time": 0.0,
                        "p95_response_time": 0.0,
                        "p99_response_time": 0.0,
                        "total_wait_time": 0,
                        "total_execution_time": 0,
                        "total_prefill_time": 0,
                        "total_decode_time": 0,
                        "total_swap_time": 0,
                        "total_scheduler_time": 0,
                        "avg_engine_utilization": 0.0,
                        "status": f"failed: {exc}",
                    }
                records.append(record)
        return records

    def sweep_load_balancers(
        self,
        *,
        workload: str,
        load_balancers: Iterable[str],
        arrival_rates: Iterable[float],
        engines: int,
        num_programs: int,
    ) -> list[dict[str, float | int | str]]:
        records: list[dict[str, float | int | str]] = []
        for rate in arrival_rates:
            programs = make_paper_workload(
                workload,
                seed=self.seed,
                num_programs=num_programs,
                arrival_rate=rate,
            )
            for load_balancer_name in load_balancers:
                baseline = make_baseline(
                    "autellix",
                    programs=programs,
                    token_threshold=self.token_threshold,
                    schedule_interval=self.schedule_interval,
                    load_balancer_name=load_balancer_name,
                )
                result = Simulator(
                    programs,
                    scheduler=baseline.scheduler,
                    load_balancer=baseline.load_balancer,
                    execution_model=baseline.execution_model,
                    num_engines=engines,
                    batch_size=self.batch_size,
                    schedule_interval=baseline.schedule_interval,
                    overprovision=baseline.overprovision,
                    max_time=200_000,
                ).run()
                record = result_record(
                    result,
                    workload=workload,
                    arrival_rate=rate,
                    engines=engines,
                    baseline=load_balancer_name,
                )
                record["status"] = "ok"
                records.append(record)
        return records

    def paper_suite(self, *, quick: bool = False) -> list[dict[str, float | int | str]]:
        workloads = ["sharegpt", "bfcl"] if quick else ["sharegpt", "bfcl", "lats", "mixed"]
        baselines = ["vllm", "vllm-opt", "mlfq", "autellix"]
        rates = [0.2] if quick else [0.1, 0.2, 0.4, 0.8]
        programs = 2 if quick else 32
        records: list[dict[str, float | int | str]] = []
        for workload in workloads:
            records.extend(
                self.sweep(
                    workload=workload,
                    baseline_names=baselines,
                    arrival_rates=rates,
                    engines=1,
                    num_programs=programs,
                )
            )
        for engines in ([1, 2] if quick else [1, 2, 4]):
            records.extend(
                self.sweep_load_balancers(
                    workload="sharegpt",
                    load_balancers=["round-robin", "least-used", "autellix"],
                    arrival_rates=[0.4],
                    engines=engines,
                    num_programs=programs,
                )
            )
        records.extend(
            self.sweep(
                workload="sharegpt",
                baseline_names=["fcfs", "round-robin", "mlfq", "autellix", "srpt"],
                arrival_rates=[0.5],
                engines=1,
                num_programs=programs,
            )
        )
        return records


def write_records(records: list[dict[str, float | int | str]], output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    csv_path = out / "results.csv"
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def plot_records(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    records = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    for metric in ["avg_token_latency", "p95_token_latency", "throughput", "total_wait_time"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = []
        values = []
        for row in records:
            labels.append(f"{row['workload']}:{row['baseline']}@{row['arrival_rate']}")
            values.append(float(row[metric]))
        ax.bar(range(len(values)), values)
        ax.set_title(metric)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        fig.tight_layout()
        path = out / f"{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    return figures
