"""Summarize complete paired paper-workload trials and arrival-window behavior."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(values)
    import math
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p95=values[math.ceil(.95 * len(values))-1], p99=values[math.ceil(.99 * len(values))-1])


def metrics(row):
    kind = row["plan"].get("workload_kind", "sharegpt")
    parallel = row["plan"].get("workload_family", kind) in {"lats", "mixed"}
    token_times = (row["program_token_latencies"] if parallel
                   else row["response_seconds_per_output_token"])
    arrivals, completed = row["arrival_times"], row["program_completion_times"]
    horizon = row["arrival_horizon_seconds"]
    checkpoints = []
    for index in range(1, 11):
        at = horizon * index / 10
        checkpoints.append(dict(seconds=at, unfinished_programs=sum(arrivals[p] <= at < completed[p] for p in arrivals)))
    ordered = sorted(arrivals, key=arrivals.get)
    quartiles = []
    for q in range(4):
        ids = ordered[len(ordered)*q//4:len(ordered)*(q+1)//4]
        quartiles.append(dict(programs=len(ids), mean_program_seconds=statistics.mean(row["programs"][p] for p in ids),
                              mean_seconds_per_output_token=statistics.mean(token_times[p] for p in ids)))
    devices = {int(d) for group in row["device_groups"] for d in group.split(",")}
    gpu = {}
    for device in sorted(devices):
        samples = [g for s in row.get("gpu_samples", []) for g in s["gpus"] if g["index"] == device]
        if samples:
            gpu[device] = dict(mean_utilization=statistics.mean(g["utilization"] for g in samples),
                               peak_memory_mib=max(g["used_mib"] for g in samples))
    cached = [c.get("cached_prompt_tokens") for c in row["calls"]]
    prompt_tokens = sum(c["prompt_tokens"] for c in row["calls"])
    cache = {}
    if prompt_tokens and all(c is not None for c in cached):
        cache["reported_cached_prompt_fraction"] = sum(cached) / prompt_tokens
    return dict(workload=kind, condition=row["condition"], rate=row["rate"], seed=row["seed"],
                model=row["model"], versions=row["versions"], engine_args=row["effective_engine_args"],
                devices=row["device_groups"], trace_sha256=row["trace_sha256"],
                programs=len(arrivals), calls=len(row["calls"]), output_tokens=row["completion_tokens"],
                input_tokens_per_call=distribution(c["prompt_tokens"] for c in row["calls"]),
                program_response_seconds=distribution(row["programs"].values()),
                paper_token_latency=distribution(token_times.values()),
                paper_metric="critical_path/total_output_tokens" if parallel else "program_response/total_output_tokens",
                output_tokens_per_second=row["output_tokens_per_second"], elapsed_seconds=row["elapsed_seconds"],
                arrival_horizon_seconds=horizon, drain_seconds=row["drain_seconds"],
                completions_during_arrivals=row["completions_during_arrivals"],
                maximum_submission_lag_seconds=max(row["first_submission_lag_seconds"].values()),
                arrival_quartiles=quartiles, backlog_checkpoints=checkpoints,
                calls_by_engine=dict(Counter(c["engine_id"] for c in row["calls"])), gpu=gpu, **cache)


def summarize(records):
    groups = {}
    for row in records:
        kind = row["plan"].get("workload_kind", "sharegpt")
        key = kind, row["rate"], row["seed"], tuple(row["device_groups"])
        group = groups.setdefault(key, {})
        if row["condition"] in group:
            raise ValueError(f"duplicate trial: {key}, {row['condition']}")
        group[row["condition"]] = row
    trials, comparisons = [], []
    for key, group in sorted(groups.items()):
        native = next(row for row in group.values() if row["arm"] == "sglang-native")
        baseline = metrics(native)
        trials.extend(metrics(row) for row in group.values())
        for row in group.values():
            if row is native:
                continue
            candidate = metrics(row)
            def lengths(r):
                return {(c["workload_program"], c["workload_call"]):
                        (c["prompt_tokens"], c["completion_tokens"]) for c in r["calls"]}
            def options(r):
                # Native SGLang does not allocate the Autellix host KV cache.
                return {k: v for k, v in r["effective_engine_args"].items()
                        if k not in {"autellix_native_baseline", "autellix_swap_space", "autellix_host_kv_recompute"}}
            paired = dict(same_trace=row["trace_sha256"] == native["trace_sha256"],
                          same_lengths=lengths(row) == lengths(native), same_options=options(row) == options(native),
                          same_model=row["model"] == native["model"], same_versions=row["versions"] == native["versions"])
            if not all(paired.values()):
                raise ValueError(f"unmatched workload or resources: {key}, {paired}")
            per_program = []
            for pid, response in row["programs"].items():
                calls = [c for c in row["calls"] if c["workload_program"] == pid]
                per_program.append(dict(program=pid, calls=len(calls),
                    output_tokens=sum(c["completion_tokens"] for c in calls),
                    max_input_tokens=max(c["prompt_tokens"] for c in calls),
                    native_seconds=native["programs"][pid], candidate_seconds=response,
                    response_delta_seconds=response-native["programs"][pid]))
            comparisons.append(dict(workload=key[0], rate=key[1], seed=key[2], devices=key[3], condition=row["condition"],
                pairing=paired, programs=len(row["programs"]),
                per_program=sorted(per_program, key=lambda p: -p["response_delta_seconds"]),
                faster_programs=sum(row["programs"][p] < native["programs"][p] for p in row["programs"]),
                matching_texts=sum(native["output_hashes"][c] == text for c, text in row["output_hashes"].items()),
                calls=len(row["calls"]),
                mean_response_change_pct=100*(candidate["program_response_seconds"]["mean"]/baseline["program_response_seconds"]["mean"]-1),
                mean_paper_latency_change_pct=100*(candidate["paper_token_latency"]["mean"]/baseline["paper_token_latency"]["mean"]-1),
                p95_paper_latency_change_pct=100*(candidate["paper_token_latency"]["p95"]/baseline["paper_token_latency"]["p95"]-1),
                throughput_change_pct=100*(candidate["output_tokens_per_second"]/baseline["output_tokens_per_second"]-1)))
    slo_points = []
    for workload in sorted({t["workload"] for t in trials}):
        selected = [t for t in trials if t["workload"] == workload]
        for target in [.1, .25, .5, 1., 2., 4., 8.]:
            rates = {condition: max((t["rate"] for t in selected if t["condition"] == condition
                                      and t["paper_token_latency"]["mean"] <= target), default=None)
                     for condition in sorted({t["condition"] for t in selected})}
            slo_points.append(dict(workload=workload, mean_seconds_per_output_token_target=target,
                                   maximum_tested_rate_meeting_latency=rates))
    return dict(trials=trials, comparisons=comparisons, slo_points=slo_points,
                slo_definition="Largest measured offered rate satisfying mean program token latency; finite trace backlog and drain are reported separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [row for file in args.inputs for row in json.loads(Path(file).read_text())]
    result = summarize(records)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
