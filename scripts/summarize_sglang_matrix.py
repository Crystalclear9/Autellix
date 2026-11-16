"""Summarize named policies against native SGLang at each offered load."""
import argparse
import json
from pathlib import Path


def summarize(records):
    groups = {}
    for row in records:
        key = row["rate"], row["repeat"]
        group = groups.setdefault(key, {})
        name = row["condition"]
        if name in group:
            raise ValueError(f"duplicate condition {name} at {key}")
        group[name] = row
    rows = []
    for (rate, repeat), group in sorted(groups.items()):
        baseline = next(row for row in group.values() if row["arm"] == "sglang-native")
        for name, row in group.items():
            hashes = row["output_hashes"]
            base_hashes = baseline["output_hashes"]
            def lengths(record):
                return {(c["workload_program"], c["workload_call"]):
                        (c["prompt_tokens"], c["completion_tokens"]) for c in record["calls"]}
            def common_args(record):
                return {k: v for k, v in record["effective_engine_args"].items()
                        if k not in {"autellix_native_baseline", "autellix_retain_running"}}
            latency = row["program_latency_seconds"]
            token_latency = row["paper_sequential_token_latency"]
            rows.append(dict(
                condition=name, rate=rate, repeat=repeat,
                same_trace=row["trace_sha256"] == baseline["trace_sha256"],
                same_call_lengths=lengths(row) == lengths(baseline),
                same_model=row["model"] == baseline["model"],
                same_versions=row["versions"] == baseline["versions"],
                same_engine_args=common_args(row) == common_args(baseline),
                matching_texts=sum(base_hashes.get(k) == v for k, v in hashes.items()),
                calls=len(hashes), programs=len(row["programs"]),
                output_tokens=row["completion_tokens"],
                mean_program_seconds=latency["mean"], p95_program_seconds=latency["p95"],
                mean_seconds_per_output_token=token_latency["mean"],
                p95_seconds_per_output_token=token_latency["p95"],
                output_tokens_per_second=row["output_tokens_per_second"],
                mean_latency_change_pct=100 * (latency["mean"] / baseline["program_latency_seconds"]["mean"] - 1),
                p95_latency_change_pct=100 * (latency["p95"] / baseline["program_latency_seconds"]["p95"] - 1),
                token_latency_change_pct=100 * (token_latency["mean"] / baseline["paper_sequential_token_latency"]["mean"] - 1),
                output_throughput_change_pct=100 * (row["output_tokens_per_second"] / baseline["output_tokens_per_second"] - 1),
                elapsed_seconds=row["elapsed_seconds"],
                arrival_horizon_seconds=row["arrival_horizon_seconds"],
                drain_seconds=row["drain_seconds"],
                completions_during_arrivals=row["completions_during_arrivals"],
                max_submission_lag_seconds=max(row["first_submission_lag_seconds"].values()),
            ))
    return dict(rows=rows, interpretation="Latency change below zero favors the candidate; throughput change above zero favors it. Offered program rate and finite-run output throughput are different measurements.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(json.loads(Path(args.input).read_text(encoding="utf-8")))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
