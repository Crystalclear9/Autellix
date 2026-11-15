"""Summarize every paired trial, preserving negative results and mismatches."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(records, expected_repeats=3):
    if not records:
        raise ValueError("no trial results")
    groups = {}
    for row in records:
        key = row["rate"], row["repeat"]
        arms = groups.setdefault(key, {})
        if row["arm"] in arms:
            raise ValueError(f"duplicate arm for {key}")
        arms[row["arm"]] = row
    pairs = []
    for (rate, repeat), arms in sorted(groups.items()):
        if set(arms) != {"sglang-native", "atlas"}:
            raise ValueError(f"incomplete paired trial: {rate}, {repeat}")
        baseline, candidate = arms["sglang-native"], arms["atlas"]
        def common_options(row):
            return {k:v for k,v in row.get("effective_engine_args", {}).items()
                    if k != "autellix_native_baseline"}
        matched = (baseline["trace_sha256"] == candidate["trace_sha256"]
                   and baseline["completion_tokens"] == candidate["completion_tokens"]
                   and baseline["output_hashes"] == candidate["output_hashes"]
                   and common_options(baseline) == common_options(candidate)
                   and baseline.get("model") == candidate.get("model")
                   and baseline.get("versions") == candidate.get("versions"))
        pairs.append(dict(rate=rate, repeat=repeat, matched=matched,
            native_mean_seconds=baseline["program_latency_seconds"]["mean"],
            atlas_mean_seconds=candidate["program_latency_seconds"]["mean"],
            mean_latency_reduction_pct=100*(1-candidate["program_latency_seconds"]["mean"]/
                                             baseline["program_latency_seconds"]["mean"]),
            paper_token_latency_reduction_pct=100*(1-candidate["paper_sequential_token_latency"]["mean"]/
                                                     baseline["paper_sequential_token_latency"]["mean"]),
            output_throughput_gain_pct=100*(candidate["output_tokens_per_second"]/
                                             baseline["output_tokens_per_second"]-1),
            p95_latency_reduction_pct=100*(1-candidate["program_latency_seconds"]["p95"]/
                                            baseline["program_latency_seconds"]["p95"])))
    aggregate = []
    for rate in sorted({p["rate"] for p in pairs}):
        selected = [p for p in pairs if p["rate"] == rate]
        if {p["repeat"] for p in selected} != set(range(expected_repeats)):
            raise ValueError(f"expected {expected_repeats} repeats at rate {rate}")
        result = dict(rate=rate, repeats=len(selected), all_outputs_match=all(p["matched"] for p in selected))
        for key in ("mean_latency_reduction_pct", "paper_token_latency_reduction_pct",
                    "output_throughput_gain_pct", "p95_latency_reduction_pct"):
            values = [p[key] for p in selected]
            result[key] = dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))
        aggregate.append(result)
    return dict(pairs=pairs, aggregate=aggregate,
                interpretation="positive reduction/gain favors Autellix; no confidence or generalization claim")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--expected-rates", default="0.1,1.0")
    args = p.parse_args()
    records = json.loads(Path(args.input).read_text())
    if {r["rate"] for r in records} != set(map(float,args.expected_rates.split(","))):
        raise ValueError("missing or unexpected arrival-rate group")
    result = summarize(records, args.repeats)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
