"""Summarize paired trials, per-GPU activity and pipeline trace agreement."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from integrations.sglang.scheduler import encode_request


def summarize_trial(row, trace_root=None):
    traces = []
    stage_checks = []
    directory = Path(trace_root) / Path(row["state_dir"]).name if trace_root else Path(row["state_dir"])
    for trace in sorted(directory.glob("replica-*.jsonl")):
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        traces.extend(events)
        selected = {"batch", "execute", "finished", "cancelled", "swap_out", "swap_in"}
        expected = [event for event in events if event["event"] in selected]
        digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        for stage in sorted(trace.parent.glob(trace.name + ".stage-*")):
            actual = [json.loads(line) for line in stage.read_text().splitlines()]
            stage_checks.append(dict(file=stage.name, matches=actual == expected,
                                     events=len(actual), leader_sha256=digest))
    gpu_samples = defaultdict(list)
    for sample in row.get("gpu_samples", []):
        for gpu in sample["gpus"]:
            gpu_samples[gpu["index"]].append(gpu)
    utilization = {}
    for index, samples in gpu_samples.items():
        available = [s["utilization"] for s in samples if s["utilization"] is not None]
        utilization[index] = dict(peak_memory_mib=max(s["used_mib"] for s in samples),
                                  total_mib=samples[0]["total_mib"],
                                  mean_utilization=sum(available) / len(available) if available else None,
                                  peak_utilization=max(available, default=None), samples=len(samples))
    calls_by_engine = Counter(call["engine_id"] for call in row["calls"])
    request_ids = {encode_request(call["program_id"], call["request_id"]) for call in row["calls"]}
    workload_events = [event for event in traces if event.get("rid") in request_ids
                       or request_ids.intersection(event.get("requests", []))]
    executions = [event for event in workload_events if event["event"] == "execute"]
    swaps = [event for event in workload_events if event["event"] == "swap_out"]
    admits = [event for event in workload_events if event["event"] == "admit"]
    steps = [event for event in workload_events if event["event"] == "plan_step"]
    ttfts = [call["ttft_seconds"] for call in row["calls"] if call.get("ttft_seconds") is not None]
    programs = {}
    for pid, latency in row["programs"].items():
        calls = [call for call in row["calls"] if call["workload_program"] == pid]
        programs[pid] = dict(response_seconds=latency,
                             arrival_seconds=row["arrival_times"][pid],
                             output_tokens=sum(call["completion_tokens"] for call in calls),
                             calls=len(calls),
                             calls_by_engine=dict(Counter(call["engine_id"] for call in calls)),
                             input_tokens=sum(call["prompt_tokens"] for call in calls))
    return dict(condition=row["condition"], arm=row["arm"], seed=row["seed"], rate=row["rate"],
                repeat=row["repeat"], model=row["plan"]["model"], device_groups=row["device_groups"],
                routing=row["routing"],
                trace_sha256=row["trace_sha256"], programs=len(row["programs"]), calls=len(row["calls"]),
                output_tokens=sum(c["completion_tokens"] for c in row["calls"]),
                output_tokens_per_second=row["output_tokens_per_second"],
                program_latency_seconds=row["program_latency_seconds"],
                paper_sequential_token_latency=row["paper_sequential_token_latency"],
                calls_by_engine=dict(calls_by_engine), gpu=utilization,
                per_program=programs,
                peak_total_gpu_memory_mib=max((sum(g["used_mib"] for g in sample["gpus"])
                                              for sample in row.get("gpu_samples", [])), default=None),
                trace_events=dict(Counter(event["event"] for event in workload_events)),
                admission_queues=dict(Counter(event["queue"] for event in admits)),
                swapped_tokens=sum(event.get("tokens", 0) for event in swaps),
                model_input_tokens=(sum(event["input_tokens"] for event in steps)
                                    if steps and all("input_tokens" in event for event in steps) else None),
                mean_call_ttft_seconds=sum(ttfts)/len(ttfts) if ttfts else None,
                summed_gpu_service_seconds=sum(event["seconds"] for event in executions),
                mean_executed_batch_size=(sum(len(event["requests"]) for event in executions)/len(executions)
                                          if executions else None),
                elapsed_seconds=row["elapsed_seconds"],
                arrival_horizon_seconds=row["arrival_horizon_seconds"],
                drain_seconds=row["drain_seconds"],
                completions_during_arrivals=row["completions_during_arrivals"],
                pipeline_stage_checks=stage_checks, versions=row["versions"],
                engine_args=row["effective_engine_args"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-root", help="directory of archived state folders")
    args = parser.parse_args()
    raw = [row for path in args.input for row in json.loads(Path(path).read_text())]
    rows = [summarize_trial(row, args.trace_root) for row in raw]
    pairs = []
    for row, summary in zip(raw, rows):
        if row["arm"] == "sglang-native":
            continue
        baseline = next(other for other in raw if other["arm"] == "sglang-native"
                        and other["plan"]["model"] == row["plan"]["model"]
                        and other["device_groups"] == row["device_groups"]
                        and other["seed"] == row["seed"] and other["rate"] == row["rate"])
        assert row["trace_sha256"] == baseline["trace_sha256"]
        assert summary["output_tokens"] == sum(c["completion_tokens"] for c in baseline["calls"])
        baselines = [baseline]
        if row["routing"] == "program":
            baselines += [other for other in raw if other["arm"] == row["arm"]
                          and other["routing"] in {"least-used", "round-robin"}
                          and other["plan"]["model"] == row["plan"]["model"]
                          and other["seed"] == row["seed"] and other["rate"] == row["rate"]]
        for baseline in baselines:
            pairs.append(compare(row, baseline))
    Path(args.output).write_text(json.dumps(dict(trials=rows, comparisons=pairs), indent=2))
    print(json.dumps(pairs, indent=2))


def compare(row, baseline):
    assert row["trace_sha256"] == baseline["trace_sha256"]
    assert row["output_hashes"].keys() == baseline["output_hashes"].keys()
    def lengths(trial):
        return {(call["workload_program"], call["workload_call"]):
                (call["prompt_tokens"], call["completion_tokens"]) for call in trial["calls"]}
    assert lengths(row) == lengths(baseline), "paired call input/output lengths differ"
    return dict(condition=row["condition"], baseline=baseline["condition"],
            model=row["plan"]["model"], seed=row["seed"], rate=row["rate"],
            mean_program_latency_change_pct=100*(row["program_latency_seconds"]["mean"]/
                                                baseline["program_latency_seconds"]["mean"]-1),
            paper_token_latency_change_pct=100*(row["paper_sequential_token_latency"]["mean"]/
                                               baseline["paper_sequential_token_latency"]["mean"]-1),
            throughput_change_pct=100*(row["output_tokens_per_second"]/baseline["output_tokens_per_second"]-1),
            matching_output_hashes=sum(value == baseline["output_hashes"][key]
                                       for key, value in row["output_hashes"].items()),
            calls=len(row["calls"]))


if __name__ == "__main__":
    main()
