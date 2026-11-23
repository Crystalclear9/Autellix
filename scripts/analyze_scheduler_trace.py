"""Summarize GPU scheduler events without counting warmup as workload."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(path, request_ids=None, workload_window=None):
    with Path(path).open(encoding="utf-8") as stream:
        return _summarize_events(stream, request_ids, workload_window)


def _summarize_events(stream, request_ids, workload_window):
    counts, queues, admissions, batch_sizes = Counter(), Counter(), Counter(), Counter()
    timing, transfer_blocks, transfer_copies = Counter(), Counter(), Counter()
    model_seconds, occupied_seconds, swapped_tokens = 0., 0., 0
    logical_prefill_tokens = cached_prefill_tokens = 0
    for line in stream:
        event = json.loads(line)
        if workload_window is not None and not workload_window[0] <= event["time"] <= workload_window[1]:
            continue
        if request_ids is not None and workload_window is None:
            # Internal request IDs are present in each completed call's metrics.
            ids = set(event.get("requests", []))
            if "rid" in event:
                ids.add(event["rid"])
            if ids and not ids.intersection(request_ids):
                continue
        counts[event["event"]] += 1
        if event["event"] == "execute":
            size, seconds = len(event["requests"]), event["seconds"]
            model_seconds += seconds
            occupied_seconds += size * seconds
            batch_sizes[size] += 1
        elif event["event"] == "swap_out":
            swapped_tokens += event.get("tokens", 0)
        elif event["event"] == "demote":
            queues[event["queue"]] += 1
        elif event["event"] == "admit":
            admissions[event["queue"]] += 1
        elif event["event"] == "schedule_time":
            timing["schedule"] += event["seconds"]
            logical_prefill_tokens += event.get("logical_prefill_tokens", 0)
            cached_prefill_tokens += event.get("cached_prefill_tokens", 0)
        elif event["event"] == "model_phase_time":
            timing[event["phase"]] += event["seconds"]
        elif event["event"] == "kv_transfer":
            timing["swap_" + event["direction"]] += event["seconds"]
            transfer_blocks[event["direction"]] += event["blocks"]
            if "copies" in event:
                transfer_copies[event["direction"]] += event["copies"]
    prefill = (dict(logical=logical_prefill_tokens, reused=cached_prefill_tokens,
                    computed=logical_prefill_tokens - cached_prefill_tokens) if logical_prefill_tokens else {})
    return dict(events=dict(counts), admissions_by_queue=dict(admissions), demotions_by_queue=dict(queues),
                batches_by_size=dict(batch_sizes), model_seconds=model_seconds,
                service_weighted_batch_size=occupied_seconds/model_seconds if model_seconds else None,
                swapped_out_tokens=swapped_tokens, timing_seconds=dict(timing),
                prefill_tokens=prefill,
                kv_transfer_blocks=dict(transfer_blocks), kv_transfer_copies=dict(transfer_copies))


def timing_per_token(row, replicas):
    tokens = row["completion_tokens"]
    totals = Counter()
    for replica in replicas.values():
        totals.update(replica["timing_seconds"])
    waits = [call["metrics"].get("waited") for call in row["calls"]]
    execution = sum(totals.pop(phase, 0.) for phase in ("prefill", "decode", "mixed"))
    return dict(execution_device_seconds=execution / tokens, scheduler_device_seconds=totals["schedule"] / tokens,
                swap_device_seconds=(totals["swap_in"] + totals["swap_out"]) / tokens,
                queue_call_seconds=sum(waits) / tokens if waits and all(w is not None for w in waits) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="run_sglang_matrix.py output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-root", type=Path, help="directory containing archived trial state directories")
    args = parser.parse_args()
    summaries = []
    for row in json.loads(Path(args.results).read_text(encoding="utf-8")):
        ids = {call["metrics"]["rid"] for call in row["calls"] if "rid" in call["metrics"]}
        state = Path(row["state_dir"])
        if args.state_root:
            state = args.state_root / state.name
        window = ((row["workload_started_monotonic"], row["workload_finished_monotonic"])
                  if "workload_started_monotonic" in row else None)
        replicas = {path.stem: analyze(path, ids, window) for path in sorted(state.glob("replica-*.jsonl"))}
        summary = dict(replicas.get("replica-0", dict(events={})))
        if len(replicas) > 1:
            summary["replicas"] = replicas
        if replicas and row.get("completion_tokens") and any(r["timing_seconds"] for r in replicas.values()):
            summary["timing_seconds_per_output_token"] = timing_per_token(row, replicas)
        summaries.append(dict(condition=row["condition"], rate=row["rate"], repeat=row["repeat"], **summary))
    Path(args.output).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
