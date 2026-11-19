"""Summarize GPU scheduler events without counting warmup as workload."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(path, request_ids=None):
    counts, queues, admissions, batch_sizes = Counter(), Counter(), Counter(), Counter()
    timing = Counter()
    transfer_blocks = Counter()
    model_seconds, occupied_seconds, swapped_tokens = 0., 0., 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if request_ids is not None:
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
        elif event["event"] == "model_phase_time":
            timing[event["phase"]] += event["seconds"]
        elif event["event"] == "kv_transfer":
            timing["swap_" + event["direction"]] += event["seconds"]
            transfer_blocks[event["direction"]] += event["blocks"]
    return dict(events=dict(counts), admissions_by_queue=dict(admissions), demotions_by_queue=dict(queues),
                batches_by_size=dict(batch_sizes), model_seconds=model_seconds,
                service_weighted_batch_size=occupied_seconds/model_seconds if model_seconds else None,
                swapped_out_tokens=swapped_tokens, timing_seconds=dict(timing), kv_transfer_blocks=dict(transfer_blocks))


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
        replicas = {path.stem: analyze(path, ids) for path in sorted(state.glob("replica-*.jsonl"))}
        summary = dict(replicas.get("replica-0", dict(events={})))
        if len(replicas) > 1:
            summary["replicas"] = replicas
        summaries.append(dict(condition=row["condition"], rate=row["rate"], repeat=row["repeat"], **summary))
    Path(args.output).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
