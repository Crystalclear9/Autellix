"""Measurements for the paper's program wait and prefix locality figures."""
from collections import Counter, defaultdict
import statistics


def workload_histograms(programs):
    calls = [call for program in programs for call in program.calls]
    return dict(input_tokens=dict(sorted(Counter(c.prefill_tokens for c in calls).items())),
                output_tokens=dict(sorted(Counter(c.decode_tokens for c in calls).items())),
                calls_per_program=dict(sorted(Counter(len(p.calls) for p in programs).items())))


def call_concurrency(calls, horizon, interval=5):
    events = sorted((mark, delta) for call in calls for mark, delta in (
        (call["submitted_seconds"], 1), (call["completed_seconds"], -1)))
    samples, active, cursor = [], 0, 0
    for at in range(0, int(horizon) + 1, interval):
        while cursor < len(events) and events[cursor][0] <= at:
            active += events[cursor][1]
            cursor += 1
        samples.append(dict(seconds=at, active_calls=active))
    return samples


def blocking(row):
    programs = defaultdict(list)
    by_output_tokens, by_program_calls = defaultdict(list), defaultdict(list)
    for call in row["calls"]:
        metrics = call.get("metrics", {})
        if "waited" in metrics and metrics.get("executed", 0) > 0:
            sample = (metrics["waited"], metrics["executed"])
            programs[call["workload_program"]].append(sample)
            by_output_tokens[max(1, call["completion_tokens"]).bit_length() - 1].append(sample[0] / sample[1])
    for calls in programs.values():
        waiting, executing = map(sum, zip(*calls))
        by_program_calls[len(calls).bit_length() - 1].append(waiting / executing)

    def group(buckets):
        return [dict(lower=2**bucket, upper=2**(bucket+1), samples=len(values),
                     mean_wait_to_execution=statistics.mean(values)) for bucket, values in sorted(buckets.items())]

    waited = sum(wait for calls in programs.values() for wait, _ in calls)
    executed = sum(execution for calls in programs.values() for _, execution in calls)
    return dict(call_seconds_waiting=waited, call_seconds_executing=executed,
                wait_fraction=waited / (waited + executed) if executed else None,
                by_output_tokens=group(by_output_tokens), by_program_calls=group(by_program_calls))


def prefix_opportunities(programs, block_size=16):
    """Find reusable input blocks in earlier calls, before finite-cache eviction."""
    prefixes, previous_programs = {}, set()
    groups, totals = Counter(), Counter()
    for program in programs:
        current = set()
        for call in program["calls"]:
            tokens = call.get("input_ids")
            if not tokens:
                continue
            same = across = node = 0
            nodes = []
            for start in range(0, len(tokens) - block_size + 1, block_size):
                key = (node, tuple(tokens[start:start+block_size]))
                node = prefixes.setdefault(key, len(prefixes) + 1)
                nodes.append(node)
                if node in current:
                    same = start + block_size
                if node in previous_programs:
                    across = start + block_size
            bucket = len(tokens).bit_length() - 1
            if current:
                groups[bucket, "within"] += 1
                totals[bucket, "within"] += same / len(tokens)
            if previous_programs:
                groups[bucket, "across"] += 1
                totals[bucket, "across"] += across / len(tokens)
            current.update(nodes)
        previous_programs.update(current)
    return [dict(lower_input_tokens=2**bucket, upper_input_tokens=2**(bucket+1), scope=scope,
                 calls=groups[bucket, scope], mean_reusable_input_fraction=totals[bucket, scope] / groups[bucket, scope])
            for bucket, scope in sorted(groups)]


def motivation(rows):
    return [dict(condition=row["condition"], rate=row["rate"], seed=row["seed"],
                 concurrency=call_concurrency(row["calls"], row["arrival_horizon_seconds"]),
                 blocking=blocking(row)) for row in rows]
