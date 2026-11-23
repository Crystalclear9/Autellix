"""Summarize complete paired paper-workload trials and arrival-window behavior."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


DEFAULT_SLO_TARGETS = (.01, .02, .04, .08, .1, .25, .5, 1., 2., 4., 8.)


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return dict.fromkeys(("mean", "median", "p95", "p99"))
    import math
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p95=values[math.ceil(.95 * len(values))-1], p99=values[math.ceil(.99 * len(values))-1])


def paper_token_latencies(row):
    kind = row["plan"].get("workload_kind", "sharegpt")
    parallel = row["plan"].get("workload_family", kind) in {"lats", "mixed"}
    token_times = (row["program_token_latencies"] if parallel
                   else row["response_seconds_per_output_token"])
    paper_metric = "critical_path/total_output_tokens" if parallel else "program_response/total_output_tokens"
    if row["plan"].get("workload_family") == "mixed" and "program_datasets" in row:
        token_times = {pid: (token_times[pid] if dataset == "lats" else row["response_seconds_per_output_token"][pid])
                       for pid, dataset in row["program_datasets"].items()}
        paper_metric = "per_program_response_or_critical_path/total_output_tokens"
    return token_times, paper_metric


def first_call_waits(calls):
    first_calls = {}
    for call in calls:
        if "submitted_seconds" not in call:
            continue
        pid = call["workload_program"]
        if pid not in first_calls or call["submitted_seconds"] < first_calls[pid]["submitted_seconds"]:
            first_calls[pid] = call
    waits = [call.get("metrics", {}) for call in first_calls.values()]
    waits = [m for m in waits if m.get("wait_before_first_batch") is not None and m.get("waited") is not None]
    if not waits:
        return {}
    return {"first_call_wait_seconds": dict(
        samples=len(waits),
        before_first_batch=distribution(m["wait_before_first_batch"] for m in waits),
        between_batches=distribution(m["waited"] - m["wait_before_first_batch"] for m in waits))}


def metrics(row):
    kind = row["plan"].get("workload_kind", "sharegpt")
    token_times, paper_metric = paper_token_latencies(row)
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
        if ids:
            quartiles.append(dict(programs=len(ids), mean_program_seconds=statistics.mean(row["programs"][p] for p in ids),
                                  mean_seconds_per_output_token=statistics.mean(token_times[p] for p in ids)))
    devices = {int(d) for group in row["device_groups"] for d in group.split(",")}
    gpu = {}
    for device in sorted(devices):
        samples = [g for s in row.get("gpu_samples", []) for g in s["gpus"] if g["index"] == device]
        if samples:
            utilization = [g["utilization"] for g in samples if g["utilization"] is not None]
            gpu[device] = dict(mean_utilization=statistics.mean(utilization) if utilization else None,
                               peak_memory_mib=max(g["used_mib"] for g in samples))
    cached = [c.get("sglang_cached_tokens", c.get("cached_prompt_tokens")) for c in row["calls"]]
    prompt_tokens = sum(c["prompt_tokens"] for c in row["calls"])
    cache = {}
    if prompt_tokens and all(c is not None for c in cached):
        cache["sglang_cached_tokens"] = sum(cached)
        cache["sglang_cached_tokens_per_prompt_token"] = sum(cached) / prompt_tokens
    return dict(workload=kind, workload_family=row["plan"].get("workload_family", kind),
                paper_figures=row["plan"].get("paper_figures", []),
                condition=row["condition"], rate=row["rate"], seed=row["seed"],
                model=row["model"], versions=row["versions"], engine_args=row["effective_engine_args"],
                devices=row["device_groups"], trace_sha256=row["trace_sha256"],
                programs=len(arrivals), calls=len(row["calls"]), output_tokens=row["completion_tokens"],
                input_tokens_per_call=distribution(c["prompt_tokens"] for c in row["calls"]),
                program_response_seconds=distribution(row["programs"].values()),
                paper_token_latency=distribution(token_times.values()),
                paper_metric=paper_metric,
                output_tokens_per_second=row["output_tokens_per_second"], completed_programs_per_second=len(arrivals)/row["elapsed_seconds"],
                elapsed_seconds=row["elapsed_seconds"],
                arrival_horizon_seconds=horizon, drain_seconds=row["drain_seconds"],
                completions_during_arrivals=row["completions_during_arrivals"],
                arrival_window_completed_programs_per_second=(row["completions_during_arrivals"] / horizon
                                                              if horizon > 0 else None),
                arrival_window_token_latencies=[token_times[pid] for pid, at in completed.items()
                                                if at <= horizon],
                maximum_submission_lag_seconds=max(row["first_submission_lag_seconds"].values()),
                arrival_quartiles=quartiles, backlog_checkpoints=checkpoints,
                calls_by_engine=dict(Counter(c["engine_id"] for c in row["calls"])), gpu=gpu,
                **cache, **first_call_waits(row["calls"]))


def paper_matrix(records, slo_targets=DEFAULT_SLO_TARGETS, *, strict=True):
    """Join figure points across runner files and across inference engines."""
    trials, at_load = [], defaultdict(dict)
    for row in records:
        plan = row["plan"]
        figures = tuple(plan.get("paper_figures", ()))
        if not set(figures).intersection({12, 13, 16}):
            continue
        backend = plan["backend"]
        arm = f"{backend}/{row['condition']}"
        opts = row["effective_engine_args"]
        if backend == "vllm":
            resources = (opts["dtype"], opts["max_model_len"], opts["max_num_seqs"],
                         opts["block_size"] * opts["num_gpu_blocks_override"])
        else:
            resources = (opts["dtype"], opts["context_length"], opts["max_running_requests"], opts["max_total_tokens"])
        config = (figures, plan["workload_family"], len(row["programs"]), row["model"],
                  tuple(row["device_groups"]), resources)
        latencies, _ = paper_token_latencies(row)
        make = row["elapsed_seconds"]
        horizon = row["arrival_horizon_seconds"]
        point = dict(figures=figures, workload=plan["workload_family"], backend=backend,
                     condition=row["condition"], policy=row.get("arm"), model=row["model"], devices=row["device_groups"],
                     resources=dict(zip(("dtype", "context_tokens", "batch_size", "kv_tokens"), resources)),
                     versions=row["versions"], engine_args=opts,
                     programs=len(row["programs"]), rate=row["rate"], seed=row["seed"],
                     planned_repeats=plan.get("repeats", 1),
                     trace_sha256=row["trace_sha256"], makespan_seconds=make,
                     token_latency=distribution(latencies.values()), output_tokens=row["completion_tokens"],
                     completed_programs_per_second=len(row["programs"]) / make,
                     arrival_window_completed_programs_per_second=(row["completions_during_arrivals"] / horizon
                                                                   if horizon else None),
                     **first_call_waits(row["calls"]))
        trials.append(point)
        slot = at_load[config, row["rate"], row["seed"]]
        if arm in slot:
            raise ValueError(f"duplicate figure point: {config}, {arm}, {row['rate']}, {row['seed']}")
        lengths = sorted((c["workload_program"], c["workload_call"], c["prompt_tokens"], c["completion_tokens"])
                         for c in row["calls"])
        slot[arm] = hashlib.sha256(json.dumps(lengths).encode()).hexdigest(), point

    pairs, unmatched = [], []
    reference_arms = ("vllm/vllm", "vllm/vllm-opt", "vllm/vllm-opt-multistep", "vllm/fcfs-combined",
                      "vllm/fcfs-control",
                      "vllm/mlfq", "sglang/sglang-native", "sglang/fcfs-combined", "sglang/mlfq")
    for slot in at_load.values():
        variants = [arm for arm, (_, point) in slot.items()
                    if point["policy"] in {"fcfs", "mlfq"} and arm not in reference_arms]
        for baseline_arm in (*reference_arms, *variants):
            if baseline_arm not in slot:
                continue
            reference_lengths, reference = slot[baseline_arm]
            for arm, (lengths, point) in slot.items():
                if point["condition"] != "autellix" and point["policy"] not in {"plas", "atlas"}:
                    continue
                reason = None
                if point["trace_sha256"] != reference["trace_sha256"] or lengths != reference_lengths:
                    reason = "mismatched trace or work"
                elif point["backend"] == reference["backend"] and point["versions"] != reference["versions"]:
                    reason = "mismatched backend versions"
                if reason:
                    if strict:
                        raise ValueError(f"{reason}: {arm} versus {baseline_arm}")
                    unmatched.append(dict(candidate=arm, baseline=baseline_arm, reason=reason, figures=point["figures"],
                                          workload=point["workload"], rate=point["rate"], seed=point["seed"]))
                    continue
                ratios = {stat: 100 * (value / reference["token_latency"][stat] - 1)
                          for stat, value in point["token_latency"].items()
                          if value is not None and reference["token_latency"][stat]}
                pairs.append(dict(candidate=arm, baseline=baseline_arm, figures=point["figures"],
                                  workload=point["workload"], programs=point["programs"], rate=point["rate"],
                                  seed=point["seed"], trace_sha256=point["trace_sha256"],
                                  makespan_change_pct=100 * (point["makespan_seconds"] / reference["makespan_seconds"] - 1),
                                  completion_throughput_change_pct=100 * (point["completed_programs_per_second"] /
                                                                       reference["completed_programs_per_second"] - 1),
                                  program_token_latency_change_pct=ratios))
    by_config = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (config, rate, _), slot in at_load.items():
        if rate is not None:
            for arm, (_, point) in slot.items():
                by_config[config][arm][rate].append(point)
    slo = []
    for config, arms in by_config.items():
        figures, workload, programs, model, devices, _ = config
        for target in slo_targets:
            for stat in ("mean", "p95", "p99"):
                offered, completed, in_window = {}, {}, {}
                for arm, grouped in arms.items():
                    qualifying = {rate: rows for rate, rows in grouped.items()
                                  if all(len(rows) == row["planned_repeats"] and row["token_latency"][stat] is not None
                                         and row["token_latency"][stat] <= target for row in rows)}
                    offered[arm] = max(qualifying, default=None)
                    completed[arm] = max((statistics.mean(row["completed_programs_per_second"] for row in rows)
                                          for rows in qualifying.values()), default=None)
                    in_window[arm] = max((statistics.mean(row["arrival_window_completed_programs_per_second"] for row in rows)
                                          for rows in qualifying.values() if all(row["arrival_window_completed_programs_per_second"]
                                                                                 is not None for row in rows)), default=None)
                slo.append(dict(figures=figures, workload=workload, programs=programs, model=model, devices=devices,
                                latency_statistic=stat, seconds_per_token_target=target,
                                maximum_tested_offered_programs_per_second=offered,
                                completed_programs_per_second=completed,
                                arrival_window_completed_programs_per_second=in_window))
    return dict(trials=trials, comparisons=pairs, unmatched=unmatched, slo_points=slo)


def summarize(records, slo_targets=DEFAULT_SLO_TARGETS):
    groups = {}
    for row in records:
        kind = row["plan"].get("workload_kind", "sharegpt")
        key = (kind, row["rate"], row["seed"], tuple(row["device_groups"]), row["model"],
               tuple(sorted(row["versions"].items())))
        group = groups.setdefault(key, {})
        if row["condition"] in group:
            raise ValueError(f"duplicate trial: {key}, {row['condition']}")
        group[row["condition"]] = row
    trials, comparisons = [], []
    for key, group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or 0, *item[0][2:])):
        by_case = {name: metrics(row) for name, row in group.items()}
        trials.extend(by_case.values())
        native = next((row for row in group.values() if row["arm"] == "sglang-native"
                       and not row["effective_engine_args"].get("autellix_native_defaults")), None)
        if native is None:
            native = next((row for row in group.values() if row["arm"] == "sglang-native"), None)
        if native is None:
            native = next((group[name] for name in ("vllm", "vllm-opt", "vllm-opt-multistep", "program", "native-swap")
                           if name in group), None)
        if native is None:
            continue
        baseline = by_case[native["condition"]]
        for row in group.values():
            if row is native:
                continue
            candidate = by_case[row["condition"]]
            def lengths(r):
                return {(c["workload_program"], c["workload_call"]):
                        (c["prompt_tokens"], c["completion_tokens"]) for c in r["calls"]}
            def options(r):
                # Native SGLang does not allocate the Autellix host KV cache.
                return {k: v for k, v in r["effective_engine_args"].items()
                        if k not in {"autellix_native_baseline", "autellix_swap_space", "autellix_host_kv_recompute",
                                     "autellix_cache_paused_kv", "autellix_window_activity", "autellix_refill_idle_slots",
                                     "autellix_resident_reserve_only", "resident_reserve_only",
                                     "autellix_fused_swap", "fused_swap",
                                     "autellix_retain_running", "autellix_mix_prefill_decode",
                                     "autellix_mode", "preemption_mode", "observe_native", "batched_swap",
                                     "keep_resident_kv", "recompute_on_swap_full", "paper_chunked_prefill",
                                     "publish_activity_at_window", "refill_idle_slots", "resident_decode_headroom",
                                     "min_decode_steps", "mixed_decode_quantum", "deduplicate_prefill"}}
            native_defaults = row["arm"] == "sglang-native" and row["effective_engine_args"].get("autellix_native_defaults", False)
            baseline_defaults = native["effective_engine_args"].get("autellix_native_defaults", False)
            row_options, baseline_options = options(row), options(native)
            if native_defaults or baseline_defaults:
                for opt in ("autellix_native_defaults", "attention_backend", "sampling_backend", "disable_cuda_graph",
                            "chunked_prefill_size"):
                    row_options.pop(opt, None)
                    baseline_options.pop(opt, None)
            paired = dict(same_trace=row["trace_sha256"] == native["trace_sha256"],
                          same_lengths=lengths(row) == lengths(native), same_options=row_options == baseline_options,
                          same_model=row["model"] == native["model"], same_versions=row["versions"] == native["versions"])
            if not all(paired.values()):
                raise ValueError(f"unmatched workload or resources: {key}, {paired}")
            per_program, calls_by_program = [], defaultdict(list)
            for call in row["calls"]:
                calls_by_program[call["workload_program"]].append(call)
            for pid, response in row["programs"].items():
                calls = calls_by_program[pid]
                per_program.append(dict(program=pid, calls=len(calls),
                    output_tokens=sum(c["completion_tokens"] for c in calls),
                    max_input_tokens=max(c["prompt_tokens"] for c in calls),
                    native_seconds=native["programs"][pid], candidate_seconds=response,
                    response_delta_seconds=response-native["programs"][pid]))
            comparisons.append(dict(workload=key[0], rate=key[1], seed=key[2], devices=key[3], model=row["model"],
                condition=row["condition"],
                baseline_condition=native["condition"], upstream_default_optimizations=native_defaults or baseline_defaults,
                pairing=paired, programs=len(row["programs"]),
                per_program=sorted(per_program, key=lambda p: -p["response_delta_seconds"]),
                faster_programs=sum(row["programs"][p] < native["programs"][p] for p in row["programs"]),
                matching_texts=sum(native["output_hashes"][c] == text for c, text in row["output_hashes"].items()),
                calls=len(row["calls"]),
                mean_response_change_pct=100*(candidate["program_response_seconds"]["mean"]/baseline["program_response_seconds"]["mean"]-1),
                mean_paper_latency_change_pct=100*(candidate["paper_token_latency"]["mean"]/baseline["paper_token_latency"]["mean"]-1),
                p95_paper_latency_change_pct=100*(candidate["paper_token_latency"]["p95"]/baseline["paper_token_latency"]["p95"]-1),
                p99_paper_latency_change_pct=100*(candidate["paper_token_latency"]["p99"]/baseline["paper_token_latency"]["p99"]-1),
                throughput_change_pct=100*(candidate["output_tokens_per_second"]/baseline["output_tokens_per_second"]-1)))
    configurations = defaultdict(list)
    for trial in trials:
        if trial["rate"] is not None:
            ranks = tuple(len(device.split(",")) for device in trial["devices"])
            configurations[trial["workload"], trial["model"], ranks, tuple(sorted(trial["versions"].items()))].append(trial)
    slo_points = []
    tail_slo_points = []
    for (workload, model, ranks, versions), selected in sorted(configurations.items()):
        by_condition_and_rate = defaultdict(lambda: defaultdict(list))
        for trial in selected:
            by_condition_and_rate[trial["condition"]][trial["rate"]].append(trial)
        for target in slo_targets:
            qualified = {
                condition: {rate: rows for rate, rows in rates.items()
                            if all(row["paper_token_latency"]["mean"] <= target for row in rows)}
                for condition, rates in sorted(by_condition_and_rate.items())}
            slo_points.append(dict(workload=workload, model=model, ranks_per_replica=ranks, versions=dict(versions),
                                   mean_seconds_per_output_token_target=target,
                                   maximum_tested_rate_meeting_latency={
                                       case: max(rates, default=None) for case, rates in qualified.items()},
                                   maximum_completed_rate_meeting_latency={
                                       case: max((statistics.mean(trial["completed_programs_per_second"] for trial in rows)
                                                  for rows in rates.values()), default=None)
                                       for case, rates in qualified.items()}))
            by_statistic = {}
            for statistic in ("mean", "p95", "p99"):
                observed = {}
                for condition, rates in by_condition_and_rate.items():
                    matches = [rows for rows in rates.values()
                               if all(row["paper_token_latency"][statistic] <= target for row in rows)]
                    observed[condition] = max((statistics.mean(row["arrival_window_completed_programs_per_second"]
                                                               for row in rows) for rows in matches
                                               if all(row["arrival_window_completed_programs_per_second"] is not None
                                                      for row in rows)), default=None)
                by_statistic[statistic] = observed
            tail_slo_points.append(dict(workload=workload, model=model, ranks_per_replica=ranks,
                                        versions=dict(versions), seconds_per_token_target=target,
                                        maximum_arrival_window_completion_rate=by_statistic,
                                        maximum_arrival_window_goodput={
                                            condition: max((statistics.mean(
                                                sum(v is not None and v <= target for v in row["arrival_window_token_latencies"])
                                                / row["arrival_horizon_seconds"] for row in rows)
                                                for rows in rates.values() if all(row["arrival_horizon_seconds"] > 0
                                                                                 for row in rows)), default=None)
                                            for condition, rates in by_condition_and_rate.items()}))
    for trial in trials:
        trial.pop("arrival_window_token_latencies")
    return dict(trials=trials, comparisons=comparisons, slo_points=slo_points, observed_slo_points=tail_slo_points,
                paper_matrix=paper_matrix(records, slo_targets),
                slo_definition="Largest measured offered rate with mean program token latency within target in every measured repeat; completed rates average repeats at each rate. Finite-trace backlog and drain are reported separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--slo", type=float, nargs="+", default=DEFAULT_SLO_TARGETS,
                        help="mean program seconds per token targets")
    parser.add_argument("--matrix-only", action="store_true", help="join figure results across separate runs")
    args = parser.parse_args()
    rows = (row for file in args.inputs for row in json.loads(Path(file).read_text(encoding="utf-8")))
    result = paper_matrix(rows, slo_targets=args.slo) if args.matrix_only else summarize(list(rows), slo_targets=args.slo)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
