"""Exercise new high-priority prefills competing with restored GPU prefixes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from autellix.runtime import InferenceEngine, ReplicaConfig, PolicyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-kv-recompute", action="store_true")
    parser.add_argument("--cache-paused-kv", action="store_true")
    parser.add_argument("--fused-swap", action="store_true")
    parser.add_argument("--chunked-prefill-size", type=int)
    parser.add_argument("--modern", action="store_true", help="use SGLang's Triton path")
    parser.add_argument("--mem-fraction-static", type=float, default=.35)
    args = parser.parse_args()
    results = []
    for reserve in (0, 1):
        options = dict(autellix_legacy_gpu=True, dtype="float32", context_length=512,
                       max_total_tokens=512, max_prefill_tokens=512, max_running_requests=2,
                       mem_fraction_static=args.mem_fraction_static, random_seed=0, autellix_swap_space=.25)
        if args.modern:
            options.update(autellix_legacy_gpu=False, dtype="float16", sampling_backend="pytorch")
        if args.host_kv_recompute:
            options.update(autellix_swap_space=.00001, autellix_host_kv_recompute=True)
        if args.cache_paused_kv:
            options.update(autellix_cache_paused_kv=True, autellix_window_activity=True)
        if args.fused_swap:
            options["autellix_fused_swap"] = True
        if args.chunked_prefill_size is not None:
            options["chunked_prefill_size"] = args.chunked_prefill_size
        policy = PolicyConfig(policy="plas", schedule_interval=1, overprovision=reserve,
                              boundaries=(0, .001, .002, math.inf), quanta=(.0001, .0002, .0004), beta=1e12)
        with InferenceEngine([ReplicaConfig("sglang", args.model, args.device, options)], policy=policy) as engine:
            a, b = engine.start_session(), engine.start_session()
            sampling = dict(temperature=0, max_tokens=96, ignore_eos=True)
            first = engine.submit(a, input_ids=[42]*300, sampling=sampling)
            first.progress.get(timeout=120)
            second = engine.submit(b, input_ids=[43]*300,
                                   sampling=dict(temperature=0, max_tokens=16, ignore_eos=True))
            ra, rb = first.result(180), second.result(180)
            assert ra["completion_tokens"] == 96 and rb["completion_tokens"] == 16
            reference = engine.start_session()
            rr = engine.submit(reference, input_ids=[42]*300, sampling=sampling).result(180)
            assert ra["text"] == rr["text"]
            events = [json.loads(line) for line in (engine.state_dir / "replica-0.jsonl").read_text().splitlines()]
            counts = Counter(e["event"] for e in events)
            if args.chunked_prefill_size:
                assert counts["continue_prefill"] > 0, counts
            elif args.cache_paused_kv:
                assert counts["pause_cached"] > 0 and counts["swap_out"] == 0, counts
            elif args.host_kv_recompute:
                assert counts["recompute_retract"] > 0 and counts["swap_out"] == 0, counts
            else:
                assert counts["defer_prefill"] > 0, counts
                assert counts["swap_out"] > 0 and counts["swap_in"] > 0, counts
            for sid in (a, b, reference):
                engine.end_session(sid)
            engine.wait_idle(30)
            assert engine.table.describe() == {}
            result = dict(reserve=reserve, state_dir=str(engine.state_dir), events=dict(counts),
                          resumed_greedy_matches=True, sessions_empty=True, completed_tokens=[96,16,96])
            results.append(result)
            print(json.dumps(result), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
