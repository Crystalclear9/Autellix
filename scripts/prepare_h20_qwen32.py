"""Prepare single-H20 Qwen3-32B replay plans using the existing ShareGPT and agent traces."""
import argparse
import json
import math
from pathlib import Path
import random
import re
import tempfile

from prepare_sharegpt_replay import convert, reconstruct_conversations


LLAMA_HEADER = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\s*")


def bfcl_prompt(prompt, tokenizer):
    chunks = LLAMA_HEADER.split(prompt)
    messages = []
    for role, text in zip(chunks[1::2], chunks[2::2]):
        body = text.strip().removesuffix("<|eot_id|>").removesuffix("<|eom_id|>").strip()
        if role == "assistant" and not body:
            continue
        messages.append(dict(role="tool" if role == "ipython" else role, content=body))
    if not messages:
        return prompt
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def agent_replay(path, tokenizer, kind):
    items = json.loads(path.read_bytes())
    for program in items:
        program["dataset"] = kind
        for call in program["calls"]:
            prompt = call["prompt"]
            if kind == "bfcl":
                prompt = bfcl_prompt(prompt, tokenizer)
            elif kind == "lats":
                prompt = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + call["max_tokens"] >= 32768:
                raise ValueError(f"{kind} prompt exceeds Qwen32 replay context: {program['program_id']}")
            call.update(prompt=prompt, input_ids=ids, input_tokens=len(ids))
    return items


def write_json(path, value):
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    if path.is_file() and path.read_bytes() == data:
        return
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            staged = Path(stream.name)
            stream.write(data)
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def write_comparison_plans(model, inputs, destination, device, bfcl_rates,
                           resident_decode_headroom, qwen3_fused_qk_rope, exact_prefix_cache,
                           programs, seed, graph_context, completion_only, capacity, fcfs_control,
                           *, repeats=1, quantum_scale=16, boundary_scale=128, beta=8.0, min_decode_steps=0,
                           paper_vllm_only=False):
    destination.mkdir(parents=True, exist_ok=True)
    context, batch, kv_tokens, block = 32768, 2048, 65536, 16
    vllm = dict(dtype="bfloat16", max_model_len=context, max_num_seqs=capacity,
                max_num_batched_tokens=batch, block_size=block, tensor_parallel_size=1,
                num_gpu_blocks_override=kv_tokens // block, gpu_memory_utilization=.93,
                swap_space=16, enforce_eager=False, seed=0, publish_activity=False)
    if qwen3_fused_qk_rope:
        vllm["qwen3_fused_qk_rope"] = True
    if exact_prefix_cache:
        vllm["exact_prefix_cache"] = True
    sglang = dict(dtype="bfloat16", context_length=context, max_running_requests=capacity,
                  max_total_tokens=kv_tokens, chunked_prefill_size=batch, max_prefill_tokens=batch,
                  tp_size=1, pp_size=1, dp_size=1, mem_fraction_static=.93, random_seed=0,
                  autellix_native_defaults=True)
    if completion_only:
        vllm["completion_only"] = sglang["completion_only"] = True
    paper = dict(paper_chunked_prefill=True, keep_resident_kv=True, refill_idle_slots=True,
                 resident_reserve_only=True, recompute_on_swap_full=True, fused_swap=True)
    if resident_decode_headroom:
        paper["resident_decode_headroom"] = True
    native = dict(name="vllm-opt", policy="vllm-opt", interval=1,
                  engine_args=dict(preemption_mode=None))
    plas = dict(name="autellix", policy="plas", reserve=1, interval=8, quantum_scale=quantum_scale,
                boundary_scale=boundary_scale, engine_args=paper)
    if beta != 8.0:
        plas["beta"] = beta
    vllm_cases = [native, plas]
    if paper_vllm_only:
        whole = dict(preemption_mode=None, max_num_batched_tokens=context)
        vanilla = dict(native, name="vllm", policy="vllm", engine_args=dict(whole, exact_prefix_cache=False))
        multistep = dict(native, name="vllm-opt-multistep", policy="vllm-opt-multistep",
                         interval=plas["interval"], engine_args=whole)
        vllm_cases = [vanilla, native, multistep, dict(plas, name="mlfq", policy="mlfq"), plas]
    if fcfs_control:
        vllm_cases.append(dict(plas, name="fcfs-control", policy="fcfs"))
    if min_decode_steps:
        vllm_cases.append(dict(plas, name="plas-decode-window",
                               engine_args=dict(paper, min_decode_steps=min_decode_steps)))
    for family, rates, figures in (("bfcl", bfcl_rates, [12, 13]), ("sharegpt", [None], [16])):
        trace = (inputs / f"qwen32-{family}-trace.json").resolve(strict=True)
        label = f"qwen32-{family}"
        common = dict(model=str(model), device=str(device), workload=str(trace), workload_family=family,
                      workload_kind=label, experiment="offline" if rates == [None] else "online",
                      programs=programs, unique_programs=True, rates=rates, seed=seed, repeats=repeats,
                      paper_figures=figures,
                      compact_calls=True, request_timeout=3600, scheduler_trace=False)
        capture = graph_context or (8192 if family == "sharegpt" else context)
        write_json(destination / f"{label}-vllm-plan.json", dict(
            common, backend="vllm", engine_args=dict(vllm, max_seq_len_to_capture=capture), cases=vllm_cases))
        if not paper_vllm_only:
            write_json(destination / f"{label}-sglang-plan.json", dict(
                common, backend="sglang", engine_args=sglang,
                cases=[dict(name="sglang-native", policy="sglang-native")]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--datasets", type=Path)
    sources.add_argument("--comparison-inputs", type=Path, help="reuse previously prepared Qwen3 traces")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, type=int, help="one physical H20 GPU index")
    parser.add_argument("--lats-trace", type=Path, help="retokenize this captured HotpotQA workload")
    parser.add_argument("--mixed-programs-per-source", type=int, default=2, help="sample this many programs from each Mixed source")
    parser.add_argument("--bfcl-rates", nargs="+", type=float, default=[.24],
                        help="BFCL offered loads in programs/s for comparison plans")
    parser.add_argument("--resident-decode-headroom", action="store_true",
                        help="let selected resident decodes share a forward with a long prefill")
    parser.add_argument("--qwen3-fused-qk-rope", action="store_true",
                        help="use fused Qwen3 Q/K normalization and RoPE in both vLLM cases")
    parser.add_argument("--exact-prefix-cache", action="store_true",
                        help="publish completed KV blocks and charge uncached prefill tokens")
    parser.add_argument("--comparison-programs", type=int, default=16,
                        help="distinct programs sampled from each comparison trace")
    parser.add_argument("--comparison-seed", type=int, default=2026)
    parser.add_argument("--comparison-repeats", type=int, default=1,
                        help="paired repeats with successive seeds and rotating case order")
    parser.add_argument("--graph-context", type=int,
                        help="override the vLLM CUDA graph context length in comparison plans")
    parser.add_argument("--completion-only", action="store_true",
                        help="return final completions and skip intermediate updates for both engines")
    parser.add_argument("--comparison-batch-size", type=int, default=8,
                        help="active requests per single-card comparison engine")
    parser.add_argument("--fcfs-control", action="store_true",
                        help="also run the Autellix execution path with FCFS priorities")
    parser.add_argument("--comparison-quantum-scale", type=float, default=16,
                        help="scale PLAS time slices in vLLM comparison plans")
    parser.add_argument("--comparison-boundary-scale", type=float, default=128,
                        help="scale the elapsed-service ranges used to assign queues")
    parser.add_argument("--comparison-beta", type=float, default=8,
                        help="set the PLAS wait-to-service ratio for priority promotion")
    parser.add_argument("--comparison-min-decode-steps", type=int, default=0,
                        help="add a PLAS comparison with this many decode steps required before demotion")
    parser.add_argument("--paper-vllm-only", action="store_true",
                        help="prepare vLLM, both vLLM-opt configurations, MLFQ and PLAS")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be a nonnegative GPU index")
    if any(not math.isfinite(rate) or rate <= 0 for rate in args.bfcl_rates):
        parser.error("BFCL rates must be finite and positive")
    if args.comparison_programs < 1:
        parser.error("--comparison-programs must be positive")
    if args.comparison_batch_size < 1:
        parser.error("--comparison-batch-size must be positive")
    if args.mixed_programs_per_source < 1:
        parser.error("--mixed-programs-per-source must be positive")
    if args.comparison_repeats < 1:
        parser.error("--comparison-repeats must be positive")
    if args.comparison_min_decode_steps < 0:
        parser.error("--comparison-min-decode-steps must be nonnegative")
    for option, value in (("--comparison-quantum-scale", args.comparison_quantum_scale),
                          ("--comparison-boundary-scale", args.comparison_boundary_scale),
                          ("--comparison-beta", args.comparison_beta)):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{option} must be finite and positive")
    if args.graph_context is not None and not 1 <= args.graph_context <= 32768:
        parser.error("--graph-context requires a value from 1 to 32768")

    model = args.model.resolve(strict=True)
    if args.comparison_inputs:
        write_comparison_plans(model, args.comparison_inputs, args.run_dir, args.device, args.bfcl_rates,
                               args.resident_decode_headroom, args.qwen3_fused_qk_rope, args.exact_prefix_cache,
                               args.comparison_programs, args.comparison_seed, args.graph_context,
                               args.completion_only, args.comparison_batch_size, args.fcfs_control,
                               repeats=args.comparison_repeats, quantum_scale=args.comparison_quantum_scale,
                               boundary_scale=args.comparison_boundary_scale, beta=args.comparison_beta,
                               min_decode_steps=args.comparison_min_decode_steps,
                               paper_vllm_only=args.paper_vllm_only)
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    original_template = tokenizer.chat_template
    tokenizer.chat_template = "{% set enable_thinking = false %}" + original_template
    source = args.datasets / "ShareGPT_V3_unfiltered_cleaned_split.json"
    rows = reconstruct_conversations(json.loads(source.read_bytes()))
    random.Random(42).shuffle(rows)
    sharegpt = []
    for row in rows:
        program = convert(row, tokenizer, 32767, 0)
        if program and len(program["calls"]) <= 80:
            sharegpt.append(program)
        if len(sharegpt) == 128:
            break
    if len(sharegpt) != 128:
        raise ValueError("ShareGPT must contain 128 eligible whole conversations")
    tokenizer.chat_template = original_template
    bfcl = agent_replay(args.datasets / "bfcl-captured64.json", tokenizer, "bfcl")
    lats = agent_replay(args.lats_trace or args.datasets / "lats-captured6.json", tokenizer, "lats")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    engine_args = dict(dtype="bfloat16", context_length=32768, tp_size=1, pp_size=1, dp_size=1,
                       max_running_requests=8, max_total_tokens=65536, max_prefill_tokens=32768,
                       mem_fraction_static=.88, random_seed=0, attention_backend="triton",
                       sampling_backend="pytorch")
    native = dict(name="native", policy="sglang-native")
    upstream = dict(name="native-default", policy="sglang-native", engine_args=dict(
        autellix_native_defaults=True, attention_backend=None, sampling_backend=None))
    mix_rng = random.Random(2026)
    mixed = [program for pool in (sharegpt, bfcl, lats)
             for program in mix_rng.sample(pool, args.mixed_programs_per_source)]
    ordered = [("sharegpt", "sharegpt", sharegpt, 24, [.06, .24], "plas"),
               ("bfcl", "bfcl", bfcl, 12, [.12, .48], "plas"),
               ("sharegpt-offline128", "sharegpt", sharegpt, 0, [None], "plas"),
               ("lats", "lats", lats, 0, [.02], "atlas"),
               ("mixed", "mixed", mixed, 0, [.03], "atlas")]
    for name, family, programs, count, rates, policy in ordered:
        label = f"qwen32-{name}"
        trace = args.run_dir / f"{label}-trace.json"
        if name == "sharegpt-offline128":
            trace = args.run_dir / "qwen32-sharegpt-trace.json"
        else:
            write_json(trace, programs)
        candidate = dict(name=policy, policy=policy, reserve=1, interval=8, quantum_scale=8,
                         boundary_scale=128, engine_args=dict(autellix_swap_space=32,
                                                             autellix_host_kv_recompute=True))
        plan = dict(backend="sglang", model=str(model), device=str(args.device),
                    workload=str(trace.resolve()), workload_family=family, workload_kind=label,
                    experiment="offline" if rates == [None] else "online", programs=count, rates=rates,
                    seed=2026, repeats=1, compact_calls=True, request_timeout=21600,
                    engine_args=engine_args, cases=[native, candidate, upstream])
        write_json(args.run_dir / f"{label}-plan.json", plan)
        print(f"{label}: {len(programs)} source programs, rates {rates}", flush=True)


if __name__ == "__main__":
    main()
