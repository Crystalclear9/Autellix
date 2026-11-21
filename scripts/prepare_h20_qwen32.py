"""Prepare single-H20 Qwen3-32B replay plans using the existing ShareGPT and agent traces."""
import argparse
import json
from pathlib import Path
import random
import re

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
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + call["max_tokens"] >= 32768:
                raise ValueError(f"{kind} prompt exceeds Qwen32 replay context: {program['program_id']}")
            call.update(prompt=prompt, input_ids=ids, input_tokens=len(ids))
    return items


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, type=int, help="one physical H20 GPU index")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be a nonnegative GPU index")

    from transformers import AutoTokenizer
    model = args.model.resolve(strict=True)
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
    lats = agent_replay(args.datasets / "lats-captured6.json", tokenizer, "lats")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    engine_args = dict(dtype="bfloat16", context_length=32768, tp_size=1, pp_size=1, dp_size=1,
                       max_running_requests=8, max_total_tokens=65536, max_prefill_tokens=32768,
                       mem_fraction_static=.88, random_seed=0, attention_backend="triton",
                       sampling_backend="pytorch")
    native = dict(name="native", policy="sglang-native")
    upstream = dict(name="native-default", policy="sglang-native", engine_args=dict(
        autellix_native_defaults=True, attention_backend=None, sampling_backend=None))
    mix_rng = random.Random(2026)
    mixed = [program for pool in (sharegpt, bfcl, lats) for program in mix_rng.sample(pool, 2)]
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
