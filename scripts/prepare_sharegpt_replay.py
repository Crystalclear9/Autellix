"""Reconstruct ShareGPT conversations and prepare fixed-input scheduling replay."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from collections import defaultdict


def reconstruct_conversations(rows):
    """ShareGPT's numeric ID suffix is the original message offset."""
    groups = defaultdict(dict)
    for row in rows:
        stem, separator, suffix = str(row["id"]).rpartition("_")
        pid, offset = (stem, int(suffix)) if separator and suffix.isdigit() else (str(row["id"]), 0)
        messages = groups[pid]
        for index, message in enumerate(row["conversations"], offset):
            if index in messages and messages[index] != message:
                raise ValueError(f"conflicting source fragments for {pid}, message {index}")
            messages[index] = message
    result = []
    for pid, messages in sorted(groups.items()):
        if sorted(messages) != list(range(len(messages))):
            raise ValueError(f"missing source fragments for {pid}")
        result.append(dict(id=pid, conversations=[messages[i] for i in range(len(messages))]))
    return result


def convert(row, tokenizer, context, output_cap):
    history, calls = [], []
    for message in row.get("conversations", []):
        role = {"human": "user", "gpt": "assistant", "system": "system"}.get(message.get("from"))
        content = message.get("value")
        if role is None or not isinstance(content, str):
            return None
        if role == "assistant":
            if not history or history[-1]["role"] != "user":
                return None
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            reference_tokens = len(tokenizer.encode(content, add_special_tokens=False))
            output_tokens = max(1, reference_tokens)
            if output_cap:
                output_tokens = min(output_tokens, output_cap)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_tokens = len(input_ids)
            if input_tokens + output_tokens > context:
                return None  # Drop the entire conversation, never truncate its turns.
            cid = str(len(calls))
            calls.append(dict(call_id=cid, prompt=prompt, parents=[str(len(calls)-1)] if calls else [],
                              max_tokens=output_tokens, sampling={"ignore_eos": True},
                              append_parent_outputs=False, reference_output_tokens=reference_tokens,
                              input_tokens=input_tokens, input_ids=input_ids))
        history.append(dict(role=role, content=content))
    if not calls or history[-1]["role"] != "assistant":
        return None
    return dict(program_id=str(row["id"]), dataset="sharegpt", calls=calls)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--programs", type=int, default=12)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--output-cap", type=int, default=0, help="0 preserves reference answer lengths")
    p.add_argument("--min-calls", type=int, default=1)
    p.add_argument("--max-calls", type=int, default=80, help="0 allows any number of turns")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-records", action="store_true", help="replay split records individually (legacy protocol)")
    args = p.parse_args()
    if args.programs < 1 or args.context < 2 or args.output_cap < 0:
        p.error("programs must be positive, output-cap nonnegative and context at least 2")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = Path(args.source).read_bytes()
    rows = json.loads(source)
    source_records = len(rows)
    if not args.source_records:
        rows = reconstruct_conversations(rows)
    source_calls = [sum(m.get("from") == "gpt" for m in r["conversations"]) for r in rows]
    random.Random(args.seed).shuffle(rows)
    selected, seen, examined = [], set(), 0
    for row in rows:
        examined += 1
        item = convert(row, tokenizer, args.context, args.output_cap)
        if (item and len(item["calls"]) >= args.min_calls
                and (not args.max_calls or len(item["calls"]) <= args.max_calls)
                and item["program_id"] not in seen):
            selected.append(item); seen.add(item["program_id"])
            if len(selected) == args.programs:
                break
    if len(selected) != args.programs:
        raise RuntimeError("not enough eligible whole conversations")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = dict(source_url="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
                    source_sha256=hashlib.sha256(source).hexdigest(), source_records=source_records,
                    reconstructed_conversations=not args.source_records, source_programs=len(rows),
                    source_mean_calls=sum(source_calls)/len(source_calls), source_max_calls=max(source_calls),
                    examined=examined, selected_ids=sorted(seen), seed=args.seed,
                    whole_source_records=True, calls_per_program_filter=[args.min_calls, args.max_calls or None], context=args.context,
                    output_cap=args.output_cap, model=args.model,
                    input_encoding="chat template once, pretokenized without added special tokens",
                    replay_protocol=["source fragments merged by message offset" if not args.source_records else "individual source fragments",
                                 "whole-conversation context filtering", "reference-history replay",
                                 "reference output lengths with ignore_eos" if not args.output_cap else "capped output length with ignore_eos"],
                    programs=len(selected), calls=sum(len(x["calls"]) for x in selected),
                    output_tokens=sum(c["max_tokens"] for x in selected for c in x["calls"]),
                    replay_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
