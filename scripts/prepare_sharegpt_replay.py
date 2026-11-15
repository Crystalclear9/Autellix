"""Replay all turns of selected ShareGPT source records (not quality eval)."""
import argparse
import hashlib
import json
import random
from pathlib import Path


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
            output_tokens = min(max(1, reference_tokens), output_cap)
            if len(tokenizer.encode(prompt)) + output_tokens > context:
                return None  # Drop the entire conversation, never truncate its turns.
            cid = str(len(calls))
            calls.append(dict(call_id=cid, prompt=prompt, parents=[str(len(calls)-1)] if calls else [],
                              max_tokens=output_tokens, sampling={"ignore_eos": True},
                              append_parent_outputs=False, reference_output_tokens=reference_tokens))
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
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--output-cap", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.programs < 1 or args.context < 2 or args.output_cap < 1:
        p.error("programs/output-cap must be positive and context must be at least 2")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = Path(args.source).read_bytes()
    rows = json.loads(source)
    random.Random(args.seed).shuffle(rows)
    selected, seen, examined = [], set(), 0
    for row in rows:
        examined += 1
        item = convert(row, tokenizer, args.context, args.output_cap)
        if item and 2 <= len(item["calls"]) <= 8 and item["program_id"] not in seen:
            selected.append(item); seen.add(item["program_id"])
            if len(selected) == args.programs:
                break
    if len(selected) != args.programs:
        raise RuntimeError("not enough eligible whole conversations")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = dict(source_url="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
                    source_sha256=hashlib.sha256(source).hexdigest(), source_records=len(rows),
                    examined=examined, selected_ids=sorted(seen), seed=args.seed,
                    whole_source_records=True, calls_per_program_filter=[2, 8], context=args.context,
                    output_cap=args.output_cap, model=args.model,
                    limitations=["source dataset is already split; original unsplit conversations are not reconstructed",
                                 "filtered subset, not paper trace", "reference-history replay, not live agents",
                                 "capped output length, ignore_eos; not model quality evaluation"],
                    programs=len(selected), calls=sum(len(x["calls"]) for x in selected),
                    output_tokens=sum(c["max_tokens"] for x in selected for c in x["calls"]),
                    replay_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
