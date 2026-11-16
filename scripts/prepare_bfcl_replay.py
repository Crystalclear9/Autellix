"""Replay BFCL v3 reference tool trajectories through its official simulators."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REVISION = "8bf675cc5f92e99971ba1f6bbb15bcf1d1d2eb1c"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def convert(task, answer, docs, execute, tokenizer, context):
    functions = [function for name in task["involved_classes"] for function in docs[name]]
    history = [{"role": "system", "content":
                "You are a function calling assistant. Execute the user's task using the available "
                "functions. Respond with one Python function call at a time.\nFunctions:\n" +
                json.dumps(functions, ensure_ascii=False)}]
    calls, tool_results = [], []
    for query, actions in zip(task["question"], answer["ground_truth"], strict=True):
        history.extend(query)
        for action in actions:
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            output_length = len(tokenizer.encode(action, add_special_tokens=False))
            if len(input_ids) + output_length > context:
                return None
            started = time.monotonic()
            result, _ = execute([action], task["initial_config"], task["involved_classes"],
                                "autellix_reference", task["id"], is_evaL_run=False)
            tool_seconds = time.monotonic() - started
            cid = str(len(calls))
            calls.append(dict(call_id=cid, prompt=prompt, input_ids=input_ids,
                              input_tokens=len(input_ids), max_tokens=max(1, output_length),
                              reference_output_tokens=output_length,
                              parents=[str(len(calls)-1)] if calls else [],
                              think_time=tool_results[-1]["elapsed_seconds"] if tool_results else 0,
                              append_parent_outputs=False, sampling={"ignore_eos": True}))
            tool_results.append(dict(action=action, result=result[0], elapsed_seconds=tool_seconds))
            history.extend([{"role": "assistant", "content": action},
                            {"role": "tool", "content": result[0]}])
    if not calls:
        return None
    return dict(program_id=task["id"], dataset="bfcl", calls=calls, reference_tools=tool_results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="checkout root containing bfcl and data")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--programs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context", type=int, default=32768)
    args = p.parse_args()
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    from bfcl.eval_checker.multi_turn_eval.multi_turn_utils import (
        CLASS_FILE_PATH_MAPPING, execute_multi_turn_func_call)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    data = source / "data"
    tasks = read_jsonl(data / "BFCL_v3_multi_turn_base.json")
    answers = {x["id"]: x for x in read_jsonl(data / "possible_answer/BFCL_v3_multi_turn_base.json")}
    docs = {name: read_jsonl(data / "multi_turn_func_doc" / (module.rsplit(".", 1)[1] + ".json"))
            for name, module in CLASS_FILE_PATH_MAPPING.items()}
    random.seed(args.seed)
    random.shuffle(tasks)
    programs = []
    for task in tasks:
        item = convert(task, answers[task["id"]], docs, execute_multi_turn_func_call, tokenizer, args.context)
        if item:
            programs.append(item)
        if len(programs) == args.programs:
            break
    if len(programs) != args.programs:
        raise ValueError("insufficient complete BFCL programs within context")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(programs, ensure_ascii=False, indent=2))
    files = {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in source.rglob("*") if f.is_file() and f.suffix in {".json", ".py"}}
    manifest = dict(source="https://github.com/ShishirPatil/gorilla", revision=REVISION,
                    category="multi_turn_base", seed=args.seed, model=args.model, context=args.context,
                    programs=len(programs), calls=sum(len(p["calls"]) for p in programs),
                    selected_ids=[p["program_id"] for p in programs],
                    output_tokens=sum(c["max_tokens"] for p in programs for c in p["calls"]),
                    source_files=files, replay_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    protocol="One reference function per LLM call; official simulated tool results; fixed reference history and output length; no invented assistant acknowledgements.")
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"source_files", "selected_ids"}}, indent=2))


if __name__ == "__main__":
    main()
