"""Capture local-model BFCL multi-turn agents executing the official tools."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

from prepare_bfcl_replay import read_jsonl, REVISION


def official_formatter(source):
    """Load the pinned formatter without initializing its vLLM server wrapper."""
    env = {"json": json}
    files = [(source / "bfcl/model_handler/local_inference/llama_fc.py", {"_format_prompt"}),
             (source / "bfcl/model_handler/utils.py", {"_get_language_specific_hint", "func_doc_language_specific_pre_processing"})]
    for path, names in files:
        tree = ast.parse(path.read_text())
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        for function in functions:
            function.decorator_list = []
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), env)
    return env["_format_prompt"], env["func_doc_language_specific_pre_processing"]


def parse_actions(text, allowed):
    text = text.replace("<|python_tag|>", "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            if text.startswith("{") and ";" in text:
                payload, remaining, decoder = [], text, json.JSONDecoder()
                while remaining:
                    item, end = decoder.raw_decode(remaining)
                    payload.append(item)
                    remaining = remaining[end:].strip()
                    if remaining and not remaining.startswith(";"):
                        raise ValueError("expected a separator between tool calls")
                    remaining = remaining.removeprefix(";").strip()
            else:
                payload = ast.literal_eval(text)
        if isinstance(payload, dict):
            payload = [payload]
        actions = []
        for item in payload:
            name, parameters = item["name"], item["parameters"]
            expression = name + "(" + ", ".join(f"{k}={v!r}" for k, v in parameters.items()) + ")"
            action = parse_action(expression, allowed)
            if action is None:
                return []
            actions.append(action)
        return actions
    except (ValueError, SyntaxError, TypeError, KeyError, AttributeError):
        return []


def parse_action(text, allowed):
    """Decode a function invocation into the simulator's literal-call format."""
    text = text.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, list) and len(payload) == 1:
            payload = payload[0]
        if isinstance(payload, dict) and "name" in payload:
            args = payload.get("parameters", payload.get("arguments", {}))
            text = payload["name"] + "(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
    except json.JSONDecodeError:
        pass
    # A model can add prose after a function. Parse the first syntactically
    # complete invocation, accepting only documented functions and literals.
    match = re.search(r"\b([a-zA-Z_]\w*)\s*\(", text)
    if not match or match.group(1) not in allowed:
        return None
    for end in range(match.end(), len(text) + 1):
        if text[end-1] != ")":
            continue
        expression = text[match.start():end]
        try:
            node = ast.parse(expression, mode="eval").body
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            for arg in node.args:
                ast.literal_eval(arg)
            for arg in node.keywords:
                if arg.arg is None:
                    raise ValueError("keyword expansion")
                ast.literal_eval(arg.value)
            return ast.unparse(node)
        except (SyntaxError, ValueError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="2")
    parser.add_argument("--programs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-turn", type=int, default=21)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source, out = Path(args.source).resolve(), Path(args.output)
    sys.path.insert(0, str(source))
    from bfcl.eval_checker.multi_turn_eval.multi_turn_utils import (
        CLASS_FILE_PATH_MAPPING, execute_multi_turn_func_call)
    from transformers import AutoTokenizer
    from autellix.runtime import InferenceEngine, ReplicaConfig, PolicyConfig
    data = source / "data"
    tasks = read_jsonl(data / "BFCL_v3_multi_turn_base.json")
    random.Random(args.seed).shuffle(tasks)
    tasks = tasks[:args.programs]
    docs = {name: read_jsonl(data / "multi_turn_func_doc" / (module.rsplit(".", 1)[1] + ".json"))
            for name, module in CLASS_FILE_PATH_MAPPING.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    format_prompt, preprocess_functions = official_formatter(source)
    options = dict(autellix_legacy_gpu=True, autellix_native_baseline=True, dtype="float16",
                   context_length=32768, max_running_requests=4, max_total_tokens=32768,
                   max_prefill_tokens=32768, mem_fraction_static=.9, random_seed=args.seed)
    programs, outcomes = [], []
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists():
        programs = json.loads(out.read_text())
        outcomes = json.loads(out.with_suffix(".manifest.json").read_text())["outcomes"]
    completed = {p["program_id"] for p in programs}
    if completed:
        print(f"RESUME programs={len(completed)}", flush=True)
    with InferenceEngine([ReplicaConfig("sglang", args.model, args.device, options)],
                         policy=PolicyConfig(policy="fcfs", overprovision=0)) as engine:
        for task in tasks:
            if task["id"] in completed:
                continue
            random.seed(args.seed)
            sid = engine.start_session()
            functions = copy.deepcopy([f for name in task["involved_classes"] for f in docs[name]])
            functions = preprocess_functions(functions, "multi_turn_base")
            allowed = {f["name"] for f in functions}
            history = []
            calls, tools, delay, status = [], [], 0., "completed"
            try:
                for turn, query in enumerate(task["question"]):
                    history.extend(query)
                    for step in range(args.steps_per_turn):
                        prompt = format_prompt(None, history, functions)
                        ids = tokenizer.encode(prompt, add_special_tokens=False)
                        max_output = min(4096, 32768-len(ids)-2)
                        if max_output < 1:
                            status = "context budget"
                            break
                        cid = str(len(calls))
                        result = engine.submit(sid, input_ids=ids, call_id=cid,
                                               sampling=dict(temperature=.001, max_tokens=max_output)).result(1800)
                        if result["completion_tokens"] == 0:
                            raise RuntimeError("BFCL collection returned no output tokens")
                        calls.append(dict(call_id=cid, prompt=prompt, input_ids=ids,
                                          input_tokens=len(ids), max_tokens=result["completion_tokens"],
                                          reference_output_tokens=result["completion_tokens"], reference_output=result["text"],
                                          parents=[str(len(calls)-1)] if calls else [], think_time=delay,
                                          append_parent_outputs=False, sampling={"ignore_eos": True}))
                        history.append(dict(role="assistant", content=result["text"]))
                        actions = parse_actions(result["text"], allowed)
                        print(f"BFCL id={task['id']} turn={turn} step={step} output_tokens={result['completion_tokens']} tools={len(actions)}", flush=True)
                        delay = 0.
                        if not actions:
                            break
                        started = time.monotonic()
                        observations, _ = execute_multi_turn_func_call(actions, task["initial_config"],
                            task["involved_classes"], "autellix_local", task["id"], is_evaL_run=False)
                        delay = time.monotonic() - started
                        for action, observation in zip(actions, observations):
                            tools.append(dict(turn=turn, step=step, action=action, result=observation, elapsed_seconds=delay/len(actions)))
                            history.append(dict(role="tool", name=action, content=observation))
                    else:
                        status = "per-turn step budget"
                    if status in {"context budget", "per-turn step budget"}:
                        break
            finally:
                engine.end_session(sid)
            programs.append(dict(program_id=task["id"], dataset="bfcl", calls=calls, reference_tools=tools))
            outcomes.append(dict(id=task["id"], status=status, calls=len(calls), tool_calls=len(tools),
                                 output_tokens=sum(c["max_tokens"] for c in calls)))
            out.write_text(json.dumps(programs, ensure_ascii=False, indent=2))
            manifest = dict(source="https://github.com/ShishirPatil/gorilla", revision=REVISION,
                            model=args.model, seed=args.seed, category="multi_turn_base", outcomes=outcomes,
                            steps_per_turn=args.steps_per_turn, max_output_tokens=4096, temperature=.001,
                            prompt_format="Pinned LlamaFCHandler._format_prompt and language-specific function preprocessing",
                            protocol="Actual Llama multi-turn/tool loop with official FC prompt and simulated tools; fixed captured inputs, lengths and tool delays in replay.",
                            replay_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
            out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
            print(json.dumps(outcomes[-1]), flush=True)


if __name__ == "__main__":
    main()
