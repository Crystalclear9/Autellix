"""Evaluate captured tool actions with the pinned BFCL state/response checker."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

from prepare_bfcl_replay import REVISION, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-check", action="store_true",
                        help="check official reference actions instead of captured actions")
    args = parser.parse_args()
    source, capture = Path(args.source).resolve(), Path(args.capture)
    sys.path.insert(0, str(source))
    from bfcl.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker, multi_turn_irrelevance_checker)
    tasks = {t["id"]: t for t in read_jsonl(source / "data/BFCL_v3_multi_turn_base.json")}
    answers = {t["id"]: t["ground_truth"] for t in read_jsonl(
        source / "data/possible_answer/BFCL_v3_multi_turn_base.json")}
    outcomes = []
    for program in json.loads(capture.read_text(encoding="utf-8")):
        pid = program["program_id"]
        task, reference = tasks[pid], answers[pid]
        turns = [{} for _ in task["question"]]
        for tool in program.get("reference_tools", []):
            turns[tool["turn"]].setdefault(tool["step"], []).append(tool["action"])
        actions = [[steps[step] for step in sorted(steps)] for steps in turns]
        if args.reference_check:
            actions = [[turn] for turn in reference]
        random.seed(42)
        check = multi_turn_checker(actions, reference, task, "multi_turn_base", "autellix_capture_check")
        if check["valid"]:
            check = multi_turn_irrelevance_checker(actions, reference)
        outcomes.append(dict(id=pid, **check))
    passed = sum(row["valid"] for row in outcomes)
    checker = source / "bfcl/eval_checker/multi_turn_eval/multi_turn_checker.py"
    result = dict(revision=REVISION, capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                  checker_sha256=hashlib.sha256(checker.read_bytes()).hexdigest(),
                  mode="official-reference-self-check" if args.reference_check else "captured-model-actions",
                  programs=len(outcomes), passed=passed, accuracy=passed/len(outcomes),
                  failures_by_type=dict(Counter(row["error_type"] for row in outcomes if not row["valid"])),
                  outcomes=outcomes)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=repr),
                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "outcomes"}))


if __name__ == "__main__":
    main()
