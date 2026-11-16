"""Fetch pinned BFCL v3/LATS sources and HotpotQA training/validation data."""
import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

BFCL_REV = "8bf675cc5f92e99971ba1f6bbb15bcf1d1d2eb1c"
LATS_REV = "853d81614607dd27433faf17c7b0a7d660f95d22"
HOTPOT_REV = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
HOTPOT_TRAIN_SHA = ["76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
                    "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    root = Path(args.output)
    classes = ["gorilla_file_system", "math_api", "message_api", "posting_api", "ticket_api",
               "trading_bot", "travel_booking", "vehicle_control"]
    paths = ["data/BFCL_v3_multi_turn_base.json",
             "data/possible_answer/BFCL_v3_multi_turn_base.json"]
    paths += [f"data/multi_turn_func_doc/{name}.json" for name in classes]
    paths += [f"bfcl/eval_checker/multi_turn_eval/func_source_code/{name}.py"
              for name in classes + ["long_context", "__init__"]]
    paths += ["bfcl/eval_checker/multi_turn_eval/" + name + ".py"
              for name in ["__init__", "multi_turn_utils", "multi_turn_checker"]]
    paths += ["bfcl/model_handler/local_inference/llama_fc.py", "bfcl/model_handler/utils.py",
              "bfcl/model_handler/constant.py", "bfcl/model_handler/local_inference/base_oss_handler.py",
              "bfcl/model_handler/base_handler.py", "bfcl/_llm_response_generation.py"]
    downloads = [(f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_REV}/"
                  f"berkeley-function-call-leaderboard/{path}", root / "bfcl-v3-source" / path)
                 for path in paths]
    downloads += [(f"https://raw.githubusercontent.com/andyz245/LanguageAgentTreeSearch/{LATS_REV}/hotpot/{name}.py",
                   root / "lats-source" / (name + ".py"))
                  for name in ["base", "hotpot", "hotpotqa", "lats"]]
    downloads += [(f"https://raw.githubusercontent.com/{repo}/{revision}/LICENSE", root / folder / "LICENSE")
                  for repo, revision, folder in [("ShishirPatil/gorilla", BFCL_REV, "bfcl-v3-source"),
                                                  ("andyz245/LanguageAgentTreeSearch", LATS_REV, "lats-source")]]
    downloads.append((f"{args.hf_endpoint}/datasets/hotpotqa/hotpot_qa/resolve/{HOTPOT_REV}/"
                      "distractor/validation-00000-of-00001.parquet", root / "hotpot-validation.parquet"))
    downloads += [(f"{args.hf_endpoint}/datasets/hotpotqa/hotpot_qa/resolve/{HOTPOT_REV}/"
                   f"distractor/train-{index:05d}-of-00002.parquet", root / "hotpot-train" / f"train-{index:05d}.parquet")
                  for index in range(2)]
    def fetch(item):
        url, path = item
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            path.write_bytes(content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".parquet":
            expected = HOTPOT_TRAIN_SHA[int(path.stem.split("-")[-1])] if path.parent.name == "hotpot-train" else HOTPOT_SHA
            if digest != expected:
                raise ValueError("HotpotQA parquet hash mismatch")
        return dict(url=url, path=str(path.relative_to(root)), sha256=digest, bytes=path.stat().st_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        manifest = list(pool.map(fetch, downloads))
    (root / "source-downloads.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified {len(manifest)} source files in {root}")


if __name__ == "__main__":
    main()
