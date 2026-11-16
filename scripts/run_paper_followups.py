"""Run the next dataset on a GPU after its ShareGPT pair has finished."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/masfiles/autellix")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--plans-dir")
    args = parser.parse_args()
    root, gpu = Path(args.root), args.gpu
    run = Path(args.run_dir) if args.run_dir else root / "runs/paper-suite"
    plans = Path(args.plans_dir) if args.plans_dir else root / "plans"
    dependency = run / f"sharegpt-gpu{gpu}.exit"
    while not dependency.exists():
        time.sleep(15)
    if dependency.read_text().strip() != "0":
        raise RuntimeError(f"ShareGPT job failed: {dependency}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCHDYNAMO_DISABLE="1",
                      OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", TMPDIR=str(root / "tmp"))
    base = json.loads((plans / f"sharegpt-gpu{gpu}.json").read_text())
    base["cases"] = sorted(base["cases"], key=lambda c: c["name"])

    def run_plan(name, workload, rates, count, policy="plas", seed=2026, extra_policies=()):
        plan = dict(base, workload=str(workload), rates=rates, programs=count, seed=seed, workload_kind=name)
        plan["cases"] = [dict(c, policy=policy if c["policy"] != "sglang-native" else c["policy"],
                              name=policy if c["policy"] != "sglang-native" else "native") for c in base["cases"]]
        plan["cases"] += [dict(name=p, policy=p, reserve=0 if p == "fcfs" else 1,
                                interval=8, quantum_scale=64, boundary_scale=1024) for p in extra_policies]
        path = plans / (name + ".json")
        path.write_text(json.dumps(plan, indent=2))
        with (run / (name + ".log")).open("w") as log:
            result = subprocess.run([sys.executable, "scripts/run_sglang_matrix.py", "--plan", str(path),
                                     "--output", str(run / (name + ".json"))], stdout=log, stderr=subprocess.STDOUT)
        (run / (name + ".exit")).write_text(str(result.returncode))
        result.check_returncode()

    if gpu == 0:
        base["cases"].reverse()
        run_plan("sharegpt-repeat", Path(base["workload"]), [.008], 24, seed=2027)
        run_plan("sharegpt-ablation", Path(base["workload"]), [.016], 8,
                 seed=2027, extra_policies=("fcfs", "mlfq"))
        base["engine_args"] = dict(base["engine_args"], max_running_requests=8)
        run_plan("sharegpt-batch8", Path(base["workload"]), [.016], 8, seed=2027)
    elif gpu == 2:
        workload = root / "datasets/bfcl-captured16.json"
        with (run / "bfcl-collection.log").open("w") as log:
            result = subprocess.run([sys.executable, "scripts/collect_bfcl.py", "--source", str(root / "datasets/bfcl-v3-source"),
                                     "--model", base["model"], "--device", str(gpu), "--output", str(workload),
                                     "--programs", "16"], stdout=log, stderr=subprocess.STDOUT)
        (run / "bfcl-collection.exit").write_text(str(result.returncode))
        result.check_returncode()
        run_plan("bfcl", workload, [.02, .05, .1], 48)
    else:
        workload = root / "datasets/lats-captured6.json"
        with (run / "lats-collection.log").open("w") as log:
            result = subprocess.run([sys.executable, "scripts/collect_lats.py", "--source", str(root / "datasets/lats-source"),
                                     "--data", str(root / "datasets/hotpot-train"), "--model", base["model"],
                                     "--device", str(gpu), "--output", str(workload), "--programs", "6"],
                                    stdout=log, stderr=subprocess.STDOUT)
        (run / "lats-collection.exit").write_text(str(result.returncode))
        result.check_returncode()
        run_plan("lats", workload, [.004], 6, policy="atlas")
        dependency = run / "bfcl-collection.exit"
        while not dependency.exists():
            time.sleep(15)
        if dependency.read_text().strip() != "0":
            raise RuntimeError("BFCL collection failed")
        mixed = []
        for path in [Path(base["workload"]), root / "datasets/bfcl-captured16.json", workload]:
            mixed.extend(json.loads(path.read_text()))
        mixed_path = root / "datasets/mixed-paper.json"
        mixed_path.write_text(json.dumps(mixed, ensure_ascii=False))
        run_plan("mixed", mixed_path, [.008], 12, policy="atlas")


if __name__ == "__main__":
    main()
