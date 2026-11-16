"""Preserve completed three-GPU trials, workload inputs and scheduler traces."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def relocated(path, layout):
    path = Path(path)
    if path.exists():
        return path
    mappings = {**layout.get("moves", {}), **layout.get("legacy_tmp", {})}
    for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
        if str(path) == old or str(path).startswith(old + "/"):
            return Path(new + str(path)[len(old):])
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout-manifest", help="server layout-migration.json for historical paths")
    args = parser.parse_args()
    layout = json.loads(Path(args.layout_manifest).read_text()) if args.layout_manifest else {}
    run = Path(args.run_dir).resolve()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrices = [run / "comparison-32b.json", run / "comparison-replicas.json"]
    for matrix in matrices:
        if matrix.with_suffix(".exit").read_text().strip() != "0":
            raise RuntimeError(f"trial has not completed: {matrix}")
    rows = [row for matrix in matrices for row in json.loads(matrix.read_text())]
    with tempfile.TemporaryDirectory(prefix="autellix-archive-", dir=run.parent) as temporary:
        root = Path(temporary) / "three-gpu"
        root.mkdir()
        patterns = ["comparison-*.json", "comparison-*.log", "comparison-*.exit",
                    "*plan.json", "environment.json", "hardware.csv", "*-hardware-snapshot.csv", "*-freeze.txt",
                    "summary*.json", "*source-inventory.json", "validation.json", "deployment-profile.json", "admission-test.log",
                    "cpu-tests-final.*", "pipeline-test*.log", "pipeline-test*.exit",
                    "atlas-*.json", "atlas-*.log", "native-*.json", "native-*.log",
                    "http-validation.json", "http-validation*.log", "health.json", "long-prefill-oom*.log",
                    "vllm-probe.py", "vllm-probe.log", "sharegpt-*.manifest.json"]
        for pattern in patterns:
            for source in run.glob(pattern):
                if source.is_file():
                    shutil.copy2(source, root / source.name)
        inputs = root / "inputs"
        inputs.mkdir()
        workloads = {}
        for name in sorted({row["plan"]["workload"] for row in rows}):
            source = relocated(name, layout)
            target = inputs / source.name
            shutil.copy2(source, target)
            workloads[name] = str(target.relative_to(root))
        traces = root / "traces"
        traces.mkdir()
        for name in sorted({row["state_dir"] for row in rows}):
            source = relocated(name, layout)
            shutil.copytree(source, traces / source.name)
        manifest = dict(workloads=workloads, files={})
        for path in sorted(root.rglob("*")):
            if path.is_file():
                manifest["files"][str(path.relative_to(root))] = dict(
                    bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(root, arcname="three-gpu")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{digest}  {destination.name}\n")
    print(json.dumps(dict(path=str(destination), sha256=digest, bytes=destination.stat().st_size)))


if __name__ == "__main__":
    main()
