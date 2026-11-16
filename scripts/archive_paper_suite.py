"""Archive completed paper trials, exact inputs, executable sources and traces."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/mnt/masfiles/autellix")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    server, run, output = Path(args.root), Path(args.run_dir), Path(args.output)
    names = [f"sharegpt-gpu{i}" for i in range(3)] + ["sharegpt-repeat", "sharegpt-ablation", "sharegpt-batch8", "bfcl", "lats", "mixed"]
    for name in names + ["bfcl-collection", "lats-collection"]:
        if (run / (name + ".exit")).read_text().strip() != "0":
            raise ValueError(f"unfinished trial: {name}")
    rows = [row for name in names for row in json.loads((run / (name + ".json")).read_text())]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper-archive-", dir=run.parent) as temporary:
        root = Path(temporary) / "paper-suite"
        root.mkdir()
        for source in run.iterdir():
            if source.is_file():
                shutil.copy2(source, root / source.name)
        inputs = root / "inputs"
        inputs.mkdir()
        workloads = {}
        for name in sorted({r["plan"]["workload"] for r in rows}):
            path = Path(name)
            shutil.copy2(path, inputs / path.name)
            workloads[name] = "inputs/" + path.name
            manifest = path.with_suffix(".manifest.json")
            if manifest.exists():
                shutil.copy2(manifest, inputs / manifest.name)
            search_log = path.with_suffix(".search.log")
            if search_log.exists():
                shutil.copy2(search_log, inputs / search_log.name)
        traces = root / "traces"
        traces.mkdir()
        for name in sorted({row["state_dir"] for row in rows}):
            shutil.copytree(name, traces / Path(name).name)
        for name in ["bfcl-v3-source", "lats-source"]:
            shutil.copytree(server / "datasets" / name, root / "upstream" / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(server / "datasets/source-downloads.json", root / "source-downloads.json")
        shutil.copy2(server / "layout-migration.json", root / "layout-migration.json")
        for name in ["paper-suite-networkfs-start", "paper-suite-swap-oom"]:
            failure = run.parent / name
            if failure.exists():
                shutil.copytree(failure, root / "startup-failures" / name)
        code = root / "source"
        code.mkdir()
        for folder in ["autellix", "integrations", "scripts", "tests", "requirements"]:
            shutil.copytree(server / "code" / folder, code / folder,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(server / "code/pyproject.toml", code / "pyproject.toml")
        manifest = dict(workloads=workloads, trials=len(rows), files={})
        for path in sorted(root.rglob("*")):
            if path.is_file():
                manifest["files"][str(path.relative_to(root))] = dict(
                    bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with tarfile.open(output, "w:gz") as archive:
            archive.add(root, arcname="paper-suite")
    result = dict(path=str(output), bytes=output.stat().st_size,
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(), trials=len(rows),
                  files=len(manifest["files"]))
    output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
