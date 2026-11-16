"""Move this project's persistent assets into one server workspace."""
import argparse
import json
import shutil
from pathlib import Path


def organize(root, include_legacy_tmp=False):
    root = root.resolve()
    target = root / "autellix"
    previous = json.loads((target / "layout-migration.json").read_text()) if (target / "layout-migration.json").exists() else {}
    moves = {
        "code/autellix-20261115": "code",
        "runs/autellix": "runs",
        "datasets/autellix-paper-20261115": "datasets/paper",
        "env/autellix-sglang-py310": "env/sglang",
        "env/autellix-vllm-py310": "env/vllm",
        "cache/autellix-current-20261115.tar.gz": "archives/autellix-current-20261115.tar.gz",
        "cache/autellix-audit-results.tar.gz": "archives/autellix-audit-results.tar.gz",
    }
    models = ["Meta-Llama-3.1-8B-Instruct", "SmolLM2-135M-Instruct", "Qwen3-14B", "Qwen3-32B"]
    moves.update({f"models/{name}": f"models/{name}" for name in models})
    # Qwen weights also serve other projects; retain their existing access paths.
    shared = {"models/Qwen3-14B", "models/Qwen3-32B"}
    for old, new in moves.items():
        src, dst = root / old, target / new
        if src.is_symlink() or not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        if old in shared:
            src.symlink_to(dst, target_is_directory=True)
    replacements = {str(root / old): str(target / new) for old, new in moves.items()}
    changed = list(previous.get("relocated_environment_files", []))
    for env in (target / "env").iterdir():
        candidates = list((env / "bin").glob("*")) + [env / "pyvenv.cfg"]
        candidates += list(env.glob("lib/python*/site-packages/*autellix*/*.json"))
        candidates += list(env.glob("lib/python*/site-packages/*autellix*.py"))
        for path in candidates:
            if path.is_symlink():
                before = str(path.readlink())
                after = before
                for old, new in replacements.items():
                    after = after.replace(old, new)
                if after != before:
                    replacement = path.with_name(path.name + ".relocated")
                    replacement.symlink_to(after)
                    replacement.replace(path)
                    changed.append(str(path.relative_to(target)))
                continue
            if not path.is_file():
                continue
            try:
                before = path.read_text()
            except UnicodeDecodeError:
                continue
            after = before
            for old, new in replacements.items():
                after = after.replace(old, new)
            if after != before:
                path.write_text(after)
                changed.append(str(path.relative_to(target)))
    for name in ["cache", "logs", "plans", "archives"]:
        (target / name).mkdir(exist_ok=True)
    # SQLite WAL needs local locking; keep a single workspace entry point while
    # placing live IPC/database files on the machine's local filesystem.
    runtime = Path("/tmp/autellix-runtime")
    runtime.mkdir(exist_ok=True)
    if not (target / "tmp").exists():
        (target / "tmp").symlink_to(runtime, target_is_directory=True)
    manifest = dict(root=str(target), moves=replacements, shared_model_aliases=sorted(shared),
                    relocated_environment_files=sorted(set(changed)), runtime=str(runtime),
                    legacy_tmp=previous.get("legacy_tmp", {}),
                    historical_results="Original JSON/log paths retained as recorded; resolve with moves mapping.")
    if include_legacy_tmp:
        legacy = target / "archives/legacy-tmp"
        legacy.mkdir(exist_ok=True)
        for path in sorted(Path("/tmp").glob("autellix*")):
            if path == runtime:
                continue
            destination = legacy / path.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(path), str(destination))
            manifest["legacy_tmp"][str(path)] = str(destination)
    (target / "layout-migration.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/masfiles")
    parser.add_argument("--include-legacy-tmp", action="store_true")
    args = parser.parse_args()
    organize(Path(args.root), args.include_legacy_tmp)
