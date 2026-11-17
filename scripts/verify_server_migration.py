"""Snapshot file contents and symlinks, then verify them under a migrated root."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    args = parser.parse_args()
    root, manifest = Path(args.root), Path(args.manifest)
    if not args.verify:
        files = {}
        for path in sorted(root.rglob("*")):
            name = path.relative_to(root).as_posix()
            if (path == manifest or name.startswith("logs/migration-47541")
                    or any(name.startswith(prefix) for prefix in args.exclude_prefix)):
                continue
            if path.is_symlink():
                files[name] = dict(link=str(path.readlink()))
            elif path.is_file():
                files[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
        manifest.write_text(json.dumps(dict(source_root=str(root), files=files), indent=2))
        print(json.dumps(dict(files=len(files), bytes=sum(f.get("bytes", 0) for f in files.values()))))
    else:
        files = json.loads(manifest.read_text())["files"]
        mismatches = []
        for name, expected in files.items():
            path = root / name
            if "link" in expected:
                actual = dict(link=str(path.readlink())) if path.is_symlink() else None
            else:
                actual = dict(bytes=path.stat().st_size, sha256=digest(path)) if path.is_file() else None
            if actual != expected:
                mismatches.append(name)
        result = dict(root=str(root), checked=len(files), mismatches=mismatches,
                      manifest_sha256=digest(manifest))
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        if mismatches:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
