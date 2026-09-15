#!/usr/bin/env bash
set -euo pipefail
backend="${1:-vllm}"
case "$backend" in vllm|sglang) ;; *) echo 'usage: setup_backend.sh vllm|sglang' >&2; exit 2;; esac
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${AUTELLIX_ENV_DIR:-$HOME/autellix-envs/$backend}"
if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ first.' >&2
    exit 1
fi
"$uv_bin" venv "$runtime_dir" --python 3.10 --allow-existing
"$uv_bin" pip install --python "$runtime_dir/bin/python" -r "$project_dir/requirements/$backend-py310-linux.txt"
"$uv_bin" pip install --python "$runtime_dir/bin/python" -e "$project_dir" --no-deps
"$runtime_dir/bin/python" -c 'import torch; print("torch", torch.__version__, "CUDA available", torch.cuda.is_available())'
echo "Python: $runtime_dir/bin/python"
