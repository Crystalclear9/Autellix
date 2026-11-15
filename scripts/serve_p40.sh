#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AUTELLIX_PYTHON:-/mnt/masfiles/env/autellix-sglang-py310/bin/python}"
model="${AUTELLIX_MODEL:-/mnt/masfiles/models/SmolLM2-135M-Instruct}"
state_dir="${AUTELLIX_STATE_DIR:-$(mktemp -d /tmp/autellix-p40-XXXXXX)}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$project_dir"
# SQLite must stay on a local filesystem, not the network-mounted persistent disk.
exec "$python_bin" -m autellix.runtime.server --backend sglang --model "$model" \
  --policy atlas --devices "${AUTELLIX_DEVICES:-0}" \
  --host 127.0.0.1 --port "${AUTELLIX_PORT:-8000}" \
  --state-dir "$state_dir" \
  --engine-args '{"autellix_legacy_gpu":true,"context_length":512,"max_running_requests":2,"max_total_tokens":2048,"mem_fraction_static":0.35,"random_seed":0}'
