#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/scripts/server_env.sh"
python_bin="${AUTELLIX_PYTHON:-$AUTELLIX_HOME/env/sglang/bin/python}"
model="${AUTELLIX_MODEL:-$AUTELLIX_HOME/models/Qwen3-32B}"
state_dir="${AUTELLIX_STATE_DIR:-$(mktemp -d "$TMPDIR/serve-XXXXXX")}"
export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-20,23,21}"
engine_args="${AUTELLIX_ENGINE_ARGS:-}"
if [[ -z "$engine_args" ]]; then
  engine_args='{"autellix_legacy_gpu":true,"autellix_attention_score_elements":16777216,"dtype":"float16","pp_size":3,"context_length":4096,"max_running_requests":6,"max_total_tokens":8192,"max_prefill_tokens":4096,"mem_fraction_static":0.96,"random_seed":0,"autellix_swap_space":2}'
fi
cd "$project_dir"
exec "$python_bin" -m autellix.runtime.server --backend sglang --model "$model" \
  --device-groups "${AUTELLIX_DEVICE_GROUPS:-0,1,2}" --policy atlas \
  --quantum-scale 64 --boundary-scale 1024 --schedule-interval 8 --overprovision 1 \
  --host 127.0.0.1 --port "${AUTELLIX_PORT:-8000}" --state-dir "$state_dir" \
  --engine-args "$engine_args"
