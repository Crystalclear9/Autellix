#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/scripts/server_env.sh"
export AUTELLIX_MODEL="${AUTELLIX_MODEL:-$AUTELLIX_HOME/models/Qwen3-14B}"
export AUTELLIX_DEVICE_GROUPS="${AUTELLIX_DEVICE_GROUPS:-0,1}"
export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-20,20}"
if [[ -z "${AUTELLIX_ENGINE_ARGS:-}" ]]; then
  export AUTELLIX_ENGINE_ARGS='{"autellix_legacy_gpu":true,"autellix_attention_score_elements":16777216,"autellix_mlp_chunk_size":512,"dtype":"float16","pp_size":2,"context_length":4096,"max_running_requests":8,"max_total_tokens":65536,"max_prefill_tokens":4096,"mem_fraction_static":0.9,"random_seed":0,"autellix_swap_space":2}'
fi
exec bash "$project_dir/scripts/serve_p40_three_gpu.sh"
