#!/usr/bin/env bash
# Run in the selected backend's Python environment with a small chat checkpoint.
set -euo pipefail
backend="${1:?usage: validate_backend.sh vllm|sglang /absolute/model/path}"
model="${2:?provide a small chat model path or Hugging Face model ID}"
case "$backend" in
    vllm) defaults='{"max_model_len":512,"max_num_seqs":2,"gpu_memory_utilization":0.35,"swap_space":0.25}' ;;
    sglang) defaults='{"context_length":512,"max_running_requests":2,"mem_fraction_static":0.35}' ;;
    *) echo 'backend must be vllm or sglang' >&2; exit 2 ;;
esac
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AUTELLIX_PYTHON:-python}"
engine_args="${AUTELLIX_ENGINE_ARGS:-$defaults}"
cd "$project_dir"
export AUTELLIX_GPU_BACKEND="$backend" AUTELLIX_TEST_MODEL="$model"
if [[ "$backend" == vllm ]]; then
    export AUTELLIX_TEST_CUDA_SWAP=1
fi
"$python_bin" -m unittest discover -s tests -v
"$python_bin" scripts/http_smoke.py --backend "$backend" --model "$model" \
    --engine-args "$engine_args" --output "outputs/validation/$backend-http.json"
"$python_bin" -m autellix.runtime.benchmark --backend "$backend" --model "$model" \
    --workload examples/real_workload.json --policies fcfs,mlfq,plas,atlas \
    --engine-args "$engine_args" --output "outputs/validation/$backend-dag-benchmark.json"
