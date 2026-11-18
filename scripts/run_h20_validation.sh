#!/usr/bin/env bash
set -euo pipefail
device="${1:?Provide a physical H20 index}"
model="${2:?Provide an existing small instruct model directory}"
output="${3:?Provide a directory for validation results}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AUTELLIX_PYTHON:-python}"
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES="$device" AUTELLIX_TEST_DEVICE="$device"
export AUTELLIX_GPU_BACKEND=sglang AUTELLIX_TEST_MODEL="$model" AUTELLIX_TEST_CUDA_SWAP=1
"$python_bin" -m unittest discover -s tests -v > "$output/unit-tests.log" 2>&1
"$python_bin" scripts/check_host_swap_gpu.py --output "$output/host-kv.json"
"$python_bin" scripts/check_prefill_priority_gpu.py --modern --device "$device" --model "$model" \
  --output "$output/prefill-priority.json"
"$python_bin" scripts/check_prefill_priority_gpu.py --modern --host-kv-recompute --device "$device" \
  --model "$model" --output "$output/host-kv-overflow.json"
"$python_bin" scripts/http_smoke.py --backend sglang --device "$device" --model "$model" \
  --engine-args '{"dtype":"float16","context_length":4096,"max_running_requests":4,"max_total_tokens":8192,"mem_fraction_static":0.35,"attention_backend":"triton","sampling_backend":"pytorch"}' \
  --output "$output/http.json"
