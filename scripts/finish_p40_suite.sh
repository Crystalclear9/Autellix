#!/usr/bin/env bash
# Finish the current two-card experiment queue, then validate the shared service.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/scripts/server_env.sh"
cd "$project_dir"
python_bin="${AUTELLIX_PYTHON:-$AUTELLIX_HOME/env/sglang/bin/python}"
run_dir="${1:-$AUTELLIX_HOME/runs/paper-suite}"
mkdir -p "$run_dir"
for name in bfcl lats mixed; do
  while [[ ! -s "$run_dir/$name.exit" ]]; do sleep 30; done
  [[ $(cat "$run_dir/$name.exit") == 0 ]]
done
echo 'All single-card agent pairs completed; comparing BFCL with shared two-card KV.'
"$python_bin" - "$run_dir" <<'PY'
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["AUTELLIX_HOME"])
plan = json.loads((root / "plans/bfcl-calibrated.json").read_text())
plan.update(device="0,1", rates=[.02], workload_kind="bfcl-pp2", layer_partition=[16, 16])
plan["engine_args"].update(pp_size=2, max_running_requests=8, max_total_tokens=196608,
                           autellix_attention_score_elements=2**24)
for case in plan["cases"]:
    if "autellix_swap_space" in case.get("engine_args", {}):
        case["engine_args"]["autellix_swap_space"] = 12
(Path(sys.argv[1]) / "bfcl-pp2-plan.json").write_text(json.dumps(plan, indent=2))
PY
set +e
"$python_bin" -u scripts/run_sglang_matrix.py --plan "$run_dir/bfcl-pp2-plan.json" \
  --output "$run_dir/bfcl-pp2.json" --resume > "$run_dir/bfcl-pp2.log" 2>&1
result=$?
set -e
echo "$result" > "$run_dir/bfcl-pp2.exit"
[[ "$result" == 0 ]]

echo 'BFCL two-card pair completed; validating two-stage pipeline.'
set +e
AUTELLIX_TEST_PP_MODEL="$AUTELLIX_HOME/models/SmolLM2-135M-Instruct" \
  AUTELLIX_TEST_PP_DEVICES=0,1 "$python_bin" -m unittest discover -s tests \
  -p test_pipeline_gpu.py -v > "$run_dir/pp2-regression.log" 2>&1
result=$?
set -e
echo "$result" > "$run_dir/pp2-regression.exit"
[[ "$result" == 0 ]]

echo 'Measuring KV transfers against compiled vLLM swap_blocks.'
set +e
CUDA_VISIBLE_DEVICES=0 "$AUTELLIX_HOME/env/vllm/bin/python" cuda/batched_swap_benchmark.py \
  --baseline vllm --blocks 128 --layers 32 --iterations 10 \
  --output "$run_dir/vllm-kv-transfer.json" > "$run_dir/vllm-kv-transfer.log" 2>&1
result=$?
set -e
echo "$result" > "$run_dir/vllm-kv-transfer.exit"

echo 'Starting existing Qwen3-14B weights across both P40 cards.'
nohup bash scripts/serve_p40_two_gpu.sh > "$run_dir/service-pp2.log" 2>&1 < /dev/null &
service_pid=$!
echo "$service_pid" > "$run_dir/service-pp2.pid"
export AUTELLIX_SERVICE_PID="$service_pid"
"$python_bin" - <<'PY'
import os
import time
from urllib.request import urlopen

deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    os.kill(int(os.environ["AUTELLIX_SERVICE_PID"]), 0)
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(5)
else:
    raise TimeoutError("two-card service startup exceeded 900 seconds")
PY
set +e
"$python_bin" scripts/validate_running_service.py --model "$AUTELLIX_HOME/models/Qwen3-14B" \
  --output "$run_dir/service-pp2-check.json" > "$run_dir/service-pp2-check.log" 2>&1
result=$?
set -e
echo "$result" > "$run_dir/service-pp2-check.exit"
[[ "$result" == 0 ]]
echo 'DONE P40 GPU workload pairs, pipeline regression and two-card HTTP service.'
