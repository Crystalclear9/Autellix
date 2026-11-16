#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/scripts/server_env.sh"
python_bin="${AUTELLIX_PYTHON:-$AUTELLIX_HOME/env/sglang/bin/python}"
run_name="${1:-paper-suite-$(date +%Y%m%d-%H%M%S)}"
run_dir="$AUTELLIX_HOME/runs/$run_name"
plans_dir="$AUTELLIX_HOME/plans/$run_name"
mkdir "$run_dir"
mkdir -p "$plans_dir"
cd "$project_dir"
"$python_bin" - "$AUTELLIX_HOME" "$plans_dir" <<'PY'
import json, sys
from pathlib import Path
root, plans = map(Path, sys.argv[1:])
base = json.loads(Path('docs/results/three-replica-plan.json').read_text())
base.update(model=str(root/'models/Meta-Llama-3.1-8B-Instruct'),
            workload=str(root/'datasets/sharegpt-whole128.json'),
            programs=48, seed=2026, repeats=1, request_timeout=21600)
base['cases'] = base['cases'][:2]
base['cases'][0]['name'], base['cases'][1]['name'] = 'native', 'plas'
for gpu, rate in enumerate([.004, .008, .016]):
    plan = dict(base, device=str(gpu), rates=[rate])
    plan['cases'] = base['cases'][::-1] if gpu == 1 else base['cases']
    (plans/f'sharegpt-gpu{gpu}.json').write_text(json.dumps(plan, indent=2))
PY
for gpu in 0 1 2; do
  (
    set +e
    "$python_bin" -u scripts/run_sglang_matrix.py --plan "$plans_dir/sharegpt-gpu$gpu.json" \
      --output "$run_dir/sharegpt-gpu$gpu.json" > "$run_dir/sharegpt-gpu$gpu.log" 2>&1
    result=$?
    echo "$result" > "$run_dir/sharegpt-gpu$gpu.exit"
    if [[ "$result" == 0 ]]; then
      "$python_bin" -u scripts/run_paper_followups.py --root "$AUTELLIX_HOME" --gpu "$gpu" \
        --run-dir "$run_dir" --plans-dir "$plans_dir" > "$run_dir/followup-gpu$gpu.log" 2>&1
      result=$?
    fi
    echo "$result" > "$run_dir/gpu$gpu.exit"
    exit "$result"
  ) &
  echo "$!" > "$run_dir/gpu$gpu.pid"
done
failure=0
for gpu in 0 1 2; do
  wait "$(cat "$run_dir/gpu$gpu.pid")" || failure=1
done
exit "$failure"
