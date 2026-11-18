#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/scripts/server_env.sh"
cd "$project_dir"
python_bin="${AUTELLIX_PYTHON:-$AUTELLIX_HOME/env/sglang/bin/python}"
run_dir="${1:-$AUTELLIX_HOME/runs/paper-suite}"
archive="${2:-$AUTELLIX_HOME/archives/p40-paper-suite.tar.gz}"
names=(sharegpt-gpu0 sharegpt-gpu1 sharegpt-gpu2 sharegpt-repeat sharegpt-ablation
       sharegpt-batch8 sharegpt-pp3 bfcl lats mixed bfcl-pp2)
inputs=()
for name in "${names[@]}" pp2-regression service-pp2-check; do
  while [[ ! -s "$run_dir/$name.exit" ]]; do sleep 30; done
  if [[ $(cat "$run_dir/$name.exit") != 0 ]]; then
    echo "Failed: $name. See $run_dir/$name.log" >&2
    exit 1
  fi
done
for name in "${names[@]}"; do
  input="$run_dir/$name.json"
  inputs+=("$input")
  "$python_bin" scripts/analyze_scheduler_trace.py "$input" \
    --output "$run_dir/$name-scheduler.json" > /dev/null
done
"$python_bin" scripts/summarize_paper_suite.py "${inputs[@]}" \
  --output "$run_dir/paper-suite-summary.json" > /dev/null
MPLBACKEND=Agg "$python_bin" scripts/plot_paper_suite.py "$run_dir/paper-suite-summary.json" \
  --output-dir "$run_dir"
"$python_bin" scripts/archive_paper_suite.py --root "$AUTELLIX_HOME" --include-bfcl-pp2 \
  --run-dir "$run_dir" --output "$archive"
