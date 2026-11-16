#!/usr/bin/env bash
# Persistent assets have one root; SQLite and IPC use local runtime storage.
export AUTELLIX_HOME="${AUTELLIX_HOME:-/mnt/masfiles/autellix}"
export AUTELLIX_RUNTIME_DIR="${AUTELLIX_RUNTIME_DIR:-/tmp/autellix-runtime}"
mkdir -p "$AUTELLIX_HOME/cache" "$AUTELLIX_HOME/logs" "$AUTELLIX_RUNTIME_DIR"
if [[ ! -e "$AUTELLIX_HOME/tmp" ]]; then
  ln -s "$AUTELLIX_RUNTIME_DIR" "$AUTELLIX_HOME/tmp"
fi
export TMPDIR="$AUTELLIX_HOME/tmp"
export XDG_CACHE_HOME="$AUTELLIX_HOME/cache"
export HF_HOME="$AUTELLIX_HOME/cache/huggingface"
export TORCH_HOME="$AUTELLIX_HOME/cache/torch"
export TRITON_CACHE_DIR="$AUTELLIX_HOME/cache/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false
