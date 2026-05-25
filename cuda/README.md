# Autellix CUDA Swap Kernel Scaffold

The paper reports a batched GPU-CPU KV swap path that gathers many small KV
blocks into a contiguous transfer. This directory contains a standalone
research scaffold for that path. It is not wired into the Python simulator or
vLLM integration yet.

## Target Environment

- Linux or WSL
- NVIDIA CUDA toolkit
- A CUDA-capable GPU

## Intended Flow

1. Build a standalone block-copy benchmark.
2. Validate that copied blocks round-trip correctly.
3. Compare many small transfers with one gathered transfer.
4. Only after standalone validation, integrate the transfer path with vLLM's KV
   cache block manager.

The current repository remains runnable without CUDA.
