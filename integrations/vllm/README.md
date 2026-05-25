# Autellix vLLM Integration Scaffold

This directory is a Linux/WSL-oriented scaffold for mapping the in-repository
Autellix simulator concepts to vLLM v0.6.1. It is intentionally import-safe:
the main package does not require vLLM unless a caller explicitly imports this
integration.

## Target

- vLLM v0.6.1
- Linux or WSL with CUDA
- One or more CUDA-capable GPUs

## Patch Surface

- Annotate incoming requests with `program_id`, `thread_id`, parent call IDs,
  and framework metadata.
- Maintain an Autellix process table beside vLLM's scheduler state.
- Add PLAS/ATLAS queue assignment and anti-starvation promotion at scheduler
  admission/demotion points.
- Route long requests by program locality and short requests by least-used
  engine when using a multi-engine coordinator.

## Status

This scaffold does not patch vLLM yet. It provides adapter shape and patch
notes so the simulator's public API can stay stable while a real vLLM backend
is developed.
