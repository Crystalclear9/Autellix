# vLLM 0.6.1 integration

Part of the unofficial Autellix method reproduction attempt. This adapter was
implemented independently; it is not the paper authors' released code. See the
[root README](../../README.md#reproduction-scope) for scope and validation limits.

`backend.py` creates a real LLMEngine. `scheduler.py` selects program-aware
cohorts inside the native scheduler while preserving native token budgets and
KV block ownership. No installed vLLM files are modified.

Use the root README to install a separate Linux/WSL environment and start
`autellix-serve --backend vllm`. The metadata adapter is retained for compatibility;
its `create_backend()` method constructs the real implementation.

Supported mode: decoder-only text generation, one sampled
sequence per request, prefix caching, CPU swap, and optional batched transfers.
Independent replicas may run on different GPUs. Tensor-parallel replicas use a
custom multiprocessing executor to install transfer hooks on every rank; the
two-GPU test is opt-in and requires hardware not present on the development host.
PP and chunked prefill in Autellix policy mode are rejected.
Without reserves on one GPU, `schedule_interval` maps to native multi-step execution.
The integration handles transfer-only barriers and ensures cached multi-step
worker inputs do not replay swap-in or copy operations.

With reserves (or tensor parallelism), a policy window freezes the cohort for N
decode steps while native per-step block/tensor updates remain enabled. Reserve
prefill or swap-in prepares GPU KV ahead of replacement; an early completion
triggers a `refill` before the next policy boundary. Memory pressure drops the
reserve for that window. Version checks fail closed on other vLLM releases.

Benchmark modes `vllm`, `vllm-opt`, and `vllm-opt-multistep` observe native scheduling
without replacing it. The last two separate chunked-prefill and multi-step
optimizations because v0.6.1 rejects enabling both together.

Run `tests/test_gpu_runtime.py` with the environment variables in the root README.
The tests compare interrupted greedy outputs to uninterrupted outputs from the
same loaded model, verify program inheritance, and optionally exercise two real
replicas and cancellation.
