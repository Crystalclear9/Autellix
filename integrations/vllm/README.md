# vLLM 0.6.1 integration

`backend.py` creates a real LLMEngine. `scheduler.py` selects program-aware
cohorts inside the native scheduler while preserving native token budgets and
KV block ownership. No installed vLLM files are modified.

Use the root README to install a separate Linux/WSL environment and start
`autellix-serve --backend vllm`. The metadata adapter is retained for compatibility;
its `create_backend()` method constructs the real implementation.

Supported mode: decoder-only text generation, one GPU per replica, one sampled
sequence per request, prefix caching, CPU swap, and optional batched transfers.
Independent replicas may run on different GPUs. TP/PP and chunked prefill are
rejected. `PolicyConfig.schedule_interval` maps to native multi-step execution.
The integration handles transfer-only barriers and ensures cached multi-step
worker inputs do not replay swap-in or copy operations.

`overprovision` retains displaced GPU KV blocks up to the configured reserve
count. This is a resident reserve, not arbitrary insertion into a running cached
multi-step batch. Version checks fail closed on other vLLM releases.

Run `tests/test_gpu_runtime.py` with the environment variables in the root README.
The tests compare interrupted greedy outputs to uninterrupted outputs from the
same loaded model, verify program inheritance, and optionally exercise two real
replicas and cancellation.
