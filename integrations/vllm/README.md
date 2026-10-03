# vLLM 0.6.1 integration

Part of the unofficial Autellix method reproduction attempt. This adapter was
implemented independently; it is not the paper authors' released code. See the
[root README](../../README.md#reproduction-scope) for scope and validation limits.

`backend.py` creates a real LLMEngine. Default `paper.py` builds globally ordered
batches without calling native `_schedule()`. Priority selection runs at window
boundaries; continuation materializes the saved plan, with resident reserve
refill. Native block ownership and attention metadata are preserved. Mixed
prefill/decode execution prevents native phase preference from overriding
priority. No installed vLLM files are modified.

Use the root README to install a separate Linux/WSL environment and start
`autellix-serve --backend vllm`. The metadata adapter is retained for compatibility;
its `create_backend()` method constructs the real implementation.

Supported mode: decoder-only text generation, one sampled
sequence per request, prefix caching, CPU swap, and optional batched transfers.
Independent replicas may run on different GPUs. Tensor-parallel replicas use a
custom multiprocessing executor to install transfer hooks on every rank; the
two-GPU test is opt-in and requires hardware not present on the development host.
PP is rejected. Paper mode enables mixed-phase attention support but performs
whole-call prefill admission within the configured token budget. A prompt that
cannot fit must use a larger budget. Future decode slots are reserved at window
preparation. Token metadata continues to advance once per execution step.

Use `--implementation compat` to select the former scheduler wrapper. In that
mode, without reserves on one GPU, `schedule_interval` maps to native cached
multi-step execution. With reserves or TP it uses native per-step selection
within the filtered cohort. Paper mode is the default (N=8, one reserve).
Version checks fail closed on other vLLM releases.

Benchmark modes `vllm`, `vllm-opt`, and `vllm-opt-multistep` observe native scheduling
without replacing it. The last two separate chunked-prefill and multi-step
optimizations because v0.6.1 rejects enabling both together.

Run `tests/test_gpu_runtime.py` with the environment variables in the root README.
The tests compare interrupted greedy outputs to uninterrupted outputs from the
same loaded model, verify program inheritance, and optionally exercise two real
replicas and cancellation.
