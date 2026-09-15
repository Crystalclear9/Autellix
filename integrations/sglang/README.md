# SGLang 0.4.9.post6 integration

`backend.py` starts the actual SGLang Engine. A spawn-safe scheduler process
target installs `scheduler.py` hooks inside the GPU process before serving
requests. Request IDs transport program/call identity; callers do not need to
supply a dependency graph or predict model execution time.

The adapter uses native token pools and radix cache ownership during retraction,
keeps generated tokens, and requeues requests by online PLAS/ATLAS priority.
`overprovision` retains a bounded number of computed GPU prefixes under cache
locks; locks are released on readmission, memory pressure, cancellation, or idle.

The supported mode is one GPU per replica, text generation with radix caching,
page size 1, non-overlapped execution and unchunked prefill. Multiple independent
replicas are coordinated by `autellix.runtime.InferenceEngine`. TP/PP/DP,
speculation, LoRA and HiCache are rejected in this pinned implementation.

SGLang uses cache retraction/recomputation rather than the vLLM CPU block-swap
path. `schedule_interval` changes Autellix policy refresh frequency; native
SGLang still forms its batches each iteration.

See the root README for installation, server commands, and real GPU tests.
