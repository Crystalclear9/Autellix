# SGLang 0.4.9.post6 integration

An independently implemented extension of the Autellix reproduction attempt to
SGLang. The paper describes a vLLM-based implementation. See the
[root README](../../README.md#reproduction-scope) for scope and validation limits.

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

Nonresident preempted requests now pack their computed KV into pinned CPU
storage before releasing GPU slots. Readmission restores the prefix to the
native radix cache, retaining output tokens. `autellix_swap_space` limits host
storage in GiB (default 1); unsupported non-MHA layouts fail explicitly.
Restoration locks an existing GPU prefix and allocates/transfers only its missing
suffix, so shared prefixes do not need duplicate GPU capacity. Failed allocation
or transfer releases the temporary lock and keeps the host copy for retry.
Default `--implementation paper` selects the global cohort once per N decode
steps, mixes its prefills and decodes, bypasses native prefill admission during
decode-only continuation, and disables native length-based decode retraction.
Under decode memory pressure it first releases reserves, then swaps out the
lowest-priority suffix of the frozen plan, preserving generated tokens. It does
not fail the whole batch merely because multiple active requests no longer fit.
`--implementation compat` restores the previous native scheduling hooks. Default
N=8 and one reserve are configurable project settings. Extra reserve prefills
prepare GPU prefixes ahead of completion; `refill` records mid-window replacement.
Native SGLang still updates execution metadata each iteration. Host copies are released
after successful admission or cancellation; reserve locks are released on
readmission, memory pressure, cancellation, or idle.

See the root README for installation, server commands, and real GPU tests.
