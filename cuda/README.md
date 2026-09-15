# GPU KV swap benchmark

The benchmark performs real GPU/CPU transfers, validates exact KV round-trip
correctness with a nontrivial block mapping, synchronizes CUDA for timing, and
reports median times after warm-up.

```bash
python cuda/batched_swap_benchmark.py --blocks 128 --layers 4 --iterations 10
```

The default baseline is vLLM's actual compiled `swap_blocks` operation. Use
`--baseline python` only when intentionally comparing against per-block Python
copy calls; those timings include Python dispatch overhead.

The batched path is `autellix/runtime/swap.py`, also used by the real vLLM
backend. It packs all layers' K/V blocks with PyTorch CUDA kernels, transfers
one contiguous payload through pinned host memory, and scatters into the
mapped destination blocks. No separate CUDA compiler is required for this
path. It supports contiguous `[2, blocks, ...]` KV layouts and rejects unsafe
shapes, mappings, and duplicate destinations.

The benchmark raises an error without CUDA; there is no CPU simulation fallback.
It does not assume the batched path is faster for every payload size or machine.
