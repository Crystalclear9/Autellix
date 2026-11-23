from __future__ import annotations

import argparse
import time
import json
import statistics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real GPU KV swap benchmark")
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--block-bytes", type=int, default=16 * 1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output")
    parser.add_argument("--baseline", choices=("vllm", "python"), default="vllm")
    parser.add_argument("--fused", action="store_true", help="also measure fused cross-layer CUDA packing")
    parser.add_argument("--packed-host", action="store_true", help="also measure block-major host KV")
    args = parser.parse_args(argv)
    if min(args.blocks, args.block_bytes, args.layers, args.iterations) <= 0 or args.block_bytes % 2:
        parser.error("counts must be positive; block-bytes must be divisible by 2")
    import torch
    from autellix.runtime.swap import MAX_TRANSFER_BYTES, batched_swap
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no simulated fallback")
    shape = (2, args.blocks, args.block_bytes // 2)
    dtype = getattr(torch, args.dtype)
    gpu = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(args.layers)]
    cpu = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in gpu]
    restored = [torch.empty_like(layer) for layer in gpu]
    mapping = [(i, args.blocks - i - 1) for i in range(args.blocks)]
    native_mapping = torch.tensor(mapping, dtype=torch.int64)
    if args.baseline == "vllm":
        from vllm import _custom_ops as ops

    def small(source, destination):
        for src, dst in zip(source, destination):
            if args.baseline == "vllm":
                for kv in range(2):
                    ops.swap_blocks(src[kv], dst[kv], native_mapping)
            else:
                for a, b in mapping:
                    for kv in range(2):
                        dst[kv, b].copy_(src[kv, a], non_blocking=True)
        torch.cuda.synchronize()

    def measure(operation):
        for layer in restored:
            layer.zero_()
        times = []
        for i in range(args.iterations + 2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            operation(gpu, cpu)
            operation(cpu, restored)
            torch.cuda.synchronize()
            if i >= 2:
                times.append(time.perf_counter() - start)
        for a, b in zip(gpu, restored):
            if not torch.equal(a, b):
                raise AssertionError("KV round-trip mismatch")
        return statistics.median(times)

    small_seconds = measure(small)
    batched_seconds = measure(lambda src, dst: batched_swap(src, dst, mapping))
    size = args.layers * 2 * args.blocks * args.block_bytes * 2
    blocks_per_payload = max(1, MAX_TRANSFER_BYTES // (2 * args.layers * args.block_bytes))
    copies_per_direction = (args.blocks + blocks_per_payload - 1) // blocks_per_payload
    result = dict(device=torch.cuda.get_device_name(), correctness="exact_roundtrip",
                  small_roundtrip_seconds=small_seconds, batched_roundtrip_seconds=batched_seconds,
                  small_roundtrip_payload_copies=4 * args.layers * args.blocks,
                  batched_roundtrip_payload_copies=2 * copies_per_direction,
                  small_GBps=size / small_seconds / 1e9, batched_GBps=size / batched_seconds / 1e9,
                  speedup=small_seconds / batched_seconds, config=vars(args))
    if args.fused:
        from autellix.runtime.packed_kv import PackedKVLayout

        packed_source = PackedKVLayout(gpu, channels=2)
        packed_target = PackedKVLayout(restored, channels=2)

        def fused(source, destination):
            packed = packed_source if source is gpu else packed_target
            batched_swap(source, destination, mapping, packed=packed)

        fused_seconds = measure(fused)
        result.update(fused_roundtrip_seconds=fused_seconds, fused_GBps=size / fused_seconds / 1e9,
                      fused_speedup_vs_native=small_seconds / fused_seconds,
                      fused_speedup_vs_torch=batched_seconds / fused_seconds)
    if args.packed_host:
        from autellix.runtime.packed_kv import PackedKVLayout
        from autellix.runtime.swap import packed_host_swap

        packer = PackedKVLayout(gpu, channels=2, token_major=True)
        unpacker = PackedKVLayout(restored, channels=2, token_major=True)
        block_major = torch.empty(packer.shape(args.blocks), dtype=dtype, device="cpu", pin_memory=True)

        def packed_transfer(source, destination):
            to_host = source is gpu
            packed_host_swap(packer if to_host else unpacker, block_major, native_mapping, to_host=to_host)

        packed_seconds = measure(packed_transfer)
        result.update(packed_host_roundtrip_seconds=packed_seconds, packed_host_GBps=size / packed_seconds / 1e9,
                      packed_host_speedup_vs_native=small_seconds / packed_seconds)
    print(json.dumps(result, indent=2))
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
