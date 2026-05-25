from __future__ import annotations

import argparse
import time


def simulate_copy(blocks: int, block_bytes: int, *, batched: bool) -> float:
    payload = bytearray(blocks * block_bytes)
    destination = bytearray(blocks * block_bytes)
    start = time.perf_counter()
    if batched:
        destination[:] = payload
    else:
        for idx in range(blocks):
            lo = idx * block_bytes
            hi = lo + block_bytes
            destination[lo:hi] = payload[lo:hi]
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone batched swap benchmark scaffold")
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--block-bytes", type=int, default=16 * 1024)
    args = parser.parse_args(argv)
    small = simulate_copy(args.blocks, args.block_bytes, batched=False)
    batched = simulate_copy(args.blocks, args.block_bytes, batched=True)
    print(f"small_transfers_seconds={small:.6f}")
    print(f"batched_transfer_seconds={batched:.6f}")
    print("note=this is a CPU scaffold; replace with CUDA memcpy benchmark in Linux/WSL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
