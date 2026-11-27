"""Check a >2 GiB KV round trip and measure its temporary GPU memory."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from integrations.sglang.swap import HostKV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fused", action="store_true", help="use the packed Triton transfer path")
    args = parser.parse_args()
    count = 17408
    layers = [torch.full((count, 8, 128), i, dtype=torch.float16, device="cuda") for i in range(64)]
    indices = torch.arange(count, device="cuda")
    allocator = SimpleNamespace(get_kvcache=lambda: SimpleNamespace(k_buffer=layers[:32], v_buffer=layers[32:]),
                                available_size=lambda: count, alloc=lambda n: indices[:n], free=lambda _: None)
    tree = SimpleNamespace(match_prefix=lambda _: SimpleNamespace(device_indices=indices[:0], last_device_node=None),
                           inc_lock_ref=lambda _: None, dec_lock_ref=lambda _: None,
                           insert=lambda tokens, ids: 0)
    swap = HostKV(allocator, tree, 4 * 1024**3, fused=args.fused)
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    swap.save("long", list(range(count)), indices)
    for layer in layers:
        layer.zero_()
    assert swap.restore("long")
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - initial
    # Compare in small chunks so validation itself does not allocate a large mask.
    for i, layer in enumerate(layers):
        for chunk in layer.split(1024):
            assert bool((chunk == i).all())
    assert peak_extra < 64 * 1024**2, peak_extra
    result = dict(kv_bytes=swap.used_bytes, peak_temporary_gpu_bytes=peak_extra,
                  transfer_chunk_bytes=swap.transfer_bytes, layers=len(layers), tokens=count,
                  round_trip_exact=True, fused=args.fused, device=torch.cuda.get_device_name())
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
