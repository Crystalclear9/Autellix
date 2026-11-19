"""Batched real KV transfers using CUDA gather/scatter and pinned staging.

The packing path uses PyTorch CUDA kernels. A single payload transfer covers
all layers and both K/V tensors. Synchronization protects staging lifetimes.
Supported layout: contiguous [2, blocks, ...] tensors (vLLM FlashAttention /
PagedAttention). Unsupported layouts fail rather than silently corrupt KV.
"""
from __future__ import annotations


def batched_swap(source, destination, mapping):
    import torch

    if len(source) != len(destination) or not source:
        raise ValueError("source and destination layer counts must match")
    pairs = torch.as_tensor(mapping, dtype=torch.long, device="cpu").reshape(-1, 2)
    if not len(pairs):
        return
    src_idx, dst_idx = pairs[:, 0].contiguous(), pairs[:, 1].contiguous()
    src_min, src_max = torch.aminmax(src_idx)
    dst_min, dst_max = torch.aminmax(dst_idx)
    src_min, src_max, dst_min, dst_max = (int(i) for i in (src_min, src_max, dst_min, dst_max))
    first_src, first_dst = source[0], destination[0]
    if {first_src.device.type, first_dst.device.type} != {"cpu", "cuda"}:
        raise ValueError("swap requires one CPU and one CUDA cache")
    for src, dst in zip(source, destination):
        if (src.ndim < 3 or dst.ndim != src.ndim or src.shape[0] != 2 or dst.shape[0] != 2
                or src.shape[2:] != dst.shape[2:] or not src.is_contiguous()
                or not dst.is_contiguous() or src.dtype != first_src.dtype
                or dst.dtype != src.dtype or src.device != first_src.device
                or dst.device != first_dst.device):
            raise ValueError("unsupported KV cache shape, dtype, device, or stride")
        if src_min < 0 or src_max >= src.shape[1] or dst_min < 0 or dst_max >= dst.shape[1]:
            raise IndexError("KV block mapping out of bounds")
    # Shared prefix blocks can repeat. Match vLLM's same-stream copy order.
    last_copy = {block: i for i, block in enumerate(dst_idx.tolist())}
    if len(last_copy) != len(dst_idx):
        keep = torch.tensor(list(last_copy.values()), dtype=torch.long, device="cpu")
        src_idx, dst_idx = src_idx[keep], dst_idx[keep]
    gpu = first_src.device if first_src.is_cuda else first_dst.device
    shape = (len(source), 2, len(src_idx), *first_src.shape[2:])
    staging = torch.empty(shape, dtype=first_src.dtype, device="cpu", pin_memory=True)
    with torch.cuda.device(gpu):
        if first_src.is_cuda:
            indices = src_idx.to(gpu)
            payload = torch.empty(shape, dtype=first_src.dtype, device=gpu)
            for layer, part in zip(source, payload.unbind()):
                torch.index_select(layer, 1, indices, out=part)
            staging.copy_(payload, non_blocking=True)
            torch.cuda.current_stream(gpu).synchronize()
            for i, layer in enumerate(destination):
                layer.index_copy_(1, dst_idx, staging[i])
        else:
            for layer, part in zip(source, staging.unbind()):
                torch.index_select(layer, 1, src_idx, out=part)
            payload = staging.to(gpu, non_blocking=True)
            target = dst_idx.to(gpu)
            for i, layer in enumerate(destination):
                layer.index_copy_(1, target, payload[i])
            torch.cuda.current_stream(gpu).synchronize()


def attach_cache_engine(cache_engine):
    from types import MethodType

    def swap_in(self, mapping):
        batched_swap(self.cpu_cache, self.gpu_cache, mapping)

    def swap_out(self, mapping):
        batched_swap(self.gpu_cache, self.cpu_cache, mapping)

    cache_engine.swap_in = MethodType(swap_in, cache_engine)
    cache_engine.swap_out = MethodType(swap_out, cache_engine)
