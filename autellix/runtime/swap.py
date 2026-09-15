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
    if len(set(dst_idx.tolist())) != len(dst_idx):
        raise ValueError("destination blocks must be unique")
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
        if (src_idx.min().item() < 0 or src_idx.max().item() >= src.shape[1]
                or dst_idx.min().item() < 0 or dst_idx.max().item() >= dst.shape[1]):
            raise IndexError("KV block mapping out of bounds")
    gpu = first_src.device if first_src.is_cuda else first_dst.device
    with torch.cuda.device(gpu):
        indices = src_idx.to(first_src.device)
        packed = torch.stack([layer.index_select(1, indices) for layer in source])
        if first_src.is_cuda:
            staging = torch.empty(packed.shape, dtype=packed.dtype, device="cpu", pin_memory=True)
            staging.copy_(packed, non_blocking=True)
            torch.cuda.current_stream(gpu).synchronize()
            for i, layer in enumerate(destination):
                layer.index_copy_(1, dst_idx, staging[i])
        else:
            staging = torch.empty(packed.shape, dtype=packed.dtype, device="cpu", pin_memory=True)
            staging.copy_(packed)
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
