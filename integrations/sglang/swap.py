"""Synchronous host swap for the pinned SGLang MHA token pool (page size 1)."""
class HostKV:
    def __init__(self, allocator, tree_cache, limit_bytes):
        self.allocator, self.tree = allocator, tree_cache
        pool = allocator.get_kvcache()
        if not hasattr(pool, "k_buffer") or not hasattr(pool, "v_buffer"):
            raise ValueError("Autellix host swap requires an MHA KV pool")
        self.layers = list(pool.k_buffer) + list(pool.v_buffer)
        self.limit_bytes = limit_bytes
        self.used_bytes = 0
        self.saved = {}

    def discard(self, rid):
        entry = self.saved.pop(rid, None)
        if entry:
            self.used_bytes -= entry[1].numel() * entry[1].element_size()

    def save(self, rid, tokens, indices):
        import torch
        self.discard(rid)
        size = sum(layer[0].numel() * layer.element_size() for layer in self.layers) * len(indices)
        if self.used_bytes + size > self.limit_bytes:
            raise RuntimeError("Autellix SGLang host KV budget exhausted; increase autellix_swap_space")
        packed = torch.stack([layer.index_select(0, indices.long()) for layer in self.layers])
        host = torch.empty(packed.shape, dtype=packed.dtype, device="cpu", pin_memory=True)
        host.copy_(packed, non_blocking=True)
        torch.cuda.current_stream(packed.device).synchronize()
        self.saved[rid] = (list(tokens), host)
        self.used_bytes += size

    def restore(self, rid):
        """Restore into radix cache; retain the host copy until native admission."""
        import torch
        if rid not in self.saved:
            return False
        tokens, host = self.saved[rid]
        matched = self.tree.match_prefix(tokens)
        if len(matched.device_indices) == len(tokens):
            return True
        needed = len(tokens)
        if self.allocator.available_size() < needed:
            self.tree.evict(needed - self.allocator.available_size())
        indices = self.allocator.alloc(needed)
        if indices is None:
            return False
        try:
            payload = host.to(self.layers[0].device, non_blocking=True)
            for layer, values in zip(self.layers, payload):
                layer.index_copy_(0, indices.long(), values)
            torch.cuda.current_stream(payload.device).synchronize()
        except BaseException:
            self.allocator.free(indices)
            raise
        prefix_len = self.tree.insert(tokens, indices.clone())
        self.allocator.free(indices[:prefix_len])
        return True
