"""Synchronous host swap for the pinned SGLang MHA token pool (page size 1)."""
class HostKV:
    transfer_bytes = 16 * 1024 * 1024

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
        old = self.saved.get(rid)
        old_size = old[1].numel() * old[1].element_size() if old else 0
        size = sum(layer[0].numel() * layer.element_size() for layer in self.layers) * len(indices)
        if self.used_bytes - old_size + size > self.limit_bytes:
            raise RuntimeError("Autellix SGLang host KV budget exhausted; increase autellix_swap_space")
        layer = self.layers[0]
        host = torch.empty((len(self.layers), len(indices), *layer.shape[1:]),
                           dtype=layer.dtype, device="cpu", pin_memory=layer.is_cuda)
        indices = indices.long()
        chunk = max(1, self.transfer_bytes // (layer[0].numel() * layer.element_size()))
        stream = torch.cuda.current_stream(layer.device)
        # Keep only one chunk on the GPU, including while its DMA is in flight.
        for target, layer in zip(host, self.layers):
            for start in range(0, len(indices), chunk):
                end = min(start + chunk, len(indices))
                values = layer.index_select(0, indices[start:end])
                target[start:end].copy_(values, non_blocking=True)
                stream.synchronize()
                del values
        self.saved[rid] = (list(tokens), host)
        self.used_bytes += size - old_size

    def restore(self, rid):
        """Restore into radix cache; retain the host copy until native admission."""
        import torch
        if rid not in self.saved:
            return False
        tokens, host = self.saved[rid]
        matched = self.tree.match_prefix(tokens)
        prefix = matched.device_indices
        prefix_len = len(prefix)
        if prefix_len == len(tokens):
            return True
        # Protect the shared prefix while evicting other entries. Allocating a
        # second copy of it can reject a request whose missing suffix fits.
        node = matched.last_device_node
        self.tree.inc_lock_ref(node)
        try:
            needed = len(tokens) - prefix_len
            if self.allocator.available_size() < needed:
                self.tree.evict(needed - self.allocator.available_size())
            indices = self.allocator.alloc(needed)
            if indices is None:
                return False
            try:
                layer = self.layers[0]
                chunk = max(1, self.transfer_bytes // (layer[0].numel() * layer.element_size()))
                stream = torch.cuda.current_stream(layer.device)
                copy_indices = indices.long()
                for layer, source in zip(self.layers, host):
                    for start in range(0, needed, chunk):
                        end = min(start + chunk, needed)
                        values = source[prefix_len + start:prefix_len + end].to(layer.device, non_blocking=True)
                        layer.index_copy_(0, copy_indices[start:end], values)
                        stream.synchronize()
                        del values
                shared = self.tree.insert(tokens, torch.cat((prefix, indices)))
            except BaseException:
                self.allocator.free(indices)
                raise
            # Only newly allocated duplicate slots belong to this operation.
            self.allocator.free(indices[:max(0, shared - prefix_len)])
            return True
        finally:
            self.tree.dec_lock_ref(node)
