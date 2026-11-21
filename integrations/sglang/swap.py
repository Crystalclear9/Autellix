"""Synchronous host swap for the pinned SGLang MHA token pool (page size 1)."""
from time import perf_counter


class HostKV:
    transfer_bytes = 16 * 1024 * 1024

    def __init__(self, allocator, tree_cache, limit_bytes, recompute_on_full=False, fused=False, observe=None):
        self.allocator, self.tree = allocator, tree_cache
        pool = allocator.get_kvcache()
        if not hasattr(pool, "k_buffer") or not hasattr(pool, "v_buffer"):
            raise ValueError("Autellix host swap requires an MHA KV pool")
        self.layers = list(pool.k_buffer) + list(pool.v_buffer)
        self.packed = None
        if fused:
            from autellix.runtime.packed_kv import PackedKVLayout
            self.packed = PackedKVLayout(self.layers, channels=1, token_major=True)
        self.limit_bytes = limit_bytes
        self.recompute_on_full = recompute_on_full
        self.used_bytes = 0
        self.saved = {}
        self.observe = observe
        self.transfer_seconds = 0.

    def _record_transfer(self, direction, blocks, started):
        if self.observe is not None:
            seconds = perf_counter() - started
            self.transfer_seconds += seconds
            self.observe("kv_transfer", direction=direction, blocks=blocks, seconds=seconds)

    def discard(self, rid):
        entry = self.saved.pop(rid, None)
        if entry:
            self.used_bytes -= entry[1].numel() * entry[1].element_size()

    def _move_packed(self, indices, host, *, pack):
        import torch

        layout = self.packed
        count = max(1, self.transfer_bytes // layout.bytes_per_block)
        stream = torch.cuda.current_stream(layout.device)
        for start in range(0, len(indices), count):
            end = min(start + count, len(indices))
            staging = host[start:end]
            if pack:
                payload = torch.empty_like(staging, device=layout.device)
                layout.move(indices[start:end], payload, pack=True)
                staging.copy_(payload, non_blocking=True)
            else:
                payload = staging.to(layout.device, non_blocking=True)
                layout.move(indices[start:end], payload, pack=False)
            stream.synchronize()

    def save(self, rid, tokens, indices):
        import torch
        old = self.saved.get(rid)
        old_size = old[1].numel() * old[1].element_size() if old else 0
        size = sum(layer[0].numel() * layer.element_size() for layer in self.layers) * len(indices)
        if self.used_bytes - old_size + size > self.limit_bytes:
            if self.recompute_on_full:
                return False
            raise RuntimeError("Autellix SGLang host KV budget exhausted; increase autellix_swap_space")
        layer = self.layers[0]
        started = perf_counter() if self.observe is not None else None
        shape = (self.packed.shape(len(indices)) if self.packed is not None
                 else (len(self.layers), len(indices), *layer.shape[1:]))
        host = torch.empty(shape,
                           dtype=layer.dtype, device="cpu", pin_memory=layer.is_cuda)
        indices = indices.long()
        if self.packed is not None:
            self._move_packed(indices, host, pack=True)
        else:
            chunk = max(1, self.transfer_bytes // (layer[0].numel() * layer.element_size()))
            stream = torch.cuda.current_stream(layer.device)
            for target, layer in zip(host, self.layers):
                for start in range(0, len(indices), chunk):
                    end = min(start + chunk, len(indices))
                    values = layer.index_select(0, indices[start:end])
                    target[start:end].copy_(values, non_blocking=True)
                    stream.synchronize()
                    del values
        self._record_transfer("out", len(indices), started)
        self.saved[rid] = (list(tokens), host)
        self.used_bytes += size - old_size
        return True

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
                started = perf_counter() if self.observe is not None else None
                if self.packed is not None:
                    self._move_packed(copy_indices, host[prefix_len:], pack=False)
                else:
                    for layer, source in zip(self.layers, host):
                        for start in range(0, needed, chunk):
                            end = min(start + chunk, needed)
                            values = source[prefix_len + start:prefix_len + end].to(layer.device, non_blocking=True)
                            layer.index_copy_(0, copy_indices[start:end], values)
                            stream.synchronize()
                            del values
                self._record_transfer("in", needed, started)
                shared = self.tree.insert(tokens, torch.cat((prefix, indices)))
            except BaseException:
                self.allocator.free(indices)
                raise
            # Only newly allocated duplicate slots belong to this operation.
            self.allocator.free(indices[:max(0, shared - prefix_len)])
            return True
        finally:
            self.tree.dec_lock_ref(node)
