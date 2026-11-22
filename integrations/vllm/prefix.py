"""Computed-prefix accounting for the synchronous vLLM 0.6.1 scheduler."""
from vllm.core.block.prefix_caching_block import PrefixCachingBlock
from vllm.core.interfaces import AllocStatus
from vllm.sequence import SequenceStatus
from vllm.utils import Device


class ChunkedPrefixCache:
    def __init__(self, manager):
        self.manager = manager
        self.block_size = manager.block_size
        allocator = manager.block_allocator._allocators[Device.GPU]
        self.allocator = allocator
        free_block = allocator._free_block_id
        swap_in = manager.swap_in
        can_allocate = manager.can_allocate

        def release(block):
            block_id = block.block_id
            # An unfinished full block must not enter the evictor as valid KV.
            if block_id in allocator._touched_blocks and allocator._refcounter.get(block_id) == 1:
                allocator._touched_blocks.remove(block_id)
                del allocator._cached_blocks[block.content_hash]
                allocator._decr_refcount_hashless_block(block)
            else:
                free_block(block)

        def contiguous(previous, block_ids, skip_last_block_id=True):
            end = len(block_ids) - int(skip_last_block_id)
            for i in range(len(previous), end):
                if not allocator.block_is_computed(block_ids[i]):
                    break
                previous.append(block_ids[i])
            return previous

        def restore(group):
            restored = swap_in(group)
            tracker = manager._computed_blocks_tracker
            for seq in group.get_seqs(status=SequenceStatus.SWAPPED):
                tracker.remove_seq(seq.seq_id)
                tracker.add_seq(seq.seq_id)
                self._mark(seq, 0, seq.data.get_num_computed_tokens())
            return restored

        def allocate_status(group):
            status = can_allocate(group)
            if status != AllocStatus.LATER:
                return status
            seq = group.get_seqs(status=SequenceStatus.WAITING)[0]
            required = (seq.data.get_len() + self.block_size - 1) // self.block_size
            shared = sum(allocator._block_tracker[block_id].active for block_id in self._waiting_blocks(seq))
            # Unreferenced cache entries already count as free in the allocator.
            # Reusing an active block consumes no additional physical page.
            if required - shared + manager.watermark_blocks <= manager.get_num_free_gpu_blocks():
                return AllocStatus.OK
            return status

        allocator._free_block_id = release
        allocator.get_computed_block_ids = contiguous
        manager.mark_blocks_as_computed = self.mark
        manager.swap_in = restore
        manager.can_allocate = allocate_status

    def _mark(self, seq, previous_tokens, total_tokens):
        start, end = previous_tokens // self.block_size, total_tokens // self.block_size
        if end <= start:
            return
        allocator = self.allocator
        for block_id in self.manager.get_block_table(seq)[start:end]:
            if block_id in allocator._touched_blocks:
                allocator._block_tracker[block_id].computed = True
                allocator._touched_blocks.remove(block_id)

    def mark(self, group, chunk_size):
        # vLLM calls this after building the batch metadata. Its synchronous
        # worker completes the forward before another batch can reuse these pages.
        for seq in group.get_seqs(status=SequenceStatus.RUNNING):
            computed = seq.data.get_num_computed_tokens()
            self._mark(seq, computed, computed + chunk_size)

    def reusable_tokens(self, seq):
        computed = seq.data.get_num_computed_tokens()
        block_size = self.block_size
        if seq.status == SequenceStatus.WAITING:
            prefix = sum(1 for _ in self._waiting_blocks(seq)) * block_size
        elif seq.status == SequenceStatus.RUNNING:
            prefix = len(self.manager.get_common_computed_block_ids([seq])) * block_size
        else:
            return 0
        # The last prompt block still supplies the logits for the first token.
        last_full_prefix = ((seq.data.get_len() - 1) // block_size) * block_size
        return max(0, min(prefix, last_full_prefix) - computed)

    def _waiting_blocks(self, seq):
        tokens = seq.get_token_ids()
        block_size = self.block_size
        previous = None
        for start in range(0, len(tokens) - block_size + 1, block_size):
            previous = PrefixCachingBlock.hash_block_tokens(
                start == 0, previous, tokens[start:start + block_size])
            block_id = self.allocator._cached_blocks.get(previous)
            if block_id is None or not self.allocator.block_is_computed(block_id):
                break
            yield block_id
