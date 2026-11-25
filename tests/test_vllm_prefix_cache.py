"""Use the pinned vLLM block allocator without a model or CUDA device."""
from importlib.util import find_spec
import unittest


@unittest.skipUnless(find_spec("vllm"), "requires vLLM 0.6.1")
class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        from vllm.core.block_manager_v2 import BlockSpaceManagerV2
        from integrations.vllm.prefix import ChunkedPrefixCache

        self.manager = BlockSpaceManagerV2(16, 32, 32, enable_caching=True)
        self.prefix = ChunkedPrefixCache(self.manager)

    def make_group(self, number, size=65, start=0):
        from vllm.sequence import Sequence, SequenceGroup

        seq = Sequence(number, {"prompt_token_ids": list(range(start, start + size))}, 16)
        return seq, SequenceGroup(str(number), [seq], 0.)

    def allocate(self, seq, group):
        from vllm.sequence import SequenceStatus

        self.manager.allocate(group)
        seq.status = SequenceStatus.RUNNING

    def test_partial_chunk_shares_only_completed_full_blocks(self):
        first, group = self.make_group(1)
        second, other = self.make_group(2)
        self.allocate(first, group)
        self.manager.mark_blocks_as_computed(group, 16)
        first.data.update_num_computed_tokens(16)
        self.assertEqual(self.prefix.reusable_tokens(second), 16)
        self.allocate(second, other)
        self.assertEqual(len(self.manager.get_common_computed_block_ids([second])), 1)
        self.manager.free(first)
        self.manager.free(second)
        third, _ = self.make_group(3)
        self.assertEqual(self.prefix.reusable_tokens(third), 16)
        self.assertEqual(len(self.prefix.allocator._cached_blocks), 1)

    def test_swap_preserves_computed_prefix_only(self):
        from vllm.sequence import SequenceStatus

        first, group = self.make_group(1)
        self.allocate(first, group)
        self.manager.mark_blocks_as_computed(group, 16)
        first.data.update_num_computed_tokens(16)
        self.assertEqual(len(self.manager.swap_out(group)), 1)
        first.status = SequenceStatus.SWAPPED
        self.assertEqual(self.manager.swap_in(group), [])
        first.status = SequenceStatus.RUNNING
        second, _ = self.make_group(2)
        self.assertEqual(self.prefix.reusable_tokens(second), 16)
        self.manager.free(first)

    def test_cold_restore_copies_initialized_pages_and_leaves_others_uncomputed(self):
        from vllm.core.block_manager_v2 import BlockSpaceManagerV2
        from vllm.sequence import SequenceStatus
        from integrations.vllm.prefix import ChunkedPrefixCache

        self.manager = BlockSpaceManagerV2(16, 5, 16, enable_caching=True)
        self.prefix = ChunkedPrefixCache(self.manager)
        first, group = self.make_group(1)
        self.allocate(first, group)
        self.manager.mark_blocks_as_computed(group, 16)
        first.data.update_num_computed_tokens(16)
        self.assertEqual(len(self.manager.swap_out(group)), 1)
        first.status = SequenceStatus.SWAPPED

        other, other_group = self.make_group(2, start=100)
        self.allocate(other, other_group)
        self.manager.free(other)
        self.assertEqual(len(self.manager.swap_in(group)), 1)
        first.status = SequenceStatus.RUNNING
        self.assertEqual(len(self.manager.get_common_computed_block_ids([first])), 1)
        self.manager.free(first)

    def test_block_aligned_prompt_keeps_its_last_block_for_sampling(self):
        first, group = self.make_group(1)
        self.allocate(first, group)
        self.manager.mark_blocks_as_computed(group, 64)
        second, other = self.make_group(2, size=64)
        self.assertEqual(self.prefix.reusable_tokens(second), 48)
        self.allocate(second, other)
        self.assertEqual(len(self.manager.get_common_computed_block_ids([second])), 3)

    def test_active_prefix_reduces_new_pages_needed_for_admission(self):
        from vllm.core.block_manager_v2 import BlockSpaceManagerV2
        from vllm.core.interfaces import AllocStatus
        from integrations.vllm.prefix import ChunkedPrefixCache

        self.manager = BlockSpaceManagerV2(16, 8, 8, enable_caching=True)
        original_can_allocate = self.manager.can_allocate
        prefix = ChunkedPrefixCache(self.manager)
        first, initial = self.make_group(1)
        second, next_group = self.make_group(2)
        self.allocate(first, initial)
        # Uncomputed blocks cannot provide reservation credit.
        self.assertEqual(self.manager.can_allocate(next_group), AllocStatus.LATER)
        self.manager.mark_blocks_as_computed(initial, 64)
        self.assertEqual(original_can_allocate(next_group), AllocStatus.LATER)
        self.assertEqual(self.manager.can_allocate(next_group), AllocStatus.OK)
        free_before = self.manager.get_num_free_gpu_blocks()
        self.allocate(second, next_group)
        self.assertEqual(free_before - self.manager.get_num_free_gpu_blocks(), 1)
        self.assertEqual(prefix.reusable_tokens(second), 64)

    def test_decode_can_fill_its_private_page_when_no_free_page_remains(self):
        from vllm.core.block_manager_v2 import BlockSpaceManagerV2
        from integrations.vllm.prefix import ChunkedPrefixCache

        self.manager = BlockSpaceManagerV2(16, 2, 4, enable_caching=True)
        prefix = ChunkedPrefixCache(self.manager)
        seq, group = self.make_group(1, size=17)
        self.allocate(seq, group)
        self.assertEqual(self.manager.get_num_free_gpu_blocks(), 0)
        seq.data.append_token_id(100, 0.)
        self.assertFalse(self.manager.can_append_slots(group, 1))
        self.assertTrue(prefix.can_append_slots(group, 1))
        self.assertFalse(prefix.can_append_slots(group, 16))
        self.assertEqual(self.manager.append_slots(seq, 1), [])
        self.assertEqual(self.manager.get_num_free_gpu_blocks(), 0)

        child, child_group = self.make_group(2, size=18)
        from vllm.sequence import SequenceStatus
        child.status = SequenceStatus.RUNNING
        self.manager.fork(seq, child)
        child.data.append_token_id(101, 0.)
        self.assertFalse(prefix.can_append_slots(child_group, 0))
