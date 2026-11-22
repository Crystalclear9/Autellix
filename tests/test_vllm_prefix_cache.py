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

    def make_group(self, number, size=65):
        from vllm.sequence import Sequence, SequenceGroup

        seq = Sequence(number, {"prompt_token_ids": list(range(size))}, 16)
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
        self.manager.swap_out(group)
        first.status = SequenceStatus.SWAPPED
        self.manager.swap_in(group)
        first.status = SequenceStatus.RUNNING
        second, _ = self.make_group(2)
        self.assertEqual(self.prefix.reusable_tokens(second), 16)
        self.manager.free(first)

    def test_block_aligned_prompt_keeps_its_last_block_for_sampling(self):
        first, group = self.make_group(1)
        self.allocate(first, group)
        self.manager.mark_blocks_as_computed(group, 64)
        second, other = self.make_group(2, size=64)
        self.assertEqual(self.prefix.reusable_tokens(second), 48)
        self.allocate(second, other)
        self.assertEqual(len(self.manager.get_common_computed_block_ids([second])), 3)
