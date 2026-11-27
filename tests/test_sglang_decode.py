"""Verify the regular decode fast path's metadata and native fallbacks."""

import runpy
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch


class FusedDecodeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.kernel = MagicMock()
        triton = ModuleType("triton")
        triton.jit = lambda fn: self.kernel
        triton.cdiv = lambda n, d: (n + d - 1) // d
        language = ModuleType("triton.language")
        language.constexpr = object()
        with patch.dict("sys.modules", {"triton": triton, "triton.language": language}):
            self.prepare = runpy.run_module("integrations.sglang.decode")["prepare_decode"]

    def batch(self):
        token_map = Mock()
        token_map.stride.side_effect = [128, 1]
        return NS(reqs=[object(), object()], forward_mode=NS(is_decode=lambda: True),
                  enable_overlap=False, is_hybrid=False, model_config=NS(is_encoder_decoder=False),
                  spec_algorithm=NS(is_none=lambda: True), token_to_kv_pool_allocator=NS(page_size=1),
                  sampling_info=NS(penalizer_orchestrator=NS(is_required=False)),
                  seq_lens=NS(is_cuda=True), seq_lens_sum=18, req_pool_indices=object(),
                  req_to_token_pool=NS(req_to_token=token_map), alloc_token_slots=Mock(return_value=object()),
                  input_ids=object(), output_ids=object(), out_cache_loc=None)

    def test_native_slots_store_at_previous_lengths_and_advance(self):
        batch = self.batch()
        last_output = batch.output_ids
        self.assertTrue(self.prepare(batch))
        batch.alloc_token_slots.assert_called_once_with(2)
        slots = batch.alloc_token_slots.return_value
        self.kernel.__getitem__.assert_called_once_with((1,))
        self.kernel.__getitem__.return_value.assert_called_once_with(
            batch.req_to_token_pool.req_to_token, batch.req_pool_indices, batch.seq_lens, batch.seq_lens, slots,
            2, 128, 1, 32, num_warps=1,
        )
        self.assertIs(batch.out_cache_loc, slots)
        self.assertIs(batch.input_ids, last_output)
        self.assertIsNone(batch.output_ids)
        self.assertEqual(batch.seq_lens_sum, 20)

    def test_overlap_preserves_the_inflight_batch_lengths(self):
        batch = self.batch()
        batch.enable_overlap = True
        previous = batch.seq_lens
        previous.new_empty = Mock(return_value=object())

        self.assertTrue(self.prepare(batch))
        previous.new_empty.assert_called_once_with((2,))
        self.assertIs(batch.seq_lens, previous.new_empty.return_value)
        self.kernel.__getitem__.return_value.assert_called_once_with(
            batch.req_to_token_pool.req_to_token, batch.req_pool_indices, previous, batch.seq_lens,
            batch.out_cache_loc, 2, 128, 1, 32, num_warps=1,
        )

    def test_fallback_does_not_allocate_or_mutate(self):
        for changes in (
                dict(forward_mode=NS(is_decode=lambda: False)),
                dict(is_hybrid=True),
                dict(token_to_kv_pool_allocator=NS(page_size=16)),
                dict(model_config=NS(is_encoder_decoder=True)),
                dict(sampling_info=NS(penalizer_orchestrator=NS(is_required=True)))):
            with self.subTest(changes=changes):
                batch = self.batch()
                batch.__dict__.update(changes)
                last_output = batch.output_ids
                self.assertFalse(self.prepare(batch))
                batch.alloc_token_slots.assert_not_called()
                self.assertIs(batch.output_ids, last_output)
                self.assertEqual(batch.seq_lens_sum, 18)
        self.kernel.__getitem__.assert_not_called()
