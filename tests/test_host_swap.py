"""Host restoration ownership tests using real CPU tensors, without a model."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from integrations.sglang.swap import HostKV


@unittest.skipUnless(torch, "requires torch (available in both backend environments)")
class HostSwapTests(unittest.TestCase):
    def test_failed_replacement_preserves_existing_host_backup(self):
        swap, _ = self.make_swap()
        old = swap.saved["r"]
        swap.used_bytes = old[1].numel() * old[1].element_size()
        original_size = swap.used_bytes
        swap.limit_bytes = original_size
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            swap.save("r", list(range(9)), torch.arange(9))
        self.assertIs(swap.saved["r"], old)
        self.assertEqual(swap.used_bytes, original_size)
        with patch("torch.empty", side_effect=RuntimeError("allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                swap.save("r", list(range(8)), torch.arange(8))
        self.assertIs(swap.saved["r"], old)
        self.assertEqual(swap.used_bytes, original_size)

    def test_chunked_save_and_restore_preserve_noncontiguous_tokens(self):
        swap, state = self.make_swap()
        swap.transfer_bytes = 4  # one token per transfer
        for i, layer in enumerate(swap.layers):
            layer.copy_(torch.arange(8).reshape(8, 1) + i * 100)
        indices = torch.tensor([7, 2, 5, 0, 3, 1, 6, 4])
        expected = torch.stack([layer[indices].clone() for layer in swap.layers])
        swap.used_bytes = swap.saved["r"][1].numel() * swap.saved["r"][1].element_size()
        with patch("torch.cuda.current_stream", return_value=SimpleNamespace(synchronize=lambda: None)):
            swap.save("r", list(range(8)), indices)
            self.assertTrue(torch.equal(swap.saved["r"][1], expected))
            for layer in swap.layers:
                layer.zero_()
            self.assertTrue(swap.restore("r"))
        for layer, values in zip(swap.layers, expected):
            self.assertTrue(torch.equal(layer[6:], values[6:]))
        self.assertEqual(swap.used_bytes, expected.numel() * expected.element_size())

    def make_swap(self, fail_alloc=False):
        layer = torch.zeros((8, 1))
        prefix = torch.arange(6)
        state = SimpleNamespace(locks=0, allocated=[], freed=[], inserted=None)
        def lock(node):
            state.locks += 1
        def unlock(node):
            state.locks -= 1
        def alloc(count):
            state.allocated.append(count)
            return None if fail_alloc or count > 2 else torch.arange(6, 6 + count)
        def insert(tokens, indices):
            self.assertEqual(state.locks, 1)
            state.inserted = indices.tolist()
            return 6
        allocator = SimpleNamespace(get_kvcache=lambda: SimpleNamespace(k_buffer=[layer], v_buffer=[layer.clone()]),
                                    available_size=lambda: 2, alloc=alloc,
                                    free=lambda indices: state.freed.extend(indices.tolist()))
        tree = SimpleNamespace(match_prefix=lambda _: SimpleNamespace(device_indices=prefix, last_device_node="prefix"),
                               inc_lock_ref=lock, dec_lock_ref=unlock, insert=insert,
                               evict=lambda _: self.fail("the missing suffix already fits"))
        swap = HostKV(allocator, tree, 1024)
        swap.saved["r"] = (list(range(8)), torch.arange(16, dtype=torch.float).reshape(2, 8, 1))
        return swap, state

    def test_restore_only_missing_suffix_preserves_existing_prefix(self):
        swap, state = self.make_swap()
        with patch("torch.cuda.current_stream", return_value=SimpleNamespace(synchronize=lambda: None)):
            self.assertTrue(swap.restore("r"))
        self.assertEqual(state.allocated, [2])
        self.assertEqual(state.inserted, list(range(8)))
        self.assertEqual(state.freed, [])
        self.assertEqual(state.locks, 0)
        self.assertEqual(swap.layers[0][:6].sum().item(), 0)
        self.assertEqual(swap.layers[0][6:].flatten().tolist(), [6, 7])
        self.assertEqual(swap.layers[1][6:].flatten().tolist(), [14, 15])
        self.assertIn("r", swap.saved)  # retained until native admission

    def test_allocation_failure_releases_prefix_lock(self):
        swap, state = self.make_swap(fail_alloc=True)
        self.assertFalse(swap.restore("r"))
        self.assertEqual(state.locks, 0)
        self.assertIn("r", swap.saved)

    def test_failed_transfer_frees_only_new_slots_and_unlocks_prefix(self):
        swap, state = self.make_swap()
        with patch("torch.cuda.current_stream", side_effect=RuntimeError("transfer failed")):
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                swap.restore("r")
        self.assertEqual(state.freed, [6, 7])
        self.assertEqual(state.locks, 0)
        self.assertIn("r", swap.saved)
