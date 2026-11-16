import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from integrations.sglang.attention import prefix_attention


@unittest.skipUnless(torch, "requires torch")
class PrefixAttentionTests(unittest.TestCase):
    def test_query_tiles_keep_absolute_causal_positions(self):
        torch.manual_seed(13)
        for prefix in (0, 5):
            for causal in (False, True):
                count, total = 17, prefix + 17
                q = torch.randn(total, 4, 8)
                k, v = torch.randn(total, 2, 8), torch.randn(total, 2, 8)
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                    enable_gqa=True, is_causal=causal)[0].transpose(0, 1)[prefix:]
                output = torch.empty_like(q[prefix:])
                with patch("integrations.sglang.attention.MAX_SCORE_ELEMENTS", 128):
                    prefix_attention(None, q[prefix:], output, k, v, torch.arange(total)[None],
                                     torch.tensor([0]), torch.tensor([total]), torch.tensor([prefix]),
                                     torch.tensor([count]), enable_gqa=True, causal=causal)
                torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)

    def test_ragged_batch_uses_each_request_cache_mapping(self):
        torch.manual_seed(11)
        specs = [(5, 1), (0, 3), (4, 2)]
        keys, values = torch.randn(32, 2, 8), torch.randn(32, 2, 8)
        mapping = torch.randperm(32)[:24].reshape(3, 8)
        query = torch.randn(6, 4, 8)
        expected, start = [], 0
        for slot, (prefix, count) in enumerate(specs):
            total = prefix + count
            q = torch.cat([torch.zeros(prefix, 4, 8), query[start:start + count]])
            result = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None], keys[mapping[slot, :total]].transpose(0, 1)[None],
                values[mapping[slot, :total]].transpose(0, 1)[None],
                is_causal=True, enable_gqa=True)
            expected.append(result[0].transpose(0, 1)[prefix:])
            start += count
        out = torch.empty_like(query)
        prefix_attention(None, query, out, keys, values, mapping, torch.arange(3),
                         torch.tensor([p + n for p, n in specs]),
                         torch.tensor([p for p, n in specs]), torch.tensor([n for p, n in specs]),
                         enable_gqa=True, causal=True)
        torch.testing.assert_close(out, torch.cat(expected), atol=1e-6, rtol=1e-5)

    def test_suffix_matches_full_causal_attention(self):
        torch.manual_seed(7)
        for kv_heads in (1, 4):
            for causal in (False, True):
                for prefix, count in ((0, 7), (5, 1), (5, 3), (64, 2)):
                    with self.subTest(kv_heads=kv_heads, causal=causal, prefix=prefix, count=count):
                        n = prefix + count
                        q = torch.randn(n, 4, 8)
                        k, v = torch.randn(n, kv_heads, 8), torch.randn(n, kv_heads, 8)
                        ref = torch.nn.functional.scaled_dot_product_attention(
                            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                            is_causal=causal, enable_gqa=kv_heads != 4, scale=.3)
                        out = torch.empty_like(q[prefix:])
                        original = torch.nn.functional.scaled_dot_product_attention
                        with patch("torch.nn.functional.scaled_dot_product_attention", wraps=original) as sdpa:
                            prefix_attention(None, q[prefix:], out, k, v, torch.arange(n)[None],
                                             torch.tensor([0]), torch.tensor([n]), torch.tensor([prefix]),
                                             torch.tensor([count]), scaling=.3, enable_gqa=kv_heads != 4, causal=causal)
                        torch.testing.assert_close(out, ref[0].transpose(0, 1)[prefix:], atol=1e-6, rtol=1e-5)
                        self.assertEqual(sdpa.call_args.args[0].shape[-2], count)
