"""Native SDPA for only the uncached query suffix."""


def prefix_attention(self, query, output, k_cache, v_cache, req_to_token,
                     req_pool_indices, seq_lens, extend_prefix_lens,
                     extend_seq_lens, scaling=None, enable_gqa=False, causal=False):
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    start = 0
    lengths = zip(req_pool_indices.tolist(), seq_lens.tolist(),
                  extend_prefix_lens.tolist(), extend_seq_lens.tolist())
    for slot, total, prefix, count in lengths:
        indices = req_to_token[slot, :total].long()
        q = query[start:start + count].transpose(0, 1).unsqueeze(0)
        k = k_cache[indices].transpose(0, 1).unsqueeze(0)
        v = v_cache[indices].transpose(0, 1).unsqueeze(0)
        mask = None
        if causal and prefix and count > 1:
            mask = (torch.arange(total, device=query.device)[None, :]
                    <= prefix + torch.arange(count, device=query.device)[:, None])
        values = scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scaling, enable_gqa=enable_gqa,
            is_causal=causal and prefix == 0)
        output[start:start + count] = values.squeeze(0).transpose(0, 1)
        start += count
    return output
