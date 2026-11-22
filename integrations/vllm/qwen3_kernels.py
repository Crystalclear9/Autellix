"""Qwen3 head-wise RMSNorm and NeoX RoPE for packed CUDA QKV projections."""
import triton
import triton.language as tl


@triton.jit
def _qk_rms_rope(QKV, POSITIONS, CACHE, Q_WEIGHT, K_WEIGHT,
                 ROW_STRIDE: tl.constexpr, Q_HEADS: tl.constexpr, HEADS: tl.constexpr,
                 DIM: tl.constexpr, EPS: tl.constexpr, HEAD_GROUP: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    heads = tl.program_id(1) * HEAD_GROUP + tl.arange(0, HEAD_GROUP)
    cols = tl.arange(0, WIDTH)
    half = DIM // 2
    valid = (heads[:, None] < HEADS) & (cols[None, :] < half)
    offset = row * ROW_STRIDE + heads[:, None] * DIM + cols[None, :]
    x = tl.load(QKV + offset, valid, 0).to(tl.float32)
    y = tl.load(QKV + offset + half, valid, 0).to(tl.float32)
    w = tl.where(heads < Q_HEADS, Q_WEIGHT, K_WEIGHT)
    wx = tl.load(w[:, None] + cols[None, :], valid, 0).to(tl.float32)
    wy = tl.load(w[:, None] + cols[None, :] + half, valid, 0).to(tl.float32)
    dtype = QKV.dtype.element_ty
    rms = tl.rsqrt((tl.sum(x * x, 1) + tl.sum(y * y, 1)) / DIM + EPS)[:, None]
    # Preserve the intermediate rounding of the two vLLM CUDA operations.
    nx = ((x * rms).to(dtype).to(tl.float32) * wx).to(dtype).to(tl.float32)
    ny = ((y * rms).to(dtype).to(tl.float32) * wy).to(dtype).to(tl.float32)
    pos = tl.load(POSITIONS + row)
    cos = tl.load(CACHE + pos * DIM + cols, cols < half, 0).to(tl.float32)
    sin = tl.load(CACHE + pos * DIM + cols + half, cols < half, 0).to(tl.float32)
    x_cos = (nx * cos[None, :]).to(dtype).to(tl.float32)
    y_sin = (ny * sin[None, :]).to(dtype).to(tl.float32)
    y_cos = (ny * cos[None, :]).to(dtype).to(tl.float32)
    x_sin = (nx * sin[None, :]).to(dtype).to(tl.float32)
    tl.store(QKV + offset, x_cos - y_sin, valid)
    tl.store(QKV + offset + half, y_cos + x_sin, valid)


def qk_rms_rope(qkv, positions, cache, q_weight, k_weight, query_heads, kv_heads, eps):
    dim = q_weight.numel()
    heads = query_heads + kv_heads
    group = 4 if qkv.shape[0] > 32 else 1
    _qk_rms_rope[(qkv.shape[0], triton.cdiv(heads, group))](
        qkv, positions, cache, q_weight, k_weight, qkv.stride(0), query_heads, heads,
        dim, eps, group, triton.next_power_of_2(dim // 2), num_warps=4 if group > 1 else 1,
        enable_fp_fusion=False)
