"""Fuse the pinned SGLang executor's regular decode metadata update."""

import triton
import triton.language as tl


@triton.jit
def _write_decode_slots(token_map, request_rows, lengths, next_lengths, slots, count,
                        row_stride: tl.constexpr, column_stride: tl.constexpr,
                        block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < count
    rows = tl.load(request_rows + offsets, active, 0).to(tl.int64)
    positions = tl.load(lengths + offsets, active, 0)
    locations = tl.load(slots + offsets, active, 0)
    tl.store(token_map + rows * row_stride + positions * column_stride, locations, active)
    tl.store(next_lengths + offsets, positions + 1, active)


def prepare_decode(batch):
    """Prepare an existing unpenalized single-token CUDA batch."""
    if (not batch.forward_mode.is_decode() or batch.is_hybrid
            or batch.model_config.is_encoder_decoder or not batch.spec_algorithm.is_none()
            or batch.token_to_kv_pool_allocator.page_size != 1 or not batch.seq_lens.is_cuda
            or batch.sampling_info.penalizer_orchestrator.is_required):
        return False

    count = len(batch.reqs)
    slots = batch.alloc_token_slots(count)
    token_map = batch.req_to_token_pool.req_to_token
    lengths = batch.seq_lens
    # Overlap holds a result copy that still owns the previous batch lengths.
    next_lengths = lengths.new_empty((count,)) if batch.enable_overlap else lengths
    _write_decode_slots[(triton.cdiv(count, 32),)](
        token_map, batch.req_pool_indices, lengths, next_lengths, slots, count,
        token_map.stride(0), token_map.stride(1), 32, num_warps=1,
    )
    batch.seq_lens = next_lengths
    batch.out_cache_loc = slots
    batch.input_ids, batch.output_ids = batch.output_ids, None
    batch.seq_lens_sum += count
    return True
