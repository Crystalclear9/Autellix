"""Pack every layer of a GPU KV transfer with one Triton kernel."""
import triton
import triton.language as tl


@triton.jit
def _copy_blocks(pointers, indices, payload, count, capacity: tl.constexpr,
                 elements: tl.constexpr, rows: tl.constexpr, channels: tl.constexpr,
                 token_major: tl.constexpr, to_payload: tl.constexpr, tile: tl.constexpr):
    row = tl.program_id(1)
    tiles = tl.cdiv(elements, tile)
    slot = tl.program_id(0) // tiles
    offset = (tl.program_id(0) % tiles) * tile + tl.arange(0, tile)
    index = tl.load(indices + slot).to(tl.int64)
    layer = tl.load(pointers + row // channels).to(tl.pointer_type(payload.dtype.element_ty))
    cache_offset = ((row % channels) * capacity + index) * elements + offset
    if token_major:
        payload_offset = (slot.to(tl.int64) * rows + row) * elements + offset
    else:
        payload_offset = (row.to(tl.int64) * count + slot) * elements + offset
    if to_payload:
        values = tl.load(layer + cache_offset, offset < elements, other=0)
        tl.store(payload + payload_offset, values, offset < elements)
    else:
        values = tl.load(payload + payload_offset, offset < elements, other=0)
        tl.store(layer + cache_offset, values, offset < elements)


class PackedKVLayout:
    def __init__(self, layers, *, channels, token_major=False):
        import torch

        first = layers[0]
        if any(layer.shape != first.shape or layer.device != first.device or layer.dtype != first.dtype
               or not layer.is_contiguous() for layer in layers):
            raise ValueError("packed KV layers must share a contiguous layout")
        self.device, self.dtype = first.device, first.dtype
        self.channels, self.token_major = channels, token_major
        self.capacity = first.shape[0 if channels == 1 else 1]
        self.block_shape = first.shape[1 if channels == 1 else 2:]
        self.elements = first.numel() // (channels * self.capacity)
        self.rows = len(layers) * channels
        self.bytes_per_block = self.rows * self.elements * first.element_size()
        self.pointers = torch.tensor([layer.data_ptr() for layer in layers], dtype=torch.int64, device=self.device)

    def shape(self, count):
        if self.token_major:
            return (count, self.rows, *self.block_shape)
        return (self.rows // self.channels, self.channels, count, *self.block_shape)

    def move(self, indices, payload, *, pack):
        # Index and layer pointer loads are shared by each vector of a block.
        tile = 512
        indices = indices.to(device=self.device, non_blocking=True).contiguous()
        count = len(indices)
        grid = (count * triton.cdiv(self.elements, tile), self.rows)
        _copy_blocks[grid](self.pointers, indices, payload, count, self.capacity, self.elements,
                           self.rows, self.channels, self.token_major, pack, tile)
