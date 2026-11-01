"""Original block-scaled INT8 storage kernels for saved floating activations."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def activation_pack_kernel(source, packed, scales, elements, block: tl.constexpr):
    group = tl.program_id(0)
    positions = group * block + tl.arange(0, block)
    values = tl.load(source + positions, positions < elements, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(values), 0)
    scale = tl.where(maximum > 0.0, maximum / 127.0, 1.0)
    rounded = tl.floor(values / scale + 0.5)
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(packed + positions, quantized, positions < elements)
    tl.store(scales + group, scale)


@triton.jit(do_not_specialize=["elements"])
def activation_unpack_kernel(packed, scales, restored, elements, block: tl.constexpr):
    group = tl.program_id(0)
    positions = group * block + tl.arange(0, block)
    scale = tl.load(scales + group)
    values = tl.load(packed + positions, positions < elements, other=0).to(tl.float32)
    tl.store(restored + positions, values * scale, positions < elements)


@triton.jit(do_not_specialize=["elements"])
def activation_pack4_kernel(source, packed, scales, elements, block: tl.constexpr):
    group = tl.program_id(0)
    pair = tl.arange(0, block // 2)
    even = group * block + pair * 2
    odd = even + 1
    a = tl.load(source + even, even < elements, other=0.0).to(tl.float32)
    b = tl.load(source + odd, odd < elements, other=0.0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(a), 0), tl.max(tl.abs(b), 0))
    scale = tl.where(maximum > 0.0, maximum / 7.0, 1.0)
    qa = tl.minimum(tl.maximum(tl.floor(a / scale + 0.5), -7.0), 7.0).to(tl.int32)
    qb = tl.minimum(tl.maximum(tl.floor(b / scale + 0.5), -7.0), 7.0).to(tl.int32)
    encoded = (qa & 15) | ((qb & 15) << 4)
    tl.store(packed + group * (block // 2) + pair, encoded.to(tl.uint8), even < elements)
    tl.store(scales + group, scale)


@triton.jit(do_not_specialize=["elements"])
def activation_unpack4_kernel(packed, scales, restored, elements, block: tl.constexpr):
    group = tl.program_id(0)
    pair = tl.arange(0, block // 2)
    even = group * block + pair * 2
    odd = even + 1
    encoded = tl.load(packed + group * (block // 2) + pair, even < elements, other=0)
    a = (encoded & 15).to(tl.int32)
    b = ((encoded >> 4) & 15).to(tl.int32)
    a = tl.where(a >= 8, a - 16, a).to(tl.float32)
    b = tl.where(b >= 8, b - 16, b).to(tl.float32)
    scale = tl.load(scales + group)
    tl.store(restored + even, a * scale, even < elements)
    tl.store(restored + odd, b * scale, odd < elements)
