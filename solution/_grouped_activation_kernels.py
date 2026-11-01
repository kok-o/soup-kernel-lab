"""Several independent codec blocks per CTA; unchanged byte/scales format."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def grouped_activation_pack8_kernel(source, packed, scales, elements,
                                    block: tl.constexpr, groups: tl.constexpr):
    group = tl.program_id(0)*groups+tl.arange(0, groups)
    positions = group[:, None]*block+tl.arange(0, block)[None, :]
    values = tl.load(source+positions, positions < elements, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(values), 1)
    scale = tl.where(maximum > 0.0, maximum/127.0, 1.0)
    rounded = tl.floor(values/scale[:, None]+0.5)
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(packed+positions, quantized, positions < elements)
    tl.store(scales+group, scale, group*block < elements)


@triton.jit(do_not_specialize=["elements"])
def grouped_activation_pack4_kernel(source, packed, scales, elements,
                                    block: tl.constexpr, groups: tl.constexpr):
    group = tl.program_id(0)*groups+tl.arange(0, groups)
    pair = tl.arange(0, block//2)
    even = group[:, None]*block+pair[None, :]*2
    odd = even+1
    a = tl.load(source+even, even < elements, other=0.0).to(tl.float32)
    b = tl.load(source+odd, odd < elements, other=0.0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(a), 1), tl.max(tl.abs(b), 1))
    scale = tl.where(maximum > 0.0, maximum/7.0, 1.0)
    qa = tl.minimum(tl.maximum(tl.floor(a/scale[:, None]+0.5), -7.0), 7.0).to(tl.int32)
    qb = tl.minimum(tl.maximum(tl.floor(b/scale[:, None]+0.5), -7.0), 7.0).to(tl.int32)
    encoded = (qa & 15) | ((qb & 15) << 4)
    tl.store(packed+group[:, None]*(block//2)+pair[None, :], encoded.to(tl.uint8),
             even < elements)
    tl.store(scales+group, scale, group*block < elements)


@triton.jit(do_not_specialize=["elements"])
def grouped_activation_unpack8_kernel(packed, scales, restored, elements,
                                      block: tl.constexpr, tile: tl.constexpr):
    position = tl.program_id(0)*tile+tl.arange(0, tile)
    scale = tl.load(scales+position//block, position < elements, other=1.0)
    value = tl.load(packed+position, position < elements, other=0).to(tl.float32)
    tl.store(restored+position, value*scale, position < elements)


@triton.jit(do_not_specialize=["elements"])
def grouped_activation_unpack4_kernel(packed, scales, restored, elements,
                                      block: tl.constexpr, tile: tl.constexpr):
    pair = tl.program_id(0)*(tile//2)+tl.arange(0, tile//2)
    even = pair*2
    odd = even+1
    encoded = tl.load(packed+pair, even < elements, other=0)
    a = (encoded & 15).to(tl.int32)
    b = ((encoded >> 4) & 15).to(tl.int32)
    a = tl.where(a >= 8, a-16, a).to(tl.float32)
    b = tl.where(b >= 8, b-16, b).to(tl.float32)
    scale = tl.load(scales+even//block, even < elements, other=1.0)
    tl.store(restored+even, a*scale, even < elements)
    tl.store(restored+odd, b*scale, odd < elements)
