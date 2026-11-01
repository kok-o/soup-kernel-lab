"""Original lossless bool-mask bitpacking and native-order masked scaling."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def dropout_mask_pack_kernel(mask, packed, elements, groups: tl.constexpr):
    group = tl.program_id(0) * groups + tl.arange(0, groups)
    bit = tl.arange(0, 8)
    position = group[:, None] * 8 + bit[None, :]
    selected = tl.load(mask + position, position < elements, other=0).to(tl.int32)
    encoded = tl.sum(selected << bit[None, :], 1).to(tl.uint8)
    tl.store(packed + group, encoded, group * 8 < elements)


@triton.jit(do_not_specialize=["elements"])
def dropout_mask_gradient_kernel(incoming, packed, output, elements,
                                 scale: tl.constexpr, block: tl.constexpr):
    position = tl.program_id(0) * block + tl.arange(0, block)
    valid = position < elements
    encoded = tl.load(packed + position // 8, valid, other=0).to(tl.int32)
    keep = ((encoded >> (position % 8)) & 1).to(tl.float32)
    gradient = tl.load(incoming + position, valid, other=0).to(tl.float32)
    # Native CUDA order is float(mask)*gradient*scale. In particular 0*Inf is NaN.
    tl.store(output + position, keep * gradient * scale, valid)
