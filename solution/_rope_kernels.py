"""Original Qwen half-rotation kernels; all products and sums remain FP32."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements", "tokens"])
def rotary_forward_kernel(source, cosine, sine, output, elements, tokens,
                          source_batch, source_head, source_token,
                          output_batch, output_head, output_token,
                          factor_batch, factor_token, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    head = tl.program_id(1)
    valid = index < elements
    dim = index % 128
    token = (index // 128) % tokens
    batch = index // (128 * tokens)
    partner = tl.where(dim < 64, dim + 64, dim - 64)
    source_row = batch * source_batch + head * source_head + token * source_token
    output_row = batch * output_batch + head * output_head + token * output_token
    factor_row = batch * factor_batch + token * factor_token
    a = tl.load(source + source_row + dim, valid, other=0).to(tl.float32)
    b = tl.load(source + source_row + partner, valid, other=0).to(tl.float32)
    c = tl.load(cosine + factor_row + dim, valid, other=0).to(tl.float32)
    s = tl.load(sine + factor_row + dim, valid, other=0).to(tl.float32)
    rotated = tl.where(dim < 64, -b, b)
    value = a * c + rotated * s
    tl.store(output + output_row + dim, value, valid)


@triton.jit(do_not_specialize=["elements", "tokens"])
def rotary_backward_kernel(incoming, cosine, sine, output, elements, tokens,
                           source_batch, source_head, source_token,
                           output_batch, output_head, output_token,
                           factor_batch, factor_token, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    head = tl.program_id(1)
    valid = index < elements
    dim = index % 128
    token = (index // 128) % tokens
    batch = index // (128 * tokens)
    partner = tl.where(dim < 64, dim + 64, dim - 64)
    source_row = batch * source_batch + head * source_head + token * source_token
    output_row = batch * output_batch + head * output_head + token * output_token
    factor_row = batch * factor_batch + token * factor_token
    a = tl.load(incoming + source_row + dim, valid, other=0).to(tl.float32)
    b = tl.load(incoming + source_row + partner, valid, other=0).to(tl.float32)
    c = tl.load(cosine + factor_row + dim, valid, other=0).to(tl.float32)
    s = tl.load(sine + factor_row + partner, valid, other=0).to(tl.float32)
    rotated = b * s
    contribution = tl.where(dim < 64, rotated, -rotated)
    tl.store(output + output_row + dim, a * c + contribution, valid)
