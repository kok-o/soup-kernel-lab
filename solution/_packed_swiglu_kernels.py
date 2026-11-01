"""Starter-derived SwiGLU backward with direct INT4 reads; EXCLUDED from own quota."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def example_swiglu_packed_bwd_kernel(incoming, gate_bytes, gate_scale, up_bytes, up_scale,
                                    gate_grad, up_grad, elements, block: tl.constexpr):
    position = tl.program_id(0) * block + tl.arange(0, block)
    valid = position < elements
    shift = (position % 2) * 4
    ga = tl.load(gate_bytes + position // 2, valid, other=0).to(tl.int32)
    ua = tl.load(up_bytes + position // 2, valid, other=0).to(tl.int32)
    gq = (ga >> shift) & 15
    uq = (ua >> shift) & 15
    gq = tl.where(gq >= 8, gq - 16, gq).to(tl.float32)
    uq = tl.where(uq >= 8, uq - 16, uq).to(tl.float32)
    gs = tl.load(gate_scale + position // 64, valid, other=1.0)
    us = tl.load(up_scale + position // 64, valid, other=1.0)
    # The former unpack stores FP16 before the starter reloads it into FP32.
    gate = (gq * gs).to(tl.float16).to(tl.float32)
    up = (uq * us).to(tl.float16).to(tl.float32)
    grad = tl.load(incoming + position, valid, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(gate_grad + position, dgate.to(gate_grad.dtype.element_ty), valid)
    tl.store(up_grad + position, dup.to(up_grad.dtype.element_ty), valid)
