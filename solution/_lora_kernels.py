"""Original LoRA scale/add kernels with the two original FP16 rounding boundaries."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def lora_scale_add_kernel(base, delta, output, elements, scale: tl.constexpr,
                          block: tl.constexpr):
    positions = tl.program_id(0) * block + tl.arange(0, block)
    valid = positions < elements
    a = tl.load(base + positions, valid, other=0).to(tl.float32)
    b = tl.load(delta + positions, valid, other=0).to(tl.float32)
    # PEFT first stores delta*scale in delta.dtype, then adds to the FP32 base.
    scaled = (b * scale).to(delta.dtype.element_ty).to(tl.float32)
    tl.store(output + positions, a + scaled, valid)


@triton.jit(do_not_specialize=["elements"])
def lora_scale_gradient_kernel(incoming, delta_grad, elements, scale: tl.constexpr,
                               block: tl.constexpr):
    positions = tl.program_id(0) * block + tl.arange(0, block)
    valid = positions < elements
    # AddBackward casts the incoming FP32 gradient to FP16 before MulBackward.
    gradient = tl.load(incoming + positions, valid, other=0)
    narrowed = gradient.to(delta_grad.dtype.element_ty).to(tl.float32)
    tl.store(delta_grad + positions, narrowed * scale, valid)
