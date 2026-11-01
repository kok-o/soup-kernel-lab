"""Upcast each FP16 MLP branch before one FP32 sum, without cast temporaries."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["elements"])
    def fork_gradient_sum_kernel(left, right, output, elements, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < elements
        first = tl.load(left + offsets, valid, other=0).to(tl.float32)
        second = tl.load(right + offsets, valid, other=0).to(tl.float32)
        tl.store(output + offsets, first + second, valid)

    return fork_gradient_sum_kernel


def fork_gradient_sum(left, right, dtype):
    import torch

    from .dtype_cast import _convert

    supported = (left.is_cuda and left.device == right.device
                 and left.dtype == right.dtype == torch.float16 and dtype == torch.float32
                 and left.shape == right.shape and left.is_contiguous() and right.is_contiguous()
                 and not torch.is_grad_enabled())
    if not supported:
        return _convert(left, dtype) + _convert(right, dtype)
    output = torch.empty_like(left, dtype=dtype)
    if left.numel():
        with torch.cuda.device(left.device):
            _kernel()[((left.numel() + 4095) // 4096,)](
                left, right, output, left.numel(), block=4096,
                num_warps=4, enable_fp_fusion=False,
            )
    return output
