"""Frozen FP16 head with original Triton products and a first-order backward.

PyTorch remains the fallback for CPU, non-FP16 compute and trainable weights.
Only the scoped chunked-head implementation calls this function.
"""

from functools import lru_cache


def _product(left, right, splits=1):
    import torch
    import triton

    from ._head_kernels import head_product_kernel, head_reduce_kernel

    rows, reduction = left.shape
    columns = right.shape[1]
    output = torch.empty((rows, columns), dtype=torch.float16, device=left.device)
    if rows == 0:
        return output
    partial = (output if splits == 1 else torch.empty(
        (splits, rows, columns), dtype=torch.float32, device=left.device
    ))
    tile_rows, tile_columns, tile_k = 32, 64, 32
    grid = (triton.cdiv(rows, tile_rows) * triton.cdiv(columns, tile_columns), splits)
    with torch.cuda.device(left.device):
        head_product_kernel[grid](
            left, right, partial, rows, columns, reduction,
            *left.stride(), *right.stride(), splits,
            tile_rows, tile_columns, tile_k, num_warps=4,
        )
        if splits > 1:
            head_reduce_kernel[(triton.cdiv(rows * columns, 128),)](
                partial, output, rows * columns, splits=splits, block=128, num_warps=4,
            )
    return output


@lru_cache(maxsize=1)
def _linear_autograd():
    import torch
    import triton
    from torch.autograd.function import once_differentiable

    class FrozenHead(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight):
            compute_hidden = hidden.to(torch.float16).contiguous()
            compute_weight = weight.to(torch.float16).contiguous()
            ctx.save_for_backward(compute_weight)
            ctx.input_dtype = hidden.dtype
            return _product(compute_hidden, compute_weight.t())

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            if not ctx.needs_input_grad[0]:
                return None, None
            weight, = ctx.saved_tensors
            # The baseline autocast linear rounds dLogits to its compute dtype.
            incoming = incoming.to(torch.float16).contiguous()
            splits = triton.next_power_of_2(min(32, triton.cdiv(weight.shape[0], 8192)))
            hidden_grad = _product(incoming, weight, splits=splits)
            return hidden_grad.to(ctx.input_dtype), None

    return FrozenHead


def frozen_linear(hidden, weight):
    """Match bias-free frozen F.linear when the effective CUDA dtype is FP16."""
    import torch
    import torch.nn.functional as functional

    fp16_compute = hidden.dtype == weight.dtype == torch.float16
    if hidden.is_cuda and torch.is_autocast_enabled("cuda"):
        fp16_compute = torch.get_autocast_dtype("cuda") == torch.float16
    supported = (
        hidden.is_cuda and weight.is_cuda and not weight.requires_grad
        and hidden.ndim == weight.ndim == 2 and hidden.shape[1] == weight.shape[1]
        and fp16_compute
    )
    if not supported:
        return functional.linear(hidden, weight)
    return _linear_autograd().apply(hidden, weight)
