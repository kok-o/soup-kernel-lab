"""Narrow FP32 frozen-weight Qwen RMSNorm patch; unsupported inputs use HF."""

from functools import lru_cache, wraps


@lru_cache(maxsize=1)
def _autograd():
    import torch
    import triton
    from torch.autograd.function import once_differentiable

    from ._rmsnorm_kernels import rmsnorm_backward_kernel, rmsnorm_forward_kernel

    class _FrozenRMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source, weight, epsilon):
            width = source.shape[-1]
            output = torch.empty_like(source)
            inverse = torch.empty(source.numel() // width, dtype=torch.float32,
                                  device=source.device)
            with torch.cuda.device(source.device):
                rmsnorm_forward_kernel[(inverse.numel(),)](
                    source, weight, output, inverse, width, epsilon,
                    triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
                )
            ctx.save_for_backward(source, weight, inverse)
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            source, weight, inverse = ctx.saved_tensors
            width = source.shape[-1]
            gradient = torch.empty_like(source)
            with torch.cuda.device(source.device):
                rmsnorm_backward_kernel[(inverse.numel(),)](
                    incoming.contiguous(), source, weight, inverse, gradient, width,
                    triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
                )
            return gradient, None, None

    return _FrozenRMSNorm


def frozen_rmsnorm(source, weight, epsilon):
    return _autograd().apply(source, weight, epsilon)


def patch_qwen_rmsnorm():
    import torch

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    except (ImportError, AttributeError):
        return False
    original = Qwen2RMSNorm.forward
    if getattr(original, "_setww_rmsnorm", False):
        return True

    @wraps(original)
    def forward(self, hidden_states):
        weight = self.weight
        if (not self.training or not hidden_states.is_cuda or not hidden_states.is_contiguous()
                or hidden_states.dtype != torch.float32 or weight.dtype != torch.float32
                or weight.requires_grad or not weight.is_contiguous()
                or weight.device != hidden_states.device or hidden_states.numel() == 0
                or hidden_states.shape[-1] != weight.numel() or weight.numel() > 4096):
            return original(self, hidden_states)
        return frozen_rmsnorm(hidden_states, weight, self.variance_epsilon)

    forward._setww_rmsnorm = True
    Qwen2RMSNorm.forward = forward
    return True
