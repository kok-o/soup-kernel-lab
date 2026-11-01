"""Fuse only vanilla PEFT NF4 LoRA's clone, FP16 scaling and FP32 addition.

Reference operation order: peft v0.20.0 src/peft/tuners/lora/bnb.py.
Dropout and both adapter GEMMs use their original modules and full padded shape.
The result is a fresh allocation; no base tensor or saved view is modified.
"""

import math
from functools import lru_cache, wraps


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    class _LoRAAdd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, base, delta, scale):
            from ._lora_kernels import lora_scale_add_kernel

            output = torch.empty_like(base)
            ctx.scale, ctx.delta_dtype = scale, delta.dtype
            lora_scale_add_kernel[((base.numel() + 1023) // 1024,)](
                base, delta, output, base.numel(), scale=scale, block=1024,
                num_warps=4, enable_fp_fusion=False,
            )
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            from ._lora_kernels import lora_scale_gradient_kernel

            gradient = None
            if ctx.needs_input_grad[1]:
                incoming = incoming.contiguous()
                gradient = torch.empty_like(incoming, dtype=ctx.delta_dtype)
                lora_scale_gradient_kernel[((incoming.numel() + 1023) // 1024,)](
                    incoming, gradient, incoming.numel(), scale=ctx.scale, block=1024,
                    num_warps=4, enable_fp_fusion=False,
                )
            return incoming if ctx.needs_input_grad[0] else None, gradient, None

    return _LoRAAdd


def fused_lora_add(base, delta, scale):
    import torch

    supported = (base.is_cuda and delta.is_cuda and base.device == delta.device
                 and base.dtype == torch.float32 and delta.dtype == torch.float16
                 and base.shape == delta.shape and base.is_contiguous()
                 and delta.is_contiguous() and base.numel() > 0
                 and isinstance(scale, (int, float)) and math.isfinite(scale))
    if not supported:
        return base.clone() + delta * scale
    return _autograd().apply(base, delta, float(scale))


def patch_vanilla_nf4_lora():
    import torch

    try:
        from peft.tuners.lora.bnb import Linear4bit
    except (ImportError, AttributeError):
        return False
    original = Linear4bit.forward
    if getattr(original, "_setww_lora_scale_add", False):
        return True

    @wraps(original)
    def forward(self, x, *args, **kwargs):
        adapters = self.active_adapters
        supported = (self.training and torch.is_grad_enabled() and x.is_cuda
                     and not args and not kwargs and not self.disable_adapters
                     and not self.merged and len(adapters) == 1
                     and adapters[0] in self.lora_A and adapters[0] in self.lora_B
                     and adapters[0] not in self.lora_variant
                     and not self.base_layer.weight.requires_grad
                     and torch.is_autocast_enabled("cuda")
                     and torch.get_autocast_dtype("cuda") == torch.float16)
        if not supported:
            return original(self, x, *args, **kwargs)
        adapter = adapters[0]
        scale = self.scaling[adapter]
        if not isinstance(scale, (int, float)) or not math.isfinite(scale):
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        base = self.base_layer(x)
        delta = self.lora_B[adapter](self.lora_A[adapter](self.lora_dropout[adapter](x)))
        return fused_lora_add(base, delta, scale)

    forward._setww_lora_scale_add = True
    Linear4bit.forward = forward
    return True
