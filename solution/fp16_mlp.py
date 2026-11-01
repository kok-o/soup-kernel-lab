"""Preserve FP32 MLP boundaries while keeping NF4/SwiGLU intermediates FP16.

The unmodified starter SwiGLU is reused and remains excluded from own kernels.
Separate fork gradients upcast before accumulation, matching the original casts.
"""

from functools import lru_cache, wraps


@lru_cache(maxsize=1)
def _fork_cast():
    import torch

    from .dtype_cast import _convert

    class _Fork(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source):
            ctx.dtype = source.dtype
            ctx.set_materialize_grads(False)
            value = _convert(source, torch.float16)
            return value.view_as(value), value.view_as(value)

        @staticmethod
        def backward(ctx, gate_gradient, up_gradient):
            if gate_gradient is None:
                return _convert(up_gradient, ctx.dtype) if up_gradient is not None else None
            if up_gradient is None:
                return _convert(gate_gradient, ctx.dtype)
            from .fork_sum import fork_gradient_sum

            return fork_gradient_sum(gate_gradient, up_gradient, ctx.dtype)

    return _Fork


def patch_fp16_mlp():
    import torch

    try:
        from bitsandbytes.nn import Linear4bit
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    except (ImportError, AttributeError):
        return False
    original = Qwen2MLP.forward
    if getattr(original, "_setww_fp16_mlp", False):
        return True

    @wraps(original)
    def forward(self, hidden):
        modules = (self.gate_proj, self.up_proj, self.down_proj)
        if (not self.training or not hidden.is_cuda or hidden.dtype != torch.float32
                or not hidden.is_contiguous()
                or getattr(getattr(self, "config", None), "hidden_act", "silu") != "silu"
                or not all(isinstance(layer, Linear4bit) and not layer.weight.requires_grad
                           and layer.compute_dtype == torch.float16 and layer.compute_type_is_set
                           for layer in modules)):
            return original(self, hidden)
        from .dtype_cast import training_cast
        from .swiglu import swiglu

        gate_input, up_input = _fork_cast().apply(hidden)
        output = self.down_proj(swiglu(self.gate_proj(gate_input), self.up_proj(up_input)))
        return training_cast(output, hidden.dtype)

    forward._setww_fp16_mlp = True
    Qwen2MLP.forward = forward
    return True
