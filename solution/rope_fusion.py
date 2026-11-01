"""Fuse Qwen RoPE and the query's existing SDPA FP16 cast.

Keys stay FP32 until after repeat_kv: moving that cast would also narrow its
six-head gradient reduction. Reference: transformers v5.16.1 modeling_qwen2.py.
"""

from contextvars import ContextVar
from functools import lru_cache, wraps

_active = ContextVar("setww_fp32_rotary_scope", default=False)


def _launch(kernel, source, cosine, sine, output):
    elements = source.shape[0] * source.shape[2] * 128
    factor_batch = cosine.stride(0) if cosine.shape[0] != 1 else 0
    kernel[((elements + 511) // 512, source.shape[1])](
        source, cosine, sine, output, elements, source.shape[2],
        *source.stride()[:3], *output.stride()[:3], factor_batch, cosine.stride(1),
        block=512, num_warps=4, enable_fp_fusion=False,
    )


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    class _Rotary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, cosine, sine):
            from ._rope_kernels import rotary_forward_kernel

            qout = torch.empty_like(query, dtype=torch.float16)
            kout = torch.empty_like(key)
            _launch(rotary_forward_kernel, query, cosine, sine, qout)
            _launch(rotary_forward_kernel, key, cosine, sine, kout)
            ctx.save_for_backward(cosine, sine)
            ctx.layouts = [(value.shape, value.stride()) for value in (query, key)]
            ctx.set_materialize_grads(False)
            return qout, kout

        @staticmethod
        @once_differentiable
        def backward(ctx, qgrad, kgrad):
            from ._rope_kernels import rotary_backward_kernel

            cosine, sine = ctx.saved_tensors
            result = []
            for incoming, (shape, stride) in zip((qgrad, kgrad), ctx.layouts, strict=True):
                if incoming is None:
                    result.append(None)
                    continue
                output = torch.empty_strided(shape, stride, device=incoming.device,
                                             dtype=torch.float32)
                _launch(rotary_backward_kernel, incoming, cosine, sine, output)
                result.append(output)
            return *result, None, None

    return _Rotary


def fused_rotary(query, key, cosine, sine):
    """Guarded CUDA primitive: Q becomes FP16; K and both input gradients stay FP32."""
    import torch

    supported = (query.is_cuda and key.device == query.device and cosine.device == query.device
                 and sine.device == query.device and query.ndim == key.ndim == 4
                 and query.dtype == key.dtype == cosine.dtype == sine.dtype == torch.float32
                 and query.shape[0] == key.shape[0] and query.shape[2] == key.shape[2]
                 and query.shape[-1] == key.shape[-1] == 128
                 and query.stride(-1) == key.stride(-1) == 1
                 and cosine.ndim == 3 and cosine.shape == sine.shape
                 and cosine.shape[0] in (1, query.shape[0])
                 and cosine.shape[1:] == (query.shape[2], 128)
                 and cosine.stride() == sine.stride() and cosine.stride(-1) == 1
                 and not cosine.requires_grad and not sine.requires_grad
                 and query.numel() > 0)
    if not supported:
        def rotate(value):
            half = value.shape[-1] // 2
            return torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        c, s = cosine.unsqueeze(1), sine.unsqueeze(1)
        return (query*c + rotate(query)*s).half(), key*c + rotate(key)*s
    return _autograd().apply(query, key, cosine, sine)


def patch_qwen_rotary():
    import torch

    try:
        from transformers.models.qwen2 import modeling_qwen2 as model
    except ImportError:
        return False
    attention = getattr(model, "Qwen2Attention", None)
    if attention is None or not hasattr(model, "apply_rotary_pos_emb"):
        return False
    original_rotary = model.apply_rotary_pos_emb
    if getattr(original_rotary, "_setww_rotary", False):
        return True
    original_attention = attention.forward

    @wraps(original_rotary)
    def rotary(query, key, cosine, sine, unsqueeze_dim=1):
        if (not _active.get() or unsqueeze_dim != 1 or query.dtype != torch.float32
                or key.dtype != torch.float32 or query.shape[1] != 12 or key.shape[1] != 2
                or cosine.requires_grad or sine.requires_grad):
            return original_rotary(query, key, cosine, sine, unsqueeze_dim)
        return fused_rotary(query, key, cosine, sine)

    @wraps(original_attention)
    def forward(self, *args, **kwargs):
        supported = (self.training and torch.is_grad_enabled() and len(args) < 4
                     and getattr(self.config, "_attn_implementation", None) == "sdpa"
                     and not kwargs.get("past_key_values") and not kwargs.get("output_attentions")
                     and torch.is_autocast_enabled("cuda")
                     and torch.get_autocast_dtype("cuda") == torch.float16)
        token = _active.set(supported)
        try:
            return original_attention(self, *args, **kwargs)
        finally:
            _active.reset(token)

    rotary._setww_rotary = True
    model.apply_rotary_pos_emb = rotary
    attention.forward = forward
    return True
