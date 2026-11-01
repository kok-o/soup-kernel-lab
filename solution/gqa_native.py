"""Native T4 efficient GQA with shared K/V and FP32 head-gradient reduction.

Uses the installed aten efficient-attention API, never a math fallback or a ready
kernel library. Two KV groups share one output buffer. Only FP16 K/V with two
heads are saved. Native per-query-head K/V gradients are summed in FP32.
"""

import weakref
from contextvars import ContextVar
from functools import lru_cache, wraps

_mask_cache = ContextVar("setww_gqa_mask_cache", default=None)
_varlen_enabled = False


def _attention_bias(mask, dtype):
    import torch

    cache = _mask_cache.get()
    key = (id(mask), mask._version, dtype)
    previous = cache.get(key) if cache is not None else None
    if previous is not None and previous[0]() is mask:
        return previous[1]
    if mask.dtype == torch.bool:
        bias = torch.zeros_like(mask, dtype=dtype).masked_fill_(~mask, float("-inf"))
    else:
        bias = mask.to(dtype)
    if bias.stride(-2) % 8 or bias.stride(-1) != 1:
        # Native Cutlass bias rows require eight-element alignment.
        padded = torch.empty((*bias.shape[:-1], (bias.shape[-1] + 7) // 8 * 8),
                             dtype=dtype, device=bias.device)
        padded[..., :bias.shape[-1]].copy_(bias)
        bias = padded[..., :bias.shape[-1]]
    if cache is not None:
        cache[key] = (weakref.ref(mask), bias)
    return bias


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    class _GQA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, bias, scale):
            q, k, v = (x.to(torch.float16) for x in (query, key, value))
            batch, heads, tokens, width = q.shape
            groups, repeat = k.shape[1], heads // k.shape[1]
            output = torch.empty((batch, tokens, heads, width), device=q.device,
                                 dtype=q.dtype).transpose(1, 2)
            statistics, pieces = [], []
            expanded_bias = bias.expand(batch, repeat, tokens, tokens)
            for group in range(groups):
                start, stop = group*repeat, (group+1)*repeat
                kg = k[:, group:group+1].expand(-1, repeat, -1, -1)
                vg = v[:, group:group+1].expand(-1, repeat, -1, -1)
                piece, lse, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention(
                    q[:, start:stop], kg, vg, expanded_bias, True, 0.0, False, scale=scale,
                )
                output[:, start:stop].copy_(piece)
                pieces.append(piece)
                statistics.extend((lse, seed, offset))
            # Save the two native B,T,H,D-backed outputs. The assembled output
            # only serves o_proj; frozen o_proj does not retain its activation.
            ctx.save_for_backward(q, k, v, bias, *pieces, *statistics)
            ctx.scale, ctx.repeat = scale, repeat
            ctx.input_dtypes = (query.dtype, key.dtype, value.dtype)
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            q, k, v, bias, *saved = ctx.saved_tensors
            pieces, statistics = saved[:k.shape[1]], saved[k.shape[1]:]
            batch, heads, tokens, _ = q.shape
            repeat = ctx.repeat
            gradients = [torch.empty_like(tensor, dtype=dtype) for tensor, dtype in
                         zip((q, k, v), ctx.input_dtypes, strict=True)]
            expanded_bias = bias.expand(batch, repeat, tokens, tokens)
            for group in range(k.shape[1]):
                start, stop = group*repeat, (group+1)*repeat
                kg = k[:, group:group+1].expand(-1, repeat, -1, -1)
                vg = v[:, group:group+1].expand(-1, repeat, -1, -1)
                lse, seed, offset = statistics[group*3:group*3+3]
                # Original native output already has the implicit Cutlass
                # o_strideM = group_heads*head_dim. No backward copy is needed.
                native_output = pieces[group]
                gq, gk, gv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                    incoming[:, start:stop], q[:, start:stop], kg, vg, expanded_bias,
                    native_output, lse, seed, offset, 0.0, [True, True, True, False],
                    False, scale=ctx.scale,
                )
                if gk.shape[1] != repeat or gv.shape[1] != repeat:
                    raise RuntimeError("native attention did not return per-head KV gradients")
                gradients[0][:, start:stop].copy_(gq)
                gradients[1][:, group:group+1].copy_(gk.float().sum(1, keepdim=True))
                gradients[2][:, group:group+1].copy_(gv.float().sum(1, keepdim=True))
                del gq, gk, gv, native_output
            return *gradients, None, None

    return _GQA


def shared_kv_attention(query, key, value, attention_mask, scale):
    import torch

    if _varlen_enabled:
        from .varlen_gqa import metadata, varlen_attention

        packed = metadata(query)
        if packed is not None:
            return varlen_attention(query, key, value, *packed, scale)
    bias = _attention_bias(attention_mask, torch.float16)
    return _autograd().apply(query, key, value, bias, scale)


def patch_qwen_shared_kv():
    import torch

    try:
        from transformers.models.qwen2 import modeling_qwen2 as model
    except ImportError:
        return False
    attention = getattr(model, "Qwen2Attention", None)
    registry = getattr(model, "ALL_ATTENTION_FUNCTIONS", None)
    if attention is None or registry is None or not hasattr(registry, "register"):
        return False
    original = registry["sdpa"]
    if getattr(original, "_setww_shared_kv", False):
        return True

    @wraps(original)
    def sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
             is_causal=None, position_bias=None, **kwargs):
        supported = (isinstance(module, attention) and module.training and torch.is_grad_enabled()
                     and query.is_cuda and query.ndim == key.ndim == value.ndim == 4
                     and query.shape[1] == 12 and key.shape[1] == value.shape[1] == 2
                     and query.shape[-1] == key.shape[-1] == value.shape[-1] == 128
                     and query.shape[2] == key.shape[2] == value.shape[2]
                     and all(t.stride(-1) == 1 for t in (query, key, value))
                     and all(t.dtype in (torch.float16, torch.float32) for t in (query, key, value))
                     and attention_mask is not None and attention_mask.ndim == 4
                     and attention_mask.shape[-2:] == (query.shape[2], key.shape[2])
                     and attention_mask.shape[1] == 1 and not attention_mask.requires_grad
                     and dropout == 0.0 and position_bias is None
                     and not kwargs.get("output_attentions")
                     and torch.is_autocast_enabled("cuda")
                     and torch.get_autocast_dtype("cuda") == torch.float16)
        if not supported:
            return original(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, is_causal=is_causal,
                            position_bias=position_bias, **kwargs)
        result = shared_kv_attention(query, key, value, attention_mask, scaling)
        return result.transpose(1, 2).contiguous(), None

    sdpa._setww_shared_kv = True
    registry.register("sdpa", sdpa)
    original_model = model.Qwen2Model.forward

    @wraps(original_model)
    def forward(self, *args, **kwargs):
        token = _mask_cache.set({})
        try:
            return original_model(self, *args, **kwargs)
        finally:
            _mask_cache.reset(token)

    model.Qwen2Model.forward = forward
    return True
