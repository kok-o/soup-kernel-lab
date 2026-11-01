"""Prototype: two independent KV groups in one eight-sequence native CUTLASS call.

API/layout source: PyTorch v2.11.0 native/transformers/cuda/attention{,_backward}.cu.
No native kernel code is copied. The selected runtime is not patched by this module.
Original padded dropout/RoPE precede packing. KV expand is still a zero-stride view;
six native FP16 head gradients widen before each group's FP32 reduction.
"""

from functools import lru_cache


def grouped_cumulative(cumulative, tokens):
    import torch

    from .compact_decoder import _current

    layout = _current.get()
    cache = (layout.attention_metadata if layout is not None
             and layout.cumulative is cumulative and layout.tokens == tokens else None)
    key = (id(cumulative), cumulative._version, tokens)
    if cache is not None and key in cache:
        return cache[key]
    result = torch.cat((cumulative, cumulative[1:] + tokens))
    if cache is not None:
        cache[key] = result
    return result


def fold_groups(packed):
    _, tokens, heads, width = packed.shape
    return packed.reshape(1, tokens, 2, heads // 2, width).permute(
        0, 2, 1, 3, 4).reshape(1, 2 * tokens, heads // 2, width)


def unfold_groups(folded, tokens):
    _, _, heads, width = folded.shape
    return folded.reshape(1, 2, tokens, heads, width).permute(
        0, 2, 1, 3, 4).reshape(tokens, 2 * heads, width)


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    from .varlen_gqa import _pack

    class _GroupedRaggedGQA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, indices, cumulative, scale, max_sequence):
            # Retain only the folded native buffers, not both layouts across layers.
            qfold, kfold, vfold = [fold_groups(_pack(tensor, indices))
                                  for tensor in (query, key, value)]
            valid_tokens = qfold.shape[1] // 2
            grouped = grouped_cumulative(cumulative, valid_tokens)
            kg = kfold.expand(-1, -1, 6, -1)
            vg = vfold.expand(-1, -1, 6, -1)
            native, lse, seed, offset, maxq, maxk = torch.ops.aten._efficient_attention_forward(
                qfold, kg, vg, None, grouped, grouped,
                max_sequence or query.shape[2], max_sequence or query.shape[2],
                0.0, 1, True, scale=scale,
            )
            batch, heads, tokens, width = query.shape
            compact_output = unfold_groups(native, valid_tokens)
            if indices is None:
                output = compact_output
            else:
                output = torch.zeros((batch * tokens, heads, width), device=query.device,
                                     dtype=torch.float16)
                output.index_copy_(0, indices, compact_output)
            ctx.save_for_backward(qfold, kfold, vfold, indices, cumulative, grouped,
                                  native, lse, seed, offset)
            ctx.input_types = tuple(tensor.dtype for tensor in (query, key, value))
            ctx.shape, ctx.valid_tokens = query.shape, valid_tokens
            ctx.maxima, ctx.scale = (maxq, maxk), scale
            return output.view(batch, tokens, heads, width).transpose(1, 2)

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            qfold, kfold, vfold, indices, _original_cu, grouped, native, lse, seed, offset = (
                ctx.saved_tensors)
            packed_incoming = fold_groups(_pack(incoming, indices))
            kg = kfold.expand(-1, -1, 6, -1)
            vg = vfold.expand(-1, -1, 6, -1)
            gq, gk, gv, _ = torch.ops.aten._efficient_attention_backward(
                packed_incoming, qfold, kg, vg, None, native, grouped, grouped,
                *ctx.maxima, lse, 0.0, seed, offset, 1, False, scale=ctx.scale,
            )
            if gk.shape[2] != 6 or gv.shape[2] != 6:
                raise RuntimeError("grouped native attention did not return six KV head gradients")
            batch, heads, tokens, width = ctx.shape
            valid = ctx.valid_tokens
            qgradient = unfold_groups(gq.to(ctx.input_types[0]), valid)
            kv = []
            for value, dtype in zip((gk, gv), ctx.input_types[1:], strict=True):
                reduced = value[0].float().reshape(2, valid, 6, width).sum(2).to(dtype)
                kv.append(reduced.transpose(0, 1).contiguous())
            results = []
            for gradient, count in zip((qgradient, *kv), (heads, 2, 2), strict=True):
                if indices is not None:
                    dense = torch.zeros((batch * tokens, count, width), device=gradient.device,
                                        dtype=gradient.dtype)
                    dense.index_copy_(0, indices, gradient)
                    gradient = dense
                results.append(gradient.view(batch, tokens, count, width).transpose(1, 2))
            return *results, None, None, None, None

    return _GroupedRaggedGQA


def grouped_varlen_attention(query, key, value, indices, cumulative, scale, max_sequence=None):
    from .varlen_gqa import varlen_attention

    supported = (query.ndim == key.ndim == value.ndim == 4 and query.shape[1] == 12
                 and key.shape[1] == value.shape[1] == 2 and query.shape[-1] == 128
                 and (indices is not None or query.shape[0] == 1))
    if not supported:
        return varlen_attention(query, key, value, indices, cumulative, scale, max_sequence)
    return _autograd().apply(query, key, value, indices, cumulative, scale, max_sequence)
