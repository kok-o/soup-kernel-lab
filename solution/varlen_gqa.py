"""Native CUTLASS variable-length attention after original padded dropout/RoPE.

Only real Q/K/V rows of this batch are packed. Two KV groups remain separate;
per-head native FP16 K/V gradients widen before their FP32 six-head reduction.
The native output of each group is retained in its original backward layout.
"""

from functools import lru_cache


def metadata(query):
    import torch

    from .gqa_native import _mask_cache
    from .mlp_unpadding import _active_tokens

    active = _active_tokens.get()
    if active is None or tuple(active[1]) != (query.shape[0], query.shape[2]):
        return None
    indices = active[0]
    cache = _mask_cache.get()
    key = ("varlen", id(indices), indices._version)
    if cache is not None and key in cache:
        return cache[key]
    lengths = (active[2] if len(active) > 2 else
               torch.bincount(indices // query.shape[2], minlength=query.shape[0]).to(torch.int32))
    if torch.any(lengths == 0).item():
        result = None
    else:
        cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        result = indices, cumulative
    if cache is not None:
        cache[key] = result
    return result


def _pack(tensor, indices):
    import torch

    batch, heads, tokens, width = tensor.shape
    if indices is None:
        return tensor.transpose(1, 2).to(torch.float16)
    return tensor.transpose(1, 2).reshape(batch*tokens, heads, width).index_select(
        0, indices).to(torch.float16).unsqueeze(0)


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    class _VarlenGQA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, indices, cumulative, scale, max_sequence):
            packed = [_pack(tensor, indices) for tensor in (query, key, value)]
            qpacked, kpacked, vpacked = packed
            batch, heads, tokens, width = query.shape
            groups, repeat = key.shape[1], heads // key.shape[1]
            allocate = torch.empty if indices is None else torch.zeros
            output = allocate((batch*tokens, heads, width), device=query.device,
                              dtype=torch.float16)
            pieces, statistics, maxima = [], [], []
            for group in range(groups):
                start, stop = group*repeat, (group+1)*repeat
                kg = kpacked[:, :, group:group+1].expand(-1, -1, repeat, -1)
                vg = vpacked[:, :, group:group+1].expand(-1, -1, repeat, -1)
                piece, lse, seed, offset, maxq, maxk = torch.ops.aten._efficient_attention_forward(
                    qpacked[:, :, start:stop], kg, vg, None, cumulative, cumulative,
                    max_sequence or tokens, max_sequence or tokens, 0.0, 1, True, scale=scale,
                )
                if indices is None:
                    output[:, start:stop].copy_(piece[0])
                else:
                    output[:, start:stop].index_copy_(0, indices, piece[0])
                pieces.append(piece)
                statistics.extend((lse, seed, offset))
                maxima.append((maxq, maxk))
            ctx.save_for_backward(*packed, indices, cumulative, *pieces, *statistics)
            ctx.input_types = [tensor.dtype for tensor in (query, key, value)]
            ctx.shape, ctx.groups, ctx.repeat = query.shape, groups, repeat
            ctx.dense_packed = indices is None
            ctx.maxima, ctx.scale = maxima, scale
            return output.view(batch, tokens, heads, width).transpose(1, 2)

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            qpacked, kpacked, vpacked, indices, cumulative, *saved = ctx.saved_tensors
            pieces, statistics = saved[:ctx.groups], saved[ctx.groups:]
            batch, heads, tokens, width = ctx.shape
            allocate = torch.empty if ctx.dense_packed else torch.zeros
            gradients = [allocate((batch*tokens, count, width), device=incoming.device,
                                  dtype=dtype) for count, dtype in
                         zip((heads, ctx.groups, ctx.groups), ctx.input_types, strict=True)]
            packed_incoming = _pack(incoming, indices)
            for group in range(ctx.groups):
                start, stop = group*ctx.repeat, (group+1)*ctx.repeat
                kg = kpacked[:, :, group:group+1].expand(-1, -1, ctx.repeat, -1)
                vg = vpacked[:, :, group:group+1].expand(-1, -1, ctx.repeat, -1)
                lse, seed, offset = statistics[group*3:group*3+3]
                maxq, maxk = ctx.maxima[group]
                gq, gk, gv, _ = torch.ops.aten._efficient_attention_backward(
                    packed_incoming[:, :, start:stop], qpacked[:, :, start:stop], kg, vg,
                    None, pieces[group], cumulative, cumulative, maxq, maxk, lse,
                    0.0, seed, offset, 1, False, scale=ctx.scale,
                )
                if gk.shape[2] != ctx.repeat or gv.shape[2] != ctx.repeat:
                    raise RuntimeError("varlen attention did not return per-head KV gradients")
                pieces_grad = (gq[0].to(ctx.input_types[0]),
                               gk[0].float().sum(1, keepdim=True).to(ctx.input_types[1]),
                               gv[0].float().sum(1, keepdim=True).to(ctx.input_types[2]))
                targets = (gradients[0][:, start:stop], gradients[1][:, group:group+1],
                           gradients[2][:, group:group+1])
                for target, gradient in zip(targets, pieces_grad, strict=True):
                    if indices is None:
                        target.copy_(gradient)
                    else:
                        target.index_copy_(0, indices, gradient)
            result = [gradient.view(batch, tokens, gradient.shape[1], width).transpose(1, 2)
                      for gradient in gradients]
            return *result, None, None, None, None

    return _VarlenGQA


def varlen_attention(query, key, value, indices, cumulative, scale, max_sequence=None):
    return _autograd().apply(query, key, value, indices, cumulative, scale, max_sequence)


def patch_qwen_varlen_attention():
    from . import gqa_native

    gqa_native._varlen_enabled = True
    return True
