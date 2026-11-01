"""Native padded dropout RNG; save compact mask bits, never padded scatter inputs."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    from .dropout_bits import masked_gradient, pack_mask

    class _CompactDropout(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source, indices, batch, tokens, probability):
            width = source.shape[-1]
            # Keep exactly the native padded input shape, dtype, layout and RNG.
            # Inside this Function the reconstruction has no inner autograd graph.
            padded = source.new_zeros((batch*tokens, width))
            padded.index_copy_(0, indices, source.reshape(-1, width))
            output, mask = torch.ops.aten.native_dropout(
                padded.view(batch, tokens, width), probability, True)
            compact = output.reshape(-1, width).index_select(0, indices).view_as(source)
            if ctx.needs_input_grad[0]:
                valid_mask = mask.reshape(-1, width).index_select(0, indices)
                ctx.save_for_backward(pack_mask(valid_mask))
            ctx.scale = 1.0/(1.0-probability)
            return compact

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            packed, = ctx.saved_tensors
            return masked_gradient(incoming, packed, ctx.scale), None, None, None, None

    return _CompactDropout


def compact_native_dropout(source, indices, padded_shape, probability):
    return _autograd().apply(source, indices, *padded_shape, probability)
