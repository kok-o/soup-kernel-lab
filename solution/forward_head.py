"""Precompute frozen-head dHidden with the actual AMP scale during forward.

No logits survive forward. Unexpected upstream scaling uses exact recomputation.
The scale is observed from the existing GradScaler; optimizer/settings are unchanged.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable
    from torch.nn import functional

    class _ForwardHead(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, target, mask, denominator, chunk,
                    compact, backend, expected_scale, linear):
            ctx.linear = linear or functional.linear
            kind = hidden.device.type
            ctx.autocast_enabled = torch.is_autocast_enabled(kind)
            ctx.autocast_dtype = torch.get_autocast_dtype(kind)
            compute_weight = weight.to(ctx.autocast_dtype) if ctx.autocast_enabled else weight
            flat = hidden.reshape(-1, hidden.shape[-1])
            targets = target.reshape(-1)
            supervised = targets.ne(-100)
            visible = mask.reshape(-1).bool() if compact else torch.ones_like(supervised)
            groups = (supervised.nonzero().flatten(), (visible & ~supervised).nonzero().flatten())
            wide_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
            token_loss = torch.zeros(targets.numel(), dtype=wide_dtype, device=hidden.device)
            entropy = torch.zeros_like(token_loss)
            total = token_loss.new_zeros(())
            correct = target.new_zeros(())
            hidden_gradient = torch.zeros_like(flat)
            scale = token_loss.new_tensor(expected_scale)
            for group_index, group in enumerate(groups):
                for start in range(0, group.numel(), chunk):
                    ids = group[start:start + chunk]
                    piece = flat.index_select(0, ids).detach()
                    needs_gradient = group_index == 0 and ctx.needs_input_grad[0]
                    piece.requires_grad_(needs_gradient)
                    with torch.set_grad_enabled(needs_gradient), torch.autocast(
                        kind, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype
                    ):
                        logits = ctx.linear(piece, compute_weight)
                        terms = backend(logits, targets.index_select(0, ids), need_metrics=True)
                        total = total + terms.loss_sum.detach()
                        token_loss.index_copy_(0, ids, terms.token_loss.detach().to(wide_dtype))
                        entropy.index_copy_(0, ids, terms.token_entropy.detach().to(wide_dtype))
                        correct = correct + terms.correct.sum()
                        if needs_gradient:
                            gradient, = torch.autograd.grad(
                                terms.loss_sum, piece,
                                grad_outputs=(scale / denominator).to(terms.loss_sum.dtype),
                            )
                            hidden_gradient.index_copy_(0, ids, gradient)
                            del gradient
                    del logits, terms, piece
            ctx.expected_scale = expected_scale
            ctx.chunk, ctx.backend = chunk, backend
            ctx.save_for_backward(hidden, weight, compute_weight, targets, groups[0],
                                  denominator, hidden_gradient)
            token_loss = token_loss.view(hidden.shape[:2])
            entropy = entropy.view(hidden.shape[:2])
            ctx.mark_non_differentiable(token_loss, entropy, correct)
            return total / denominator, token_loss, entropy, correct

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_loss, _token_grad, _entropy_grad, _correct_grad):
            hidden, _canonical, weight, targets, indices, denominator, cached = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return (None,) * 10
            if torch.equal(grad_loss, grad_loss.new_tensor(ctx.expected_scale)):
                return (cached.view_as(hidden),) + (None,) * 9
            # Custom loss composition/accumulation may send a different scale.
            # Never rescale a gradient that was already rounded for AMP.
            flat = hidden.reshape(-1, hidden.shape[-1])
            gradient = torch.zeros_like(flat)
            for start in range(0, indices.numel(), ctx.chunk):
                ids = indices[start:start + ctx.chunk]
                piece = flat.index_select(0, ids).detach().requires_grad_(True)
                with torch.enable_grad(), torch.autocast(
                    hidden.device.type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype
                ):
                    logits = ctx.linear(piece, weight)
                    terms = ctx.backend(logits, targets.index_select(0, ids), need_metrics=False)
                    piece_gradient, = torch.autograd.grad(
                        terms.loss_sum, piece,
                        grad_outputs=(grad_loss / denominator).to(terms.loss_sum.dtype),
                    )
                gradient.index_copy_(0, ids, piece_gradient)
            return (gradient.view_as(hidden),) + (None,) * 9

    return _ForwardHead


def forward_head_terms(hidden, weight, target, mask, denominator, chunk,
                       compact, backend, expected_scale, linear):
    return _autograd().apply(hidden, weight, target, mask, denominator, chunk,
                             compact, backend, expected_scale, linear)
