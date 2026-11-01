"""Causal CE and reusable token statistics; CUDA uses original Triton kernels.

The public interface also supplies an ordinary PyTorch CPU path. Metrics are
detached: only CE participates in training, just as in Soup/TRL.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass
class CEMetrics:
    """All fields have shape (batch, sequence), including the final position.

    With causal_ce, token_loss[t] predicts labels[t+1]; the final position has
    zero loss/correct/valid. With aligned_ce all supplied targets are already
    aligned. Ignored targets always have zero loss/correct/valid; entropy
    includes ALL positions. The consumer applies attention_mask for entropy.
    """

    token_loss: Any
    entropy: Any
    correct: Any
    valid: Any

    def row_losses(self):
        """Return Soup rewind's Python means and supervised-token counts."""
        import torch

        with torch.no_grad():
            counts = self.valid.sum(dim=1)
            means = self.token_loss.float().sum(dim=1) / counts.clamp(min=1)
        return means.tolist(), counts.tolist()


def _validate(logits, labels):
    import torch

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits (B, T, V) and labels (B, T)")
    if min(logits.shape) < 1:
        raise ValueError("batch, sequence and vocabulary must be nonempty")
    if logits.dtype not in (torch.float16, torch.float32):
        raise TypeError("R1 supports FP16 and FP32 logits")
    if labels.dtype != torch.int64 or labels.device != logits.device:
        raise ValueError("labels must be int64 on the logits device")


def _shifted_labels(labels, ignore_index):
    import torch.nn.functional as functional

    return functional.pad(labels[:, 1:], (0, 1), value=ignore_index)


def _cpu_tokens(logits, labels, ignore_index, label_shift):
    import torch
    import torch.nn.functional as functional

    targets = _shifted_labels(labels, ignore_index) if label_shift else labels
    token_loss = functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)
    with torch.no_grad():
        log_probs = logits.detach().float().log_softmax(dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        correct = (logits.detach().argmax(dim=-1) == targets) & (targets != ignore_index)
    return token_loss, entropy, correct


@lru_cache(maxsize=1)
def _cuda_autograd():
    import torch
    import triton

    from ._ce_kernels import ce_backward_kernel, ce_merge_kernel, ce_partition_kernel

    class TiledCausalCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, labels, ignore_index, label_shift, consume_logits):
            batch, sequence, vocab = logits.shape
            rows = batch * sequence
            block = 4096
            partitions = triton.cdiv(vocab, block)
            options = {"device": logits.device, "dtype": torch.float32}
            maxima = torch.empty((rows, partitions), **options)
            totals = torch.empty_like(maxima)
            weighted = torch.empty_like(maxima)
            indices = torch.empty((rows, partitions), device=logits.device, dtype=torch.int32)
            lse = torch.empty((batch, sequence), **options)
            loss = torch.empty_like(lse)
            entropy = torch.empty_like(lse)
            correct = torch.empty((batch, sequence), device=logits.device, dtype=torch.bool)
            strides = logits.stride()
            label_strides = labels.stride()
            with torch.cuda.device(logits.device):
                ce_partition_kernel[(rows, partitions)](
                    logits, maxima, totals, weighted, indices,
                    sequence, vocab, *strides, partitions, block_size=block, num_warps=4,
                )
                ce_merge_kernel[(rows,)](
                    logits, labels, maxima, totals, weighted, indices, lse, loss, entropy, correct,
                    sequence, vocab, *strides, *label_strides,
                    partitions, ignore_index, label_shift,
                    merge_block=triton.next_power_of_2(partitions), num_warps=4,
                )
            ctx.save_for_backward(logits, labels, lse)
            ctx.ignore_index = ignore_index
            ctx.label_shift = label_shift
            ctx.block = block
            ctx.consume_logits, ctx.consumed = consume_logits, False
            ctx.mark_non_differentiable(entropy, correct)
            ctx.set_materialize_grads(False)
            return loss, entropy, correct

        @staticmethod
        def backward(ctx, grad_loss, _grad_entropy, _grad_correct):
            if grad_loss is None:
                return None, None, None, None, None
            if ctx.consumed:
                raise RuntimeError("owned CE logits support one backward only")
            logits, labels, lse = ctx.saved_tensors
            batch, sequence, vocab = logits.shape
            gradient = (logits.detach() if ctx.consume_logits else
                        torch.empty(logits.shape, device=logits.device, dtype=logits.dtype))
            with torch.cuda.device(logits.device):
                ce_backward_kernel[(batch * sequence, triton.cdiv(vocab, ctx.block))](
                    logits, labels, lse, grad_loss, gradient,
                    sequence, vocab, *logits.stride(), *labels.stride(), *grad_loss.stride(),
                    ctx.ignore_index, ctx.label_shift, block_size=ctx.block, num_warps=4,
                )
            if ctx.consume_logits:
                ctx.consumed = True
                torch.autograd.graph.increment_version(logits)
            return gradient, None, None, None, None

    return TiledCausalCE


def causal_ce(logits, labels, num_items_in_batch=None, ignore_index=-100):
    """Return (scalar_loss, CEMetrics), matching ForCausalLMLoss normalization.

    num_items_in_batch=None means mean over supervised shifted labels. An
    explicit count means sum/count (it need not equal this microbatch's count).
    Inputs are finite logits with labels in [0, V) or ignore_index. An all-ignored
    mean is NaN, like PyTorch; its ignored logits still have zero gradient.
    """
    return _token_ce(logits, labels, num_items_in_batch, ignore_index, label_shift=1)


def aligned_ce(logits, targets, num_items_in_batch=None, ignore_index=-100):
    """Same [B,T,V] interface with already-aligned [B,T] targets, including the last.

    Intended for flattened token chunks after the original sequences have been
    shifted once. Chunk boundaries do not drop targets or create dummy logits.
    CEMetrics.valid therefore includes the final position when supervised.
    """
    return _token_ce(logits, targets, num_items_in_batch, ignore_index, label_shift=0)


def _token_ce(
    logits, labels, num_items_in_batch, ignore_index, label_shift, consume_logits=False,
):
    import torch

    _validate(logits, labels)
    if logits.is_cuda:
        token_loss, entropy, correct = _cuda_autograd().apply(
            logits, labels, ignore_index, label_shift, consume_logits
        )
    elif logits.device.type == "cpu":
        token_loss, entropy, correct = _cpu_tokens(logits, labels, ignore_index, label_shift)
    else:
        raise ValueError("R1 supports CUDA and CPU")
    targets = _shifted_labels(labels, ignore_index) if label_shift else labels
    valid = targets != ignore_index
    if num_items_in_batch is None:
        denominator = valid.sum()
    else:
        denominator = torch.as_tensor(num_items_in_batch, device=logits.device)
        if denominator.numel() != 1:
            raise ValueError("num_items_in_batch must be a scalar")
    loss = token_loss.sum() / denominator
    stats = CEMetrics(token_loss.detach(), entropy.detach(), correct.detach(), valid)
    return loss, stats


def owned_aligned_ce(logits, targets, num_items_in_batch=None, ignore_index=-100):
    """Consume a temporary contiguous logits buffer during its single backward.

    Private frozen-head contract: logits have no other consumer after metrics
    were detached. Generic aligned_ce/causal_ce remain non-mutating and reusable.
    """
    if logits.is_cuda and not logits.is_contiguous():
        raise ValueError("owned CE requires contiguous temporary logits")
    return _token_ce(logits, targets, num_items_in_batch, ignore_index,
                     label_shift=0, consume_logits=True)
