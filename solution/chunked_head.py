"""R2: bounded-memory frozen output head with scaled backward recomputation.

Original implementation for issue #2. Architecture references, not copied code:
Soup 0.77.0 stream_loss_chunk.py; Liger paper, arXiv:2410.10989.
This module uses reference PyTorch CE until R1 registers its original Triton CE.
It is not itself a claim of author Triton time or of measured T4 improvement.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ChunkTerms:
    """R1 boundary: differentiable SUM loss; detached position-wise metrics.

    All inputs are [positions, vocab] logits and [positions] shifted targets.
    need_metrics=False may leave the three optional fields unset. Do not divide
    loss_sum here: R2 applies the single global denominator in both passes.
    """

    loss_sum: torch.Tensor
    token_loss: torch.Tensor | None = None
    token_entropy: torch.Tensor | None = None
    correct: torch.Tensor | None = None


@dataclass(frozen=True)
class ChunkedStep:
    token_loss: torch.Tensor       # [batch, seq], loss of labels[t+1]; last is 0
    token_entropy: torch.Tensor    # [batch, seq]; includes prompt/last valid token
    correct: torch.Tensor          # scalar count on supervised shifted positions
    labelled: torch.Tensor         # scalar count on supervised shifted positions


def reference_terms(logits, target, *, need_metrics):
    """FP32 reference matching the task's CE and Accelerate-upcast metrics.

    Float64 is retained solely for numerical tests/gradcheck. On the fixed task
    logits are fp16 and the CE/metrics reduction is fp32.
    """
    import torch
    from torch.nn import functional

    wide = logits if logits.dtype == torch.float64 else logits.float()
    if not need_metrics:
        return ChunkTerms(functional.cross_entropy(
            wide, target, ignore_index=-100, reduction="sum"
        ))
    token_loss = functional.cross_entropy(wide, target, ignore_index=-100, reduction="none")
    with torch.no_grad():
        logp = wide.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        correct = wide.argmax(-1).eq(target) & target.ne(-100)
    return ChunkTerms(token_loss.sum(), token_loss.detach(), entropy, correct)


_ce_backend: Callable = reference_terms
_active_head: ContextVar = ContextVar("setww_r2_active_head", default=None)
_chunk_size = 128
_compact = False
_forward_gradient = False
_linear_backend: Callable | None = None


def register_linear_backend(backend: Callable | None) -> None:
    """Register a frozen linear implementation; None restores reference functional.linear."""
    if backend is not None and not callable(backend):
        raise TypeError("linear backend must be callable or None")
    global _linear_backend
    _linear_backend = backend


def register_ce_backend(backend: Callable) -> None:
    """Connect R1's CE via the ChunkTerms contract; captured per forward.

    The backward loss_sum must support autograd, accept the actual incoming
    scale and round its logits gradient as the original AMP path does.
    """
    if not callable(backend):
        raise TypeError("CE backend must be callable")
    global _ce_backend
    _ce_backend = backend


def adapt_causal_ce(causal_ce: Callable) -> Callable:
    """Adapt R1 00d00d6's causal [B,T,V] API to arbitrary shifted positions.

    R1 ignores the final position of every input sequence. Encode targets for
    the first n-1 positions in synthetic labels; handle exactly the final real
    position with reference CE. No dummy logits or duplicated kernel rows.
    R1 supplies entropy for every real position. A future pointwise R1 entry
    can replace this adapter, eliminating its one-row reference tail.
    """
    def terms(logits, target, *, need_metrics):
        import torch
        from torch.nn import functional

        if logits.dtype not in (torch.float16, torch.float32):
            return reference_terms(logits, target, need_metrics=need_metrics)
        causal_labels = functional.pad(target[:-1], (1, 0), value=-100).unsqueeze(0)
        loss, stats = causal_ce(logits.unsqueeze(0), causal_labels, num_items_in_batch=1)
        tail = functional.cross_entropy(logits[-1:].float(), target[-1:], ignore_index=-100,
                               reduction="sum")
        if not need_metrics:
            return ChunkTerms(loss + tail)
        token_loss = stats.token_loss[0].clone()
        token_loss[-1] = tail.detach()
        correct = stats.correct[0].clone()
        correct[-1] = logits[-1].detach().argmax().eq(target[-1]) & target[-1].ne(-100)
        return ChunkTerms(loss + tail, token_loss, stats.entropy[0], correct)
    return terms


def adapt_aligned_ce(aligned_ce: Callable) -> Callable:
    """Use every real chunk position through R1's already-aligned CE entry."""
    def terms(logits, target, *, need_metrics):
        import torch

        if logits.dtype not in (torch.float16, torch.float32):
            return reference_terms(logits, target, need_metrics=need_metrics)
        loss, stats = aligned_ce(
            logits.unsqueeze(0), target.unsqueeze(0), num_items_in_batch=1
        )
        if not need_metrics:
            return ChunkTerms(loss)
        return ChunkTerms(loss, stats.token_loss[0], stats.entropy[0], stats.correct[0])
    return terms


@lru_cache(maxsize=1)
def _chunked_autograd():
    import torch
    from torch.autograd.function import once_differentiable
    from torch.nn import functional

    class _ChunkedHead(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, target, mask, denominator, chunk, compact, backend):
            ctx.linear = _linear_backend or functional.linear
            device_type = hidden.device.type
            ctx.autocast_enabled = torch.is_autocast_enabled(device_type)
            ctx.autocast_dtype = torch.get_autocast_dtype(device_type)
            # Frozen fp32 weights do NOT use Torch's normal autocast weight cache.
            # Materialize one compute copy for this forward/backward, not per chunk.
            compute_weight = weight.to(ctx.autocast_dtype) if ctx.autocast_enabled else weight
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            target = target.reshape(-1)
            supervised = target.ne(-100)
            indices = ((mask.reshape(-1).bool() | supervised).nonzero().flatten()
                       if compact else torch.arange(target.numel(), device=hidden.device))
            metric_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
            token_loss = torch.zeros(target.numel(), dtype=metric_dtype, device=hidden.device)
            entropy = torch.zeros_like(token_loss)
            total = token_loss.new_zeros(())
            correct = target.new_zeros(())
            for start in range(0, indices.numel(), chunk):
                ids = indices[start:start + chunk]
                logits = ctx.linear(flat_hidden.index_select(0, ids), compute_weight)
                terms = backend(logits, target.index_select(0, ids), need_metrics=True)
                metric_values = (terms.token_loss, terms.token_entropy, terms.correct)
                if any(value is None for value in metric_values):
                    raise ValueError("CE backend must return all per-token metrics in forward")
                total = total + terms.loss_sum.detach()
                token_loss.index_copy_(0, ids, terms.token_loss.detach().to(metric_dtype))
                entropy.index_copy_(0, ids, terms.token_entropy.detach().to(metric_dtype))
                correct = correct + terms.correct.sum()
                del logits, terms
            backward_indices = supervised.nonzero().flatten()
            # Save both the canonical weight (version check) and the one compute
            # copy. No [batch*seq, vocab] logits or their graph survives forward.
            ctx.save_for_backward(
                hidden, weight, compute_weight, target, backward_indices, denominator
            )
            ctx.chunk, ctx.backend = chunk, backend
            token_loss = token_loss.view(hidden.shape[:2])
            entropy = entropy.view(hidden.shape[:2])
            ctx.mark_non_differentiable(token_loss, entropy, correct)
            return total / denominator, token_loss, entropy, correct

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_loss, _token_grad, _entropy_grad, _correct_grad):
            hidden, _canonical_weight, weight, target, indices, denominator = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return (None,) * 8
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            hidden_grad = torch.zeros_like(flat_hidden)
            for start in range(0, indices.numel(), ctx.chunk):
                ids = indices[start:start + ctx.chunk]
                piece = flat_hidden.index_select(0, ids).detach().requires_grad_(True)
                # Backward normally runs outside forward's autocast context.
                # Re-enter that exact policy and apply incoming loss scale BEFORE
                # the CE derivative is cast to fp16 for linear backward.
                with torch.enable_grad(), torch.autocast(
                    hidden.device.type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype
                ):
                    logits = ctx.linear(piece, weight)
                    terms = ctx.backend(logits, target.index_select(0, ids), need_metrics=False)
                    gradient, = torch.autograd.grad(
                        terms.loss_sum, piece,
                        grad_outputs=(grad_loss / denominator).to(terms.loss_sum.dtype),
                    )
                hidden_grad.index_copy_(0, ids, gradient)
                del logits, terms, piece, gradient
            return hidden_grad.view_as(hidden), None, None, None, None, None, None, None

    return _ChunkedHead


def chunked_head_loss(
    hidden, weight, labels, *, attention_mask=None, shift_labels=None,
    num_items_in_batch=None, chunk_size=128, compact=False, backend=None, precompute_scale=None,
):
    """Frozen bias-free linear head, first-order causal loss and exact masks.

    Chunk counts flattened positions, not tokens PER row. compact optionally
    skips padding in forward; backward always visits only supervised positions.
    All-ignored mean loss follows PyTorch's NaN convention; an explicit positive
    denominator gives zero loss and zero gradient for such a microbatch.
    """
    import torch
    from torch.nn import functional

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if weight.requires_grad:
        raise ValueError("R2 requires a frozen head weight")
    if hidden.ndim != 3 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("expected hidden [batch, seq, hidden] and weight [vocab, hidden]")
    target = (shift_labels if shift_labels is not None else
              functional.pad(labels, (0, 1), value=-100)[..., 1:])
    target = target.to(hidden.device)
    if target.shape != hidden.shape[:2]:
        raise ValueError("shifted labels must match hidden batch/sequence dimensions")
    if attention_mask is None:
        attention_mask = torch.ones_like(target)
    if attention_mask.shape != target.shape:
        raise ValueError("attention mask must match labels")
    labelled = target.ne(-100).sum()
    denominator = labelled if num_items_in_batch is None else num_items_in_batch
    metric_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    denominator = torch.as_tensor(denominator, device=hidden.device, dtype=metric_dtype)
    if denominator.numel() != 1:
        raise ValueError("loss denominator must be scalar")
    denominator = denominator.reshape(())
    if precompute_scale is None:
        loss, token_loss, entropy, correct = _chunked_autograd().apply(
            hidden, weight, target, attention_mask.to(hidden.device), denominator,
            chunk_size, compact, backend or _ce_backend,
        )
    else:
        from .forward_head import forward_head_terms

        loss, token_loss, entropy, correct = forward_head_terms(
            hidden, weight, target, attention_mask.to(hidden.device), denominator,
            chunk_size, compact, backend or _ce_backend, precompute_scale, _linear_backend,
        )
    return loss, ChunkedStep(token_loss, entropy, correct, labelled)


def _base_model(model):
    unwrap = getattr(model, "get_base_model", None)
    return unwrap() if callable(unwrap) else model


def _log_metrics(trainer, inputs, step):
    """Same reductions and metric names as TRL 0.29.1, without full logits."""
    gather = trainer.accelerator.gather_for_metrics
    mask = inputs["attention_mask"]
    entropy = (step.token_entropy * mask).sum() / mask.sum()
    trainer._metrics["train"]["entropy"].append(gather(entropy).mean().item())
    trainer._total_train_tokens += gather(mask.sum()).sum().item()
    trainer._metrics["train"]["num_tokens"] = [trainer._total_train_tokens]
    count = gather(step.labelled).sum()
    hits = gather(step.correct).sum()
    accuracy = (hits / count).item() if count > 0 else 0.0
    trainer._metrics["train"]["mean_token_accuracy"].append(accuracy)


def patch_chunked_head(chunk_size=128, *, compact=False, forward_gradient=False) -> bool:
    """Idempotent, task-specific model/trainer patch; returns False for fakes.

    Activation is scoped to eligible SFTTrainer training compute_loss calls.
    Generation, evaluation, trainable heads and other tasks keep their forwards.
    No Soup built-in switch or trainer/model parameter is changed.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    import torch

    try:
        from transformers import Trainer
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        from trl import SFTTrainer
    except (ImportError, AttributeError):
        return False
    if not callable(getattr(SFTTrainer, "compute_loss", None)):
        return False
    global _chunk_size, _compact, _forward_gradient
    _chunk_size, _compact, _forward_gradient = chunk_size, compact, forward_gradient

    def eligible(model):
        head = getattr(model, "lm_head", None)
        return (isinstance(model, Qwen2ForCausalLM) and model.training
                and isinstance(head, torch.nn.Linear) and head.bias is None
                and not head.weight.requires_grad)

    if not getattr(Qwen2ForCausalLM.forward, "_setww_r2", False):
        original_forward = Qwen2ForCausalLM.forward

        @wraps(original_forward)
        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None,
                    use_cache=None, logits_to_keep=0, **kwargs):
            active = _active_head.get()
            if active is None or active[0] is not self or labels is None:
                return original_forward(
                    self, input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                    logits_to_keep=logits_to_keep, **kwargs,
                )
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("R2 training expects logits_to_keep=0")
            decoded = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, **kwargs,
            )
            loss, step = chunked_head_loss(
                decoded.last_hidden_state, self.lm_head.weight, labels,
                attention_mask=attention_mask, shift_labels=kwargs.get("shift_labels"),
                num_items_in_batch=kwargs.get("num_items_in_batch"),
                chunk_size=active[1], compact=active[2],
                precompute_scale=active[3] if len(active) > 3 else None,
            )
            self._setww_r2_step = step
            return CausalLMOutputWithPast(
                loss=loss, logits=None, past_key_values=decoded.past_key_values,
                hidden_states=decoded.hidden_states, attentions=decoded.attentions,
            )

        forward._setww_r2 = True
        Qwen2ForCausalLM.forward = forward

    if not getattr(SFTTrainer.compute_loss, "_setww_r2", False):
        original_compute_loss = SFTTrainer.compute_loss

        @wraps(original_compute_loss)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            base = _base_model(model)
            args = self.args
            supported = (
                eligible(base) and "labels" in inputs and "attention_mask" in inputs
                and getattr(args, "loss_type", "nll") == "nll"
                and not getattr(args, "use_liger_kernel", False)
                and not getattr(self, "padding_free", False)
                and not getattr(self, "num_virtual_tokens", 0)
                and not getattr(self, "aux_loss_enabled", False)
                and getattr(self.accelerator, "num_processes", 1) == 1
                and getattr(self, "label_smoother", None) is None
                and getattr(self, "compute_loss_func", None) is None
            )
            if not supported:
                self._soup_chunked_step = None
                return original_compute_loss(
                    self, model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            from .frozen_storage import compact_frozen_tied_embedding

            compact_frozen_tied_embedding(base)
            inputs.pop("_prediction_loss_only", None)
            inputs["use_cache"] = False
            base._setww_r2_step = None
            scale = None
            scaler = getattr(self.accelerator, "scaler", None)
            if (_forward_gradient and scaler is not None
                    and getattr(args, "gradient_accumulation_steps", 1) == 1):
                scale = scaler.get_scale()
            token = _active_head.set((base, _chunk_size, _compact, scale))
            try:
                loss, outputs = Trainer.compute_loss(
                    self, model, inputs, return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                _active_head.reset(token)
            step, base._setww_r2_step = base._setww_r2_step, None
            if step is None:
                raise RuntimeError("R2 training forward did not produce token metrics")
            # Soup's rewind recorder consumes this established side channel.
            # This does not enable its built-in chunked-loss implementation.
            self._soup_chunked_step = step
            _log_metrics(self, inputs, step)
            return (loss, outputs) if return_outputs else loss

        compute_loss._setww_r2 = True
        SFTTrainer.compute_loss = compute_loss
    return True
