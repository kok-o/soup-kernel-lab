"""Narrow integration for Qwen2 + TRL 0.29.1 + Soup 0.77.0.

No trainer settings change. HF's parent Trainer keeps loss normalization and
model dispatch; the fixed SFT metrics reuse CE's detached token statistics.
Rewind's sampler, FIFO and logging stay in Soup.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from .ce import CEMetrics, causal_ce

_CAPTURE = ContextVar("setww_ce_capture", default=None)
_STATS_ATTR = "_setww_ce_metrics"


@dataclass
class _Capture:
    stats: CEMetrics | None = None


def record_metrics(trainer, stats, attention_mask):
    """Match the fixed TRL train metrics; metrics do not enter autograd."""
    import torch

    with torch.no_grad():
        entropy = (stats.entropy * attention_mask).sum() / attention_mask.sum()
        entropy = trainer.accelerator.gather_for_metrics(entropy).mean().item()
        trainer._metrics["train"]["entropy"].append(entropy)
        tokens = trainer.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        trainer._total_train_tokens += tokens
        trainer._metrics["train"]["num_tokens"] = [trainer._total_train_tokens]
        correct = trainer.accelerator.gather_for_metrics(stats.correct.sum()).sum()
        total = trainer.accelerator.gather_for_metrics(stats.valid.sum()).sum()
        accuracy = (correct / total).item() if total > 0 else 0.0
        trainer._metrics["train"]["mean_token_accuracy"].append(accuracy)


def _eligible(trainer, model, inputs):
    import torch

    labels = inputs.get("labels")
    if not isinstance(model, torch.nn.Module) or not getattr(model, "training", False):
        return False
    if not isinstance(labels, torch.Tensor) or not labels.is_cuda or labels.ndim != 2:
        return False
    if "attention_mask" not in inputs or "shift_labels" in inputs:
        return False
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen2":
        return False
    args = trainer.args
    return (
        not getattr(args, "use_liger_kernel", False)
        and getattr(args, "loss_type", "nll") == "nll"
        and not getattr(trainer, "padding_free", False)
        and not getattr(trainer, "num_virtual_tokens", 0)
        and not getattr(trainer, "aux_loss_enabled", False)
        and getattr(trainer, "label_smoother", None) is None
        and getattr(trainer, "compute_loss_func", None) is None
        and getattr(trainer.accelerator, "num_processes", 1) == 1
    )


def _install_model_loss(model):
    """Install on the actual Qwen instance, not the global loss mapping."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(base, "_setww_ce_loss_installed", False):
        return
    original = base.loss_function

    @wraps(original)
    def loss_function(logits, labels, vocab_size, num_items_in_batch=None, **kwargs):
        capture = _CAPTURE.get()
        if (
            capture is None
            or not logits.is_cuda
            or kwargs.get("shift_labels") is not None
            or kwargs.get("ignore_index", -100) != -100
        ):
            return original(
                logits, labels, vocab_size, num_items_in_batch=num_items_in_batch, **kwargs
            )
        loss, capture.stats = causal_ce(logits, labels, num_items_in_batch)
        return loss

    base.loss_function = loss_function
    base._setww_ce_loss_installed = True


def patch_sft_ce():
    """Keep unsupported tasks/eval on their existing path; safe to call twice."""
    from soup_cli.trainer import rewind_hf
    from trl import SFTTrainer

    if getattr(SFTTrainer.compute_loss, "_setww_ce_installed", False):
        return
    original_compute = SFTTrainer.compute_loss
    original_rows = rewind_hf.row_losses

    @wraps(original_compute)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _eligible(self, model, inputs):
            return original_compute(
                self, model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        _install_model_loss(model)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        capture = _Capture()
        token = _CAPTURE.set(capture)
        try:
            loss, outputs = super(SFTTrainer, self).compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        finally:
            _CAPTURE.reset(token)
        if capture.stats is None:
            raise RuntimeError("R1: Qwen did not call its installed causal loss")
        # Accelerate can replace logits with an FP32 tensor on return. Attach
        # AFTER that conversion, so rewind gets the right per-call statistics.
        setattr(outputs.logits, _STATS_ATTR, capture.stats)
        record_metrics(self, capture.stats, inputs["attention_mask"])
        return (loss, outputs) if return_outputs else loss

    @wraps(original_rows)
    def row_losses(logits, labels):
        stats = getattr(logits, _STATS_ATTR, None)
        if isinstance(stats, CEMetrics) and stats.token_loss.shape == labels.shape:
            return stats.row_losses()
        return original_rows(logits, labels)

    compute_loss._setww_ce_installed = True
    SFTTrainer.compute_loss = compute_loss
    rewind_hf.row_losses = row_losses
