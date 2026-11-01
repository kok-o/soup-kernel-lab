"""Recompute only Qwen attention, preserving RNG and exact checkpoint inputs.

Uses PyTorch's non-reentrant checkpoint utility. No Soup training flag changes.
Attention's saved 4D tensors are recreated in backward with their own layout.
"""

from functools import wraps


def patch_attention_recompute():
    import torch
    from torch.utils.checkpoint import checkpoint

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    except (ImportError, AttributeError):
        return False
    original = Qwen2Attention.forward
    if getattr(original, "_setww_attention_recompute", False):
        return True

    @wraps(original)
    def forward(self, *args, **kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if (hidden is None or not self.training or not hidden.is_cuda
                or not torch.is_grad_enabled() or not hidden.requires_grad
                or kwargs.get("past_key_values") is not None
                or kwargs.get("output_attentions", False)):
            return original(self, *args, **kwargs)

        def invoke(source):
            if args:
                return original(self, source, *args[1:], **kwargs)
            return original(self, **dict(kwargs, hidden_states=source))

        # Only the checkpoint boundary uses exact storage. Its inner forward
        # saves holders instead of Q/K/V/dropout buffers; outer decoder hooks
        # continue compressing the rest of the model as before.
        with torch.autograd.graph.saved_tensors_hooks(lambda value: value, lambda value: value):
            return checkpoint(invoke, hidden, use_reentrant=False, preserve_rng_state=True)

    forward._setww_attention_recompute = True
    Qwen2Attention.forward = forward
    return True
