"""Skip only masked token-wise MLP work, preserving dense attention and dropout.

Original transformation; reference architecture is Qwen2 in transformers 5.16.1.
The fixed task has no MLP LoRA or dropout. No token is moved between batches.
"""

from contextvars import ContextVar
from functools import wraps

_active_tokens = ContextVar("setww_active_mlp_tokens", default=None)


def compact_mlp(original, module, hidden, indices):
    import torch

    flat = hidden.reshape(-1, hidden.shape[-1])
    selected = original(module, flat.index_select(0, indices))
    # Padding outputs are never observed by loss/entropy/real attention queries.
    # Preserve the full shape, dtype and layout before the next residual/SDPA.
    output = torch.zeros(flat.shape[0], selected.shape[-1], device=selected.device,
                         dtype=selected.dtype)
    return output.index_copy_(0, indices, selected).view(*hidden.shape[:-1], selected.shape[-1])


def patch_tokenwise_mlp():
    import torch

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2MLP
    except (ImportError, AttributeError):
        return False
    original_mlp = Qwen2MLP.forward
    if not getattr(original_mlp, "_setww_unpadded_mlp", False):
        @wraps(original_mlp)
        def mlp_forward(self, hidden):
            active = _active_tokens.get()
            modules = (self.gate_proj, self.up_proj, self.down_proj)
            if (active is None or hidden.ndim != 3 or hidden.shape[:2] != active[1]
                    or any(getattr(layer, "lora_A", None) is not None for layer in modules)
                    or any(layer.weight.requires_grad for layer in modules)):
                return original_mlp(self, hidden)
            return compact_mlp(original_mlp, self, hidden, active[0])

        mlp_forward._setww_unpadded_mlp = True
        Qwen2MLP.forward = mlp_forward
    original_model = Qwen2ForCausalLM.forward
    if getattr(original_model, "_setww_unpadded_mlp", False):
        return True

    @wraps(original_model)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        mask = kwargs.get("attention_mask")
        supported = (self.training and torch.is_grad_enabled() and labels is not None
                     and mask is not None and mask.is_cuda and mask.ndim == 2
                     and labels.shape == mask.shape and not kwargs.get("past_key_values")
                     and not kwargs.get("output_hidden_states")
                     and not kwargs.get("output_attentions"))
        if not supported:
            return original_model(self, *args, **kwargs)
        valid = mask.bool()
        shift = kwargs.get("shift_labels")
        if shift is None:
            shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
        # Do not skip a masked position if a caller nevertheless supervises it.
        if torch.any(shift.ne(-100) & ~valid).item():
            return original_model(self, *args, **kwargs)
        ids = valid.reshape(-1).nonzero().flatten()
        if ids.numel() == 0 or ids.numel() == valid.numel():
            return original_model(self, *args, **kwargs)
        token = _active_tokens.set((ids, mask.shape, valid.sum(1, dtype=torch.int32)))
        try:
            return original_model(self, *args, **kwargs)
        finally:
            _active_tokens.reset(token)

    forward._setww_unpadded_mlp = True
    Qwen2ForCausalLM.forward = forward
    return True
