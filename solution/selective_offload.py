"""Keep later decoder layers resident and offload early compressed packets.

The existing ordered pinned transfer is unchanged. Earlier layers are consumed
last by backward, so their storage can leave the GPU for most of the step.
The count is an implementation choice, independent of training/data settings.
"""

from contextvars import ContextVar
from functools import wraps

_layer_index = ContextVar("setww_offload_layer", default=None)
_offload_layers = 16


def selective_packed_hooks(pack, unpack, *, offload_raw=False):
    from .activation_offload import offload_packed_hooks

    def selected():
        index = _layer_index.get()
        return index is not None and 0 <= index < _offload_layers

    return offload_packed_hooks(
        pack, unpack, should_offload=selected,
        layer_getter=_layer_index.get, offload_raw=offload_raw,
    )


def patch_selective_offload(layers=16):
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from .activation_offload import clear_staging, prefetch_layer

    if isinstance(layers, bool) or not isinstance(layers, int) or not 0 <= layers <= 28:
        raise ValueError("offloaded decoder layers must be between 0 and 28")
    global _offload_layers
    _offload_layers = layers
    original = Qwen2DecoderLayer.forward
    if getattr(original, "_setww_selective_offload", False):
        return True

    @wraps(original)
    def forward(self, *args, **kwargs):
        index = getattr(self.self_attn, "layer_idx", None)
        if index == 0 and self.training and torch.is_grad_enabled():
            inp = args[0] if args else kwargs.get("hidden_states")
            if isinstance(inp, torch.Tensor) and inp.requires_grad:
                inp.register_hook(lambda _grad: clear_staging())
        token = _layer_index.set(index)
        try:
            outputs = original(self, *args, **kwargs)
        finally:
            _layer_index.reset(token)
        if (self.training and torch.is_grad_enabled() and index is not None
                and 1 <= index <= _offload_layers):
            target = index - 1
            if 0 <= target < _offload_layers:
                out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                if isinstance(out_tensor, torch.Tensor) and out_tensor.requires_grad:
                    out_tensor.register_hook(lambda _grad, t=target: prefetch_layer(t))
        return outputs

    forward._setww_selective_offload = True
    Qwen2DecoderLayer.forward = forward
    return True
