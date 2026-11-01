"""Keep real tokens compact across the native frozen Qwen2 decoder.

The native model loop, decoder math, Q/K/V/O and LoRA modules are reused.
LoRA dropout reconstructs the original padded shape before the original dropout
call, then gathers valid outputs. Thus both masks and Philox consumption match.
Rope positions and four independent causal sequences keep their original order.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

_current = ContextVar("setww_compact_decoder", default=None)
_enabled = False


@dataclass
class _Layout:
    shape: tuple
    indices: Any
    cumulative: Any
    packed: bool = False
    positions: dict = field(default_factory=dict)
    attention_metadata: dict = field(default_factory=dict)

    @property
    def tokens(self):
        return self.indices.numel()

    def rotary(self, factors):
        cosine, sine = factors
        key = (id(cosine), id(sine), cosine._version, sine._version)
        if key not in self.positions:
            batch, tokens = self.shape
            self.positions[key] = tuple(factor.expand(batch, -1, -1).reshape(
                batch*tokens, factor.shape[-1]).index_select(0, self.indices).unsqueeze(0)
                for factor in factors)
        return self.positions[key]


def patch_compact_decoder():
    import torch
    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2 import modeling_qwen2 as model

    from .compact_dropout import compact_native_dropout
    from .grouped_ragged_attention import grouped_varlen_attention as varlen_attention
    from .mlp_unpadding import _active_tokens

    global _enabled
    _enabled = True
    original_model = model.Qwen2Model.forward
    if getattr(original_model, "_setww_compact_decoder", False):
        return True

    @wraps(original_model)
    def model_forward(self, *args, **kwargs):
        active = _active_tokens.get()
        mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        cache = kwargs.get("use_cache")
        if cache is None:
            cache = getattr(self.config, "use_cache", True)
        supported = (
            _enabled and self.training and torch.is_grad_enabled() and active is not None
            and mask is not None and mask.is_cuda and mask.ndim == 2
            and tuple(mask.shape) == tuple(active[1]) and not cache
            and not kwargs.get("past_key_values")
            and not kwargs.get("output_hidden_states") and not kwargs.get("output_attentions")
            and not getattr(self.config, "output_hidden_states", False)
            and not getattr(self.config, "output_attentions", False)
            and not getattr(self, "gradient_checkpointing", False)
            and not getattr(self, "has_sliding_layers", False)
            and self.config.hidden_size == 1536 and self.config.num_attention_heads == 12
            and self.config.num_key_value_heads == 2 and self.config.intermediate_size == 8960
            and self.config._attn_implementation == "sdpa"
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
            and all(layer.self_attn.attention_dropout == 0.0 and all(
                isinstance(module, Linear4bit) and not module.weight.requires_grad
                for module in (layer.self_attn.k_proj, layer.self_attn.o_proj,
                               layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj))
                for layer in self.layers)
        )
        if not supported:
            return original_model(self, *args, **kwargs)
        lengths = active[2] if len(active) > 2 else mask.bool().sum(1, dtype=torch.int32)
        if torch.any(lengths == 0).item():
            return original_model(self, *args, **kwargs)
        cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        layout = _Layout(tuple(mask.shape), active[0], cumulative)
        token = _current.set(layout)
        try:
            output = original_model(self, *args, **kwargs)
        finally:
            _current.reset(token)
        if not layout.packed:
            return output
        hidden = output[0]
        if hidden.ndim != 2 or hidden.shape != (layout.tokens, 1536):
            raise RuntimeError("compact decoder lost its real-token layout")
        full = hidden.new_zeros((layout.shape[0]*layout.shape[1], hidden.shape[-1]))
        full = full.index_copy_(0, layout.indices, hidden).view(*layout.shape, hidden.shape[-1])
        if isinstance(output, tuple):
            return (full, *output[1:])
        output.last_hidden_state = full
        return output

    model_forward._setww_compact_decoder = True
    model.Qwen2Model.forward = model_forward
    original_layer = model.Qwen2DecoderLayer.forward

    @wraps(original_layer)
    def layer_forward(self, hidden_states, *args, **kwargs):
        layout = _current.get()
        if (layout is not None and not layout.packed and self.self_attn.layer_idx == 0
                and hidden_states.ndim == 3 and tuple(hidden_states.shape[:2]) == layout.shape):
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1]).index_select(
                0, layout.indices)
            layout.packed = True
        # Delegate through the original selective-offload wrapper and layer math.
        return original_layer(self, hidden_states, *args, **kwargs)

    model.Qwen2DecoderLayer.forward = layer_forward
    original_attention = model.Qwen2Attention.forward

    @wraps(original_attention)
    def attention_forward(self, hidden_states, position_embeddings, attention_mask=None,
                          past_key_values=None, **kwargs):
        layout = _current.get()
        if (layout is None or not layout.packed or hidden_states.ndim != 2
                or hidden_states.shape != (layout.tokens, 1536)):
            return original_attention(self, hidden_states, position_embeddings, attention_mask,
                                      past_key_values=past_key_values, **kwargs)
        output, weights = original_attention(
            self, hidden_states.unsqueeze(0), layout.rotary(position_embeddings), attention_mask,
            past_key_values=past_key_values, **kwargs)
        return output.squeeze(0), weights

    model.Qwen2Attention.forward = attention_forward
    original_sdpa = model.ALL_ATTENTION_FUNCTIONS["sdpa"]

    @wraps(original_sdpa)
    def sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        layout = _current.get()
        if (layout is not None and layout.packed and isinstance(module, model.Qwen2Attention)
                and query.shape == (1, 12, layout.tokens, 128) and dropout == 0.0):
            output = varlen_attention(query, key, value, None, layout.cumulative, scaling,
                                      max_sequence=layout.shape[1])
            return output.transpose(1, 2).contiguous(), None
        return original_sdpa(module, query, key, value, attention_mask, dropout=dropout,
                             scaling=scaling, **kwargs)

    model.ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa)
    original_dropout = torch.nn.Dropout.forward

    @wraps(original_dropout)
    def dropout_forward(self, source):
        layout = _current.get()
        supported = (layout is not None and layout.packed and self.training and not self.inplace
                     and self.p == 0.05 and source.is_cuda and source.is_contiguous()
                     and source.dtype in (torch.float16, torch.float32)
                     and source.shape == (1, layout.tokens, 1536))
        if not supported:
            return original_dropout(self, source)
        return compact_native_dropout(source, layout.indices, layout.shape, self.p)

    torch.nn.Dropout.forward = dropout_forward
    return True
