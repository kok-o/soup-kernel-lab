"""Lossless FP16 storage for a frozen tied embedding/head with FP32 public semantics.

Compaction proceeds only after every FP32 value passes an exact FP16 round-trip.
Embedding output stays FP32 before existing hooks run. Outside CUDA FP16
autocast the head reconstructs the original FP32 weight for ordinary F.linear.
"""

from __future__ import annotations

from functools import wraps
from typing import Any


def compact_frozen_tied_embedding(model: Any) -> bool:
    """Preserve parameter identity/tie and all arithmetic dtypes; change storage only."""
    import torch
    import torch.nn.functional as functional

    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embedding is None or head is None:
        return False
    if getattr(embedding, "_setww_compact_frozen_storage", False):
        return True
    weight = embedding.weight
    supported = (
        isinstance(embedding, torch.nn.Embedding) and isinstance(head, torch.nn.Linear)
        and head.weight is weight and not weight.requires_grad and head.bias is None
        and weight.is_cuda and weight.dtype in (torch.float32, torch.float16)
        and embedding.max_norm is None
    )
    if not supported:
        return False
    version = (id(weight), weight._version)
    if getattr(embedding, "_setww_storage_rejected", None) == version:
        return False
    with torch.no_grad():
        compact = weight.detach() if weight.dtype == torch.float16 else weight.to(torch.float16)
        if weight.dtype == torch.float32:
            # Existing FP16 storage needs no copy or round-trip check.
            for start in range(0, weight.shape[0], 4096):
                if not torch.equal(weight[start:start + 4096], compact[start:start + 4096].float()):
                    embedding._setww_storage_rejected = version
                    return False

    def original_output_dtype(_module: Any, _inputs: Any, output: Any) -> Any:
        return output.float()

    original_head = head.forward

    @wraps(original_head)
    def head_forward(hidden: Any) -> Any:
        original_compute = (
            hidden.is_cuda and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        )
        if original_compute:
            return original_head(hidden)
        return functional.linear(hidden, head.weight.float(), head.bias)

    # Run before PEFT's input-requires-grad hook: the FP32 output remains the
    # same leaf/gradient boundary as the original prepared FP32 embedding.
    embedding.register_forward_hook(original_output_dtype, prepend=True)
    head.forward = head_forward
    weight.data = compact
    embedding._setww_compact_frozen_storage = True
    return True
