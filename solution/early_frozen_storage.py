"""Retain the tied frozen FP16 matrix during the original PEFT preparation.

PEFT still freezes all named parameters, upcasts every other eligible parameter,
installs its input-gradient hooks and performs its original checkpoint/cache steps.
Only the tied embedding/head is omitted from that call's parameter-cast iterator.
"""

import sys
from functools import wraps


def patch_early_frozen_storage():
    import torch
    from peft.utils import other

    original = other.prepare_model_for_kbit_training
    if getattr(original, "_setww_early_frozen_storage", False):
        return True

    @wraps(original)
    def prepare(model, *args, **kwargs):
        embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        weight = getattr(embedding, "weight", None)
        supported = (getattr(model, "is_loaded_in_4bit", False)
                     and getattr(getattr(model, "config", None), "model_type", None) == "qwen2"
                     and isinstance(embedding, torch.nn.Embedding)
                     and isinstance(head, torch.nn.Linear) and head.weight is weight
                     and head.bias is None and embedding.max_norm is None
                     and weight.is_cuda and weight.dtype == torch.float16)
        if not supported:
            return original(model, *args, **kwargs)
        # The native named_parameters freeze loop still sees the complete model.
        # Preserve its effect even if a future caller supplies a parameter iterator.
        weight.requires_grad_(False)
        parameters = model.parameters
        had_override = "parameters" in model.__dict__
        override = model.__dict__.get("parameters")

        def cast_parameters(recurse=True):
            for parameter in parameters(recurse=recurse):
                if parameter is not weight:
                    yield parameter

        model.parameters = cast_parameters
        try:
            result = original(model, *args, **kwargs)
        finally:
            if had_override:
                model.parameters = override
            else:
                del model.parameters
        from .frozen_storage import compact_frozen_tied_embedding

        if result is model:
            model._setww_retained_frozen_half = True
            model._setww_prevented_fp32_bytes = weight.numel()*4
            compact_frozen_tied_embedding(model)
        return result

    prepare._setww_early_frozen_storage = True
    for name, module in list(sys.modules.items()):
        if (module is not None and name.split(".")[0] in ("peft", "soup_cli", "trl")
                and module.__dict__.get("prepare_model_for_kbit_training") is original):
            module.prepare_model_for_kbit_training = prepare
    return True
