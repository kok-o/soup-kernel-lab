"""Reuse exactly decoded nested NF4 scales while their canonical state is unchanged.

Only the small FP32 scale vectors are cached, not dense weights. Canonical
QuantState objects remain nested and unchanged for checkpoints and metadata.
The existing bitsandbytes dequantizer still supplies the NF4 values/matmul.
"""

from __future__ import annotations

import weakref
from functools import wraps
from typing import Any


class NestedScaleCache:
    def __init__(self, functional: Any) -> None:
        self.functional = functional
        self.entries: dict[int, tuple[Any, tuple, Any]] = {}

    @staticmethod
    def _fingerprint(state: Any) -> tuple:
        tensors = (state.absmax, state.state2.absmax, state.state2.code, state.offset)
        versions = tuple((id(value), value._version, value.device) for value in tensors)
        return (versions, state.state2.blocksize, state.state2.dtype, state.blocksize,
                state.dtype, state.quant_type, tuple(state.shape), id(state.code))

    def get(self, state: Any) -> Any:
        import torch

        key, fingerprint = id(state), self._fingerprint(state)
        previous = self.entries.get(key)
        if previous is not None:
            identity, version, decoded = previous
            if identity() is state and version == fingerprint:
                return decoded
        with torch.no_grad():
            # Use the exact installed bnb operation and separate FP32 add.
            # Computing it once changes no rounding or quantization policy.
            scales = self.functional.dequantize_blockwise(state.absmax, state.state2)
            scales.add_(state.offset)
            if scales.dtype != torch.float32:
                scales = scales.float()
        decoded = self.functional.QuantState(
            absmax=scales, shape=state.shape, code=state.code, blocksize=state.blocksize,
            quant_type=state.quant_type, dtype=state.dtype,
        )

        def discard(_reference: Any) -> None:
            self.entries.pop(key, None)

        self.entries[key] = (weakref.ref(state, discard), fingerprint, decoded)
        return decoded


def patch_nested_scale_cache() -> bool:
    """Replace repeated scale decoding with a version-checked non-owning cache."""
    try:
        from bitsandbytes import functional
    except (ImportError, AttributeError):
        return False
    original = getattr(functional, "dequantize_4bit", None)
    if not callable(original):
        return False
    if getattr(original, "_setww_nested_scales", False):
        return True
    cache = NestedScaleCache(functional)

    @wraps(original)
    def dequantize(*args: Any, **kwargs: Any) -> Any:
        packed = args[0] if args else kwargs.get("A")
        state = args[1] if len(args) > 1 else kwargs.get("quant_state")
        supported = (
            packed is not None and packed.is_cuda and state is not None
            and state.nested and not state.state2.nested
            and state.absmax.is_cuda and getattr(state.offset, "is_cuda", False)
            and state.state2.absmax.is_cuda and state.state2.code.is_cuda
            and not any(value.requires_grad for value in (
                state.absmax, state.offset, state.state2.absmax, state.state2.code
            ))
        )
        if supported:
            decoded = cache.get(state)
            if len(args) > 1:
                args = (args[0], decoded, *args[2:])
            else:
                kwargs = dict(kwargs, quant_state=decoded)
        return original(*args, **kwargs)

    dequantize._setww_nested_scales = True
    dequantize._setww_scale_cache = cache
    functional.dequantize_4bit = dequantize
    return True
