"""Lossy storage of large saved activations, scoped to the training decoder.

The live forward tensors and parameters are unchanged. Backward reads a 4/8-bit
block-scaled reconstruction; end-to-end quality must be measured separately.
This does not enable Soup's checkpointing, offloading or other training flags.
"""

from __future__ import annotations

import weakref
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from torch import Tensor


_storage_bits = 8
_offload_storage = False
_storage_block = 256
_wide_only = False
_active_codec_hooks = ContextVar("setww_active_codec_hooks", default=None)


class _Packet:
    __slots__ = (
        "packed", "scales", "shape", "dtype", "decoded", "original", "version",
        "bits", "block", "__weakref__"
    )

    def __init__(self, packed: Tensor, scales: Tensor, original: Tensor, bits: int = 8,
                 block: int = 256) -> None:
        self.packed, self.scales = packed, scales
        self.shape, self.dtype = original.shape, original.dtype
        version_source = original._base if original._base is not None else original
        self.original, self.version = weakref.ref(version_source), original._version
        self.decoded = None
        self.bits, self.block = bits, block


def pack_activation(tensor: Tensor, block: int = 256, *, bits: int = 8) -> _Packet:
    """Pack one finite CUDA FP16/FP32 tensor; return its shape and dtype metadata."""
    import torch
    import triton

    from ._activation_kernels import activation_pack_kernel
    from ._grouped_activation_kernels import grouped_activation_pack4_kernel

    if bits not in (4, 8):
        raise ValueError("activation storage supports 4 or 8 bits")
    if block not in ((64, 256) if bits == 4 else (256,)):
        raise ValueError("INT8 uses block 256; INT4 supports blocks 64 or 256")
    if not tensor.is_cuda or tensor.dtype not in (torch.float16, torch.float32):
        raise ValueError("packing requires CUDA FP16 or FP32")
    source = tensor.contiguous()
    elements = source.numel()
    packed = torch.empty(elements if bits == 8 else (elements + 1) // 2,
                         dtype=torch.int8 if bits == 8 else torch.uint8, device=tensor.device)
    scales = torch.empty(triton.cdiv(elements, block), dtype=torch.float32, device=tensor.device)
    with torch.cuda.device(tensor.device):
        if bits == 8:
            activation_pack_kernel[(scales.numel(),)](
                source, packed, scales, elements, block=block, num_warps=4,
            )
        else:
            grouped_activation_pack4_kernel[(triton.cdiv(scales.numel(), 8),)](
                source, packed, scales, elements, block=block, groups=8, num_warps=4,
            )
    return _Packet(packed, scales, tensor, bits, block)


def unpack_activation(packet: _Packet) -> Tensor:
    """Restore values in a contiguous tensor; keep only a weak decoded cache."""
    import torch

    from ._activation_kernels import activation_unpack_kernel
    from ._grouped_activation_kernels import grouped_activation_unpack4_kernel

    original = packet.original()
    if original is not None and original._version != packet.version:
        raise RuntimeError("a packed saved activation was modified in-place")
    cached = packet.decoded() if packet.decoded is not None else None
    if cached is not None:
        return cached
    restored = torch.empty(packet.shape, dtype=packet.dtype, device=packet.packed.device)
    with torch.cuda.device(restored.device):
        if packet.bits == 8:
            activation_unpack_kernel[(packet.scales.numel(),)](
                packet.packed, packet.scales, restored, restored.numel(),
                block=packet.block, num_warps=4,
            )
        else:
            grouped_activation_unpack4_kernel[((restored.numel()+1023)//1024,)](
                packet.packed, packet.scales, restored, restored.numel(),
                block=packet.block, tile=1024, num_warps=4,
            )
    packet.decoded = weakref.ref(restored)
    return restored


def saved_activation_hooks(
    min_elements: int = 262144, *, bits: int = 8, block: int = 256, wide_only: bool = False,
) -> tuple[Callable, Callable]:
    """Return hooks with per-forward identity/version-safe, non-owning deduplication."""
    import torch

    cached = {}

    def pack(tensor: Tensor) -> Tensor | _Packet:
        supported = (
            tensor.is_cuda and tensor.layout == torch.strided
            and tensor.dtype in (torch.float16, torch.float32)
            and tensor.requires_grad and not tensor.is_leaf
            # Preserve attention's 4-D saved tensors. Quantizing these produced
            # non-finite Q/K gradients in the real T4 SDPA backward despite
            # finite reconstructed values; see the negative diagnostic evidence.
            and tensor.ndim in (2, 3) and tensor.numel() >= min_elements
        )
        if not supported:
            return tensor
        key = id(tensor)
        previous = cached.get(key)
        if previous is not None:
            original, version, reference = previous
            packet = reference()
            if original() is tensor and version == tensor._version and packet is not None:
                return packet
        packet_bits = 8 if bits == 4 and wide_only and tensor.shape[-1] != 8960 else bits
        packet = pack_activation(tensor, bits=packet_bits, block=block if packet_bits == 4 else 256)
        cached[key] = (weakref.ref(tensor), tensor._version, weakref.ref(packet))
        return packet

    def unpack(value: Tensor | _Packet) -> Tensor:
        return unpack_activation(value) if isinstance(value, _Packet) else value

    return pack, unpack


def patch_activation_storage(
    *, bits: int = 8, block: int = 256, wide_only: bool = False, offload: bool = False,
) -> bool:
    """Wrap the Qwen2 decoder only during training with autograd enabled."""
    import torch

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    except (ImportError, AttributeError):
        return False
    if bits not in (4, 8):
        raise ValueError("activation storage supports 4 or 8 bits")
    global _storage_bits, _offload_storage, _storage_block, _wide_only
    _storage_bits, _offload_storage = bits, offload
    _storage_block, _wide_only = block, wide_only
    original = Qwen2Model.forward
    if getattr(original, "_setww_activation_storage", False):
        return True

    @wraps(original)
    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.training or not torch.is_grad_enabled():
            return original(self, *args, **kwargs)
        pack, unpack = saved_activation_hooks(
            bits=_storage_bits, block=_storage_block, wide_only=_wide_only,
        )
        if _offload_storage == "selective":
            from .activation_offload import reset_offload_step, sync_d2h
            from .selective_offload import selective_packed_hooks

            reset_offload_step()
            pack, unpack = selective_packed_hooks(pack, unpack, offload_raw=True)
        elif _offload_storage:
            from .activation_offload import offload_packed_hooks

            pack, unpack = offload_packed_hooks(pack, unpack)
        token = _active_codec_hooks.set((pack, unpack))
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                out = original(self, *args, **kwargs)
            if _offload_storage == "selective":
                sync_d2h()
            return out
        finally:
            _active_codec_hooks.reset(token)

    forward._setww_activation_storage = True
    Qwen2Model.forward = forward
    return True
