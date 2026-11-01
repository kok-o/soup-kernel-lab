"""Original dtype-conversion kernels for the quantized training path.

Only replaces real FP16/FP32 conversions already performed by Linear4bit.
No extra copies, waits, arithmetic, persistent weight caches or dtype-policy
changes are added to the training graph. CPU is the reference implementation.
"""

from __future__ import annotations

from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["elements"])
    def training_dtype_cast(source, destination, elements, block: tl.constexpr):
        addresses = tl.program_id(0) * block + tl.arange(0, block)
        inside = addresses < elements
        original = tl.load(source + addresses, inside, other=0)
        converted = original.to(destination.dtype.element_ty)
        tl.store(destination + addresses, converted, inside)

    return training_dtype_cast


def _convert(source: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    import torch

    if source.dtype == dtype:
        return source
    from ._speed_config import CAST_MODE
    if CAST_MODE != "triton":
        return source.to(dtype)
    if (not source.is_cuda or not source.is_contiguous()
            or source.dtype not in (torch.float16, torch.float32)
            or dtype not in (torch.float16, torch.float32)):
        return source.to(dtype)
    import triton

    output = torch.empty_like(source, dtype=dtype)
    if source.numel():
        with torch.cuda.device(source.device):
            _kernel()[(triton.cdiv(source.numel(), 4096),)](
                source, output, source.numel(), block=4096,
            )
    return output


@lru_cache(maxsize=1)
def _cast_autograd() -> Any:
    import torch

    class _TrainingCast(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source, dtype):
            ctx.input_dtype = source.dtype
            return _convert(source, dtype)

        @staticmethod
        def backward(ctx, incoming):
            # Incoming GradScaler scale is already present before this conversion,
            # matching the original Tensor.to backward.
            return _convert(incoming, ctx.input_dtype), None

    return _TrainingCast


def training_cast(source: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if source.dtype == dtype:
        return source
    from ._speed_config import CAST_MODE
    if CAST_MODE == "native_autograd":
        return source.to(dtype)
    return _cast_autograd().apply(source, dtype)


def patch_quantized_casts() -> bool:
    """Keep bitsandbytes' math/dispatch, replace only its input/output casts.

    Linear4bit.forward casts x to compute_dtype and its result back to the
    original x dtype. Passing the already-converted x suppresses BOTH original
    casts; we supply their replacements around the original forward. Run the
    original compute-type selection on the original input before any cast.
    """
    import torch

    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        return False
    if getattr(Linear4bit.forward, "_setww_training_casts", False):
        return True
    original = Linear4bit.forward

    @wraps(original)
    def forward(self, x):
        if (not self.training or not x.is_cuda or x.dtype != torch.float32
                or self.compute_dtype != torch.float16 or not x.is_contiguous()):
            return original(self, x)
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True
        if self.compute_dtype != torch.float16:
            return original(self, x)
        result = original(self, training_cast(x, self.compute_dtype))
        return training_cast(result, x.dtype)

    forward._setww_training_casts = True
    Linear4bit.forward = forward
    return True
