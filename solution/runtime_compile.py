"""Overlap cold kernel/launcher preparation with imports and model loading.

JITFunction.warmup uses dtype mock tensors. Reading the compiled run property
initializes its host launcher and CUDA module without executing a GPU kernel.
No prebuilt caches, task changes or external process. Join before forward.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import wraps

_pending = None
_started = False


def _jobs():
    import torch
    from triton.runtime.jit import MockTensor

    from ._activation_kernels import (
        activation_pack_kernel,
        activation_unpack_kernel,
    )
    from ._dropout_kernels import dropout_mask_gradient_kernel, dropout_mask_pack_kernel
    from ._grouped_activation_kernels import (
        grouped_activation_pack4_kernel,
        grouped_activation_unpack4_kernel,
    )
    from ._lora_kernels import lora_scale_add_kernel, lora_scale_gradient_kernel
    from ._packed_swiglu_kernels import example_swiglu_packed_bwd_kernel
    from ._rmsnorm_kernels import rmsnorm_backward_kernel, rmsnorm_forward_kernel
    from ._rope_kernels import rotary_backward_kernel, rotary_forward_kernel
    from .dtype_cast import _kernel
    from .fork_sum import _kernel as fork_kernel
    from .hosthead_stats import _hosthead_row_stats
    from .swiglu import example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

    f = MockTensor(torch.float32)
    h = MockTensor(torch.float16)
    i = MockTensor(torch.int8)
    u = MockTensor(torch.uint8)
    label = MockTensor(torch.int64)
    boolean = MockTensor(torch.bool)
    jobs = []

    def add(kernel, args, **kwargs):
        jobs.append((kernel, args, dict(grid=(1,), num_warps=4, **kwargs)))

    add(activation_pack_kernel, (f, i, f, 512), block=256)
    add(activation_unpack_kernel, (i, f, f, 512), block=256)
    add(activation_pack_kernel, (h, i, f, 512), block=256)
    add(activation_unpack_kernel, (i, f, h, 512), block=256)
    add(grouped_activation_pack4_kernel, (h, u, f, 512), block=64, groups=8)
    add(grouped_activation_unpack4_kernel, (u, f, h, 512), block=64, tile=1024)
    add(rmsnorm_forward_kernel, (f, f, f, f, 1536, 1e-6, 2048), enable_fp_fusion=False)
    add(rmsnorm_backward_kernel, (f, f, f, f, f, 1536, 2048), enable_fp_fusion=False)
    from ._speed_config import CAST_MODE
    if CAST_MODE == "triton":
        add(_kernel(), (f, h, 512), block=4096)
        add(_kernel(), (h, f, 512), block=4096)
    # HostHead uses online row statistics instead of the registered CE backend.
    # Compile real full and tail layouts; the measured process still pays this cost.
    for columns in (8192, 4480):
        add(_hosthead_row_stats,
            (h, h, label, h, f, f, f, f, label, f, h, f,
             columns, columns, 1536, 0),
            feature_count=1536, block_cols=8192, block_features=2048,
            enable_fp_fusion=False)
    add(example_swiglu_fwd_kernel, (h, h, h, 8960), BLOCK=1024)
    add(example_swiglu_bwd_kernel, (h, h, h, h, h, 8960), BLOCK=1024)
    add(example_swiglu_packed_bwd_kernel, (h, u, f, u, f, h, h, 8960), block=1024)
    add(dropout_mask_pack_kernel, (boolean, u, 512), groups=128)
    add(dropout_mask_gradient_kernel, (f, u, f, 512), scale=1.0 / 0.95,
        block=1024, enable_fp_fusion=False)
    add(lora_scale_add_kernel, (f, h, f, 512), scale=2.0,
        block=1024, enable_fp_fusion=False)
    add(lora_scale_gradient_kernel, (f, h, 512), scale=2.0,
        block=1024, enable_fp_fusion=False)
    add(fork_kernel(), (h, h, f, 512), block=4096, enable_fp_fusion=False)
    for width in (1536, 256):
        strides = (512 * width, 128, width)
        for factor_batch in (0, 512 * 128):
            add(rotary_forward_kernel, (f, f, f, h if width == 1536 else f,
                4 * 512 * 128, 512, *strides, *strides, factor_batch, 128),
                block=512, enable_fp_fusion=False)
            add(rotary_backward_kernel, (h if width == 1536 else f, f, f, f,
                4 * 512 * 128, 512, *strides, *strides, factor_batch, 128),
                block=512, enable_fp_fusion=False)
    return jobs


def _compile(job):
    import torch

    kernel, args, kwargs = job
    with torch.cuda.device(0):
        compiled = kernel.warmup(*args, **kwargs)
        if compiled is not None:
            # warmup alone leaves the host launcher/CUDA module lazy in Triton 3.6.
            # Its run property initializes those handles without calling the kernel.
            _ = compiled.run


def finish_runtime_compile():
    global _pending
    pending, _pending = _pending, None
    if pending is None:
        return
    executor, futures = pending
    failures = []
    try:
        for future in futures:
            try:
                future.result()
            except Exception as error:
                failures.append(type(error).__name__)
    finally:
        executor.shutdown(wait=True)
    if failures:
        from rich.console import Console

        Console().print({"runtime_compile_fallback": failures})


def start_runtime_compile():
    import torch

    global _pending, _started
    if _started or _pending is not None or not torch.cuda.is_available():
        return False
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    except (ImportError, AttributeError):
        return False
    original = Qwen2Model.forward
    if not getattr(original, "_setww_runtime_compile", False):
        @wraps(original)
        def forward(self, *args, **kwargs):
            finish_runtime_compile()
            return original(self, *args, **kwargs)

        forward._setww_runtime_compile = True
        Qwen2Model.forward = forward
    jobs = _jobs()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="setww-cold-compile")
    _started = True
    _pending = (executor, [executor.submit(_compile, job) for job in jobs])
    return True
