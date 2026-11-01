"""Original FP32 frozen Qwen RMSNorm kernels, derived from its formula.

Reference formula: transformers v5.16.1 Qwen2RMSNorm. No external kernel code.
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_forward_kernel(source, weight, output, inverse,
                           width: tl.constexpr, epsilon: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    x = tl.load(source + row * width + col, col < width, other=0.0)
    w = tl.load(weight + col, col < width, other=0.0)
    variance = tl.sum(x * x, 0) / width
    inv = tl.rsqrt(variance + epsilon)
    normalized = x * inv
    tl.store(output + row * width + col, normalized * w, col < width)
    tl.store(inverse + row, inv)


@triton.jit
def rmsnorm_backward_kernel(incoming, source, weight, inverse, destination,
                            width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    x = tl.load(source + row * width + col, col < width, other=0.0)
    g = tl.load(incoming + row * width + col, col < width, other=0.0)
    w = tl.load(weight + col, col < width, other=0.0)
    inv = tl.load(inverse + row)
    weighted = g * w
    projection = tl.sum(weighted * x, 0) / width
    dx = weighted * inv - x * (projection * inv * inv * inv)
    tl.store(destination + row * width + col, dx, col < width)
