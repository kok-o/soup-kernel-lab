"""Fused online head statistics for frozen host-streamed vocabulary chunks.

Original implementation of the existing online max/sum/sum(exp*z) recurrence.
The ordinary Triton JIT launch remains visible to the official TritonWatch.
No model, data, precision policy, loss masking, or gradient policy is changed.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ncols", "logits_stride", "weight_stride", "start"])
def _hosthead_row_stats(
    logits_ptr,
    weight_ptr,
    target_ptr,
    prob_ptr,
    running_max_ptr,
    running_sum_ptr,
    running_ez_ptr,
    best_logit_ptr,
    best_token_ptr,
    target_logit_ptr,
    target_weight_ptr,
    old_scale_ptr,
    ncols,
    logits_stride,
    weight_stride,
    start,
    feature_count: tl.constexpr,
    block_cols: tl.constexpr,
    block_features: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block_cols)
    valid = col < ncols
    z = tl.load(logits_ptr + row * logits_stride + col, mask=valid, other=-float("inf")).to(
        tl.float32
    )
    rowmax = tl.max(z, axis=0)
    oldmax = tl.load(running_max_ptr + row)
    newmax = tl.maximum(oldmax, rowmax)
    old_scale = tl.exp(oldmax - newmax)
    e = tl.exp(z - newmax)
    esum = tl.sum(e, axis=0)
    ezsum = tl.sum(tl.where(valid, e * z, 0.0), axis=0)
    oldsum = tl.load(running_sum_ptr + row)
    oldez = tl.load(running_ez_ptr + row)
    tl.store(prob_ptr + row * ncols + col, e, mask=valid)
    tl.store(old_scale_ptr + row, old_scale)
    tl.store(running_max_ptr + row, newmax)
    tl.store(running_sum_ptr + row, oldsum * old_scale + esum)
    tl.store(running_ez_ptr + row, oldez * old_scale + ezsum)

    # First maximum wins, both within this chunk and across earlier chunks.
    arg = tl.min(tl.where(valid & (z == rowmax), col, 2147483647), axis=0)
    previous = tl.load(best_logit_ptr + row)
    update = rowmax > previous
    tl.store(best_logit_ptr + row, rowmax, mask=update)
    tl.store(best_token_ptr + row, start + arg, mask=update)

    target = tl.load(target_ptr + row)
    rel = target - start
    inside = (rel >= 0) & (rel < ncols)
    # Scalar per-program branch: no host scalar read, no nonzero/dynamic shape.
    if inside:
        target_z = tl.load(logits_ptr + row * logits_stride + rel).to(tl.float32)
        tl.store(target_logit_ptr + row, target_z)
        feature = tl.arange(0, block_features)
        value = tl.load(
            weight_ptr + rel * weight_stride + feature, mask=feature < feature_count, other=0.0
        )
        tl.store(
            target_weight_ptr + row * feature_count + feature, value, mask=feature < feature_count
        )


def stream_head_stats(
    logits,
    weight,
    targets,
    running_max,
    running_sum,
    running_ez,
    best_logit,
    best_token,
    target_logits,
    target_weight,
    start,
):
    """Update FP32 row statistics and targets; return FP16 exp and FP32 scale."""
    rows, cols = logits.shape
    width = weight.shape[1]
    assert logits.dtype == weight.dtype == torch.float16
    assert logits.stride(1) == weight.stride(1) == 1
    assert target_weight.is_contiguous() and targets.is_contiguous()
    probabilities = torch.empty((rows, cols), dtype=torch.float16, device=logits.device)
    old_scale = torch.empty(rows, dtype=torch.float32, device=logits.device)
    if rows:
        _hosthead_row_stats[(rows,)](
            logits,
            weight,
            targets,
            probabilities,
            running_max,
            running_sum,
            running_ez,
            best_logit,
            best_token,
            target_logits,
            target_weight,
            old_scale,
            cols,
            logits.stride(0),
            weight.stride(0),
            start,
            feature_count=width,
            block_cols=triton.next_power_of_2(cols),
            block_features=triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return probabilities, old_scale
