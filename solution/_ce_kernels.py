"""Original tiled CE/entropy/argmax reductions for the SetWW T4 experiment.

Only imported when a CUDA tensor reaches the public API. No external kernel
implementation is used: the tile merge follows the softmax partition identity.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["sequence", "stride_batch"])
def ce_partition_kernel(
    logits_ptr,
    max_ptr,
    sum_ptr,
    weighted_ptr,
    argmax_ptr,
    sequence,
    vocab: tl.constexpr,
    stride_batch,
    stride_token: tl.constexpr,
    stride_vocab: tl.constexpr,
    partitions: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    columns = part * block_size + tl.arange(0, block_size)
    offset = (row // sequence) * stride_batch + (row % sequence) * stride_token
    values = tl.load(
        logits_ptr + offset + columns * stride_vocab, columns < vocab, other=-float("inf")
    ).to(tl.float32)
    maximum = tl.max(values, 0)
    centered = tl.where(columns < vocab, values - maximum, 0.0)
    weights = tl.where(columns < vocab, tl.exp(centered), 0.0)
    total = tl.sum(weights, 0)
    weighted = tl.sum(weights * centered, 0)
    first = tl.min(tl.where((columns < vocab) & (values == maximum), columns, 2147483647), 0)
    destination = row * partitions + part
    tl.store(max_ptr + destination, maximum)
    tl.store(sum_ptr + destination, total)
    tl.store(weighted_ptr + destination, weighted)
    tl.store(argmax_ptr + destination, first)


@triton.jit(do_not_specialize=["sequence", "stride_batch", "label_batch"])
def ce_merge_kernel(
    logits_ptr,
    labels_ptr,
    max_ptr,
    sum_ptr,
    weighted_ptr,
    argmax_ptr,
    lse_ptr,
    loss_ptr,
    entropy_ptr,
    correct_ptr,
    sequence,
    vocab: tl.constexpr,
    stride_batch,
    stride_token: tl.constexpr,
    stride_vocab: tl.constexpr,
    label_batch,
    label_token: tl.constexpr,
    partitions: tl.constexpr,
    ignore_index: tl.constexpr,
    label_shift: tl.constexpr,
    merge_block: tl.constexpr,
):
    row = tl.program_id(0)
    parts = tl.arange(0, merge_block)
    locations = row * partitions + parts
    maxima = tl.load(max_ptr + locations, parts < partitions, other=-float("inf"))
    totals = tl.load(sum_ptr + locations, parts < partitions, other=0.0)
    weighted = tl.load(weighted_ptr + locations, parts < partitions, other=0.0)
    maximum = tl.max(maxima, 0)
    difference = tl.where(parts < partitions, maxima - maximum, 0.0)
    scale = tl.where(parts < partitions, tl.exp(difference), 0.0)
    total = tl.sum(totals * scale, 0)
    centered_weighted = tl.sum((weighted + totals * difference) * scale, 0)
    log_total = tl.log(total)
    lse = maximum + log_total
    entropy = log_total - centered_weighted / total
    indices = tl.load(argmax_ptr + locations, parts < partitions, other=2147483647)
    prediction = tl.min(tl.where(maxima == maximum, indices, 2147483647), 0)

    batch = row // sequence
    token = row % sequence
    target = tl.load(
        labels_ptr + batch * label_batch + (token + label_shift) * label_token,
        token + label_shift < sequence,
        other=ignore_index,
    )
    supervised = (target != ignore_index) & (token + label_shift < sequence)
    valid = supervised & (target >= 0) & (target < vocab)
    offset = batch * stride_batch + token * stride_token
    target_logit = tl.load(
        logits_ptr + offset + target * stride_vocab, valid, other=0.0
    ).to(tl.float32)
    loss = tl.where(valid, lse - target_logit, 0.0)
    # Invalid targets are not silently turned into ignored tokens.
    loss = tl.where(supervised & ~valid, float("nan"), loss)
    tl.store(lse_ptr + row, lse)
    tl.store(loss_ptr + row, loss)
    tl.store(entropy_ptr + row, entropy)
    tl.store(correct_ptr + row, valid & (prediction == target))


@triton.jit(do_not_specialize=[
    "sequence", "stride_batch", "label_batch", "grad_batch", "grad_token"
])
def ce_backward_kernel(
    logits_ptr,
    labels_ptr,
    lse_ptr,
    incoming_ptr,
    gradient_ptr,
    sequence,
    vocab: tl.constexpr,
    stride_batch,
    stride_token: tl.constexpr,
    stride_vocab: tl.constexpr,
    label_batch,
    label_token: tl.constexpr,
    grad_batch,
    grad_token,
    ignore_index: tl.constexpr,
    label_shift: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
    batch = row // sequence
    token = row % sequence
    target = tl.load(
        labels_ptr + batch * label_batch + (token + label_shift) * label_token,
        token + label_shift < sequence,
        other=ignore_index,
    )
    valid = (target != ignore_index) & (target >= 0) & (target < vocab)
    offset = batch * stride_batch + token * stride_token
    logits = tl.load(
        logits_ptr + offset + columns * stride_vocab, columns < vocab, other=0.0
    ).to(tl.float32)
    lse = tl.load(lse_ptr + row)
    incoming = tl.load(incoming_ptr + batch * grad_batch + token * grad_token).to(tl.float32)
    probabilities = tl.exp(logits - lse)
    # Apply the incoming (possibly GradScaler-scaled) gradient in FP32 BEFORE
    # the store converts to logits.dtype. Ignored rows have exactly zero dX.
    gradient = tl.where(valid, (probabilities - (columns == target).to(tl.float32)) * incoming, 0.0)
    tl.store(gradient_ptr + row * vocab + columns, gradient, columns < vocab)
