"""Original FP16 frozen-head GEMM and deterministic split-K dHidden kernels.

Standard blocked matrix products, implemented for the T4 experiment. Shapes
that vary with token chunks are runtime scalars, not per-length specializations.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def head_product_kernel(
    left_ptr, right_ptr, output_ptr,
    rows, columns: tl.constexpr, reduction: tl.constexpr,
    left_row: tl.constexpr, left_k: tl.constexpr,
    right_k: tl.constexpr, right_column: tl.constexpr,
    splits: tl.constexpr, tile_rows: tl.constexpr,
    tile_columns: tl.constexpr, tile_k: tl.constexpr,
):
    product = tl.program_id(0)
    split = tl.program_id(1)
    column_tiles = tl.cdiv(columns, tile_columns)
    row_ids = (product // column_tiles) * tile_rows + tl.arange(0, tile_rows)
    column_ids = (product % column_tiles) * tile_columns + tl.arange(0, tile_columns)
    k_ids = tl.arange(0, tile_k)
    blocks = tl.cdiv(reduction, splits * tile_k)
    start = split * blocks * tile_k
    accumulator = tl.full((tile_rows, tile_columns), 0.0, tl.float32)
    for step in range(blocks):
        position = start + step * tile_k + k_ids
        left = tl.load(
            left_ptr + row_ids[:, None] * left_row + position[None, :] * left_k,
            (row_ids[:, None] < rows) & (position[None, :] < reduction), other=0.0,
        )
        right = tl.load(
            right_ptr + position[:, None] * right_k + column_ids[None, :] * right_column,
            (position[:, None] < reduction) & (column_ids[None, :] < columns), other=0.0,
        )
        accumulator = tl.dot(left, right, accumulator)
    offset = (split * rows + row_ids[:, None]) * columns + column_ids[None, :]
    tl.store(
        output_ptr + offset, accumulator,
        (row_ids[:, None] < rows) & (column_ids[None, :] < columns),
    )


@triton.jit(do_not_specialize=["elements"])
def head_reduce_kernel(
    partial_ptr, output_ptr, elements,
    splits: tl.constexpr, block: tl.constexpr,
):
    points = tl.program_id(0) * block + tl.arange(0, block)
    lanes = tl.arange(0, splits)
    values = tl.load(
        partial_ptr + lanes[:, None] * elements + points[None, :],
        points[None, :] < elements, other=0.0,
    )
    result = tl.sum(values, axis=0)
    tl.store(output_ptr + points, result, points < elements)
