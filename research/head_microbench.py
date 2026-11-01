"""Bounded product tuning; isolated from the official end-to-end measurements."""

import json
import sys
from pathlib import Path


def main():
    import torch
    import torch.nn.functional as functional
    import triton
    import triton.testing
    from rich.console import Console

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research.rejected_head_gemm._head_kernels import head_product_kernel, head_reduce_kernel

    console = Console()
    torch.manual_seed(29)
    records = []
    configs = [(32, 64, 32, 4), (64, 64, 32, 4), (64, 128, 32, 4),
               (64, 128, 32, 8), (32, 128, 32, 4), (64, 64, 64, 4)]
    for rows in (512, 127):
        hidden = torch.randn(rows, 1536, dtype=torch.float16, device="cuda")
        weight = torch.randn(151936, 1536, dtype=torch.float16, device="cuda") / 1536 ** 0.5
        incoming = torch.randn(rows, 151936, dtype=torch.float16, device="cuda") / 151936 ** 0.5
        for direction, left, right in (("forward", hidden, weight.t()),
                                       ("backward", incoming, weight)):
            columns, reduction = right.shape[1], left.shape[1]
            reference = left @ right
            split_options = (1,) if direction == "forward" else (8, 16, 32)
            output = torch.empty(rows, columns, dtype=torch.float16, device="cuda")
            for splits in split_options:
                partial = output if splits == 1 else torch.empty(
                    splits, rows, columns, dtype=torch.float32, device="cuda"
                )
                for tile_rows, tile_columns, tile_k, warps in configs:
                    def launch():
                        head_product_kernel[(triton.cdiv(rows, tile_rows)
                                             * triton.cdiv(columns, tile_columns), splits)](
                            left, right, partial, rows, columns, reduction,
                            *left.stride(), *right.stride(), splits,
                            tile_rows, tile_columns, tile_k, num_warps=warps,
                        )
                        if splits > 1:
                            head_reduce_kernel[(triton.cdiv(rows * columns, 128),)](
                                partial, output, rows * columns, splits=splits,
                                block=128, num_warps=4,
                            )

                    try:
                        launch()
                        torch.testing.assert_close(output, reference, rtol=0.005, atol=0.005)
                        milliseconds = triton.testing.do_bench(launch, warmup=30, rep=100)
                        record = dict(rows=rows, direction=direction, splits=splits,
                                      tile=[tile_rows, tile_columns, tile_k], warps=warps,
                                      ms=round(milliseconds, 4))
                    except Exception as error:
                        record = dict(rows=rows, direction=direction, splits=splits,
                                      tile=[tile_rows, tile_columns, tile_k], warps=warps,
                                      error=f"{type(error).__name__}: {str(error)[:300]}")
                    records.append(record)
                    console.print(record)
            reference_ms = triton.testing.do_bench(
                lambda: functional.linear(left, right.t()), warmup=30, rep=100
            )
            records.append(dict(rows=rows, direction=direction, backend="torch",
                                ms=round(reference_ms, 4)))
            console.print(records[-1])
    Path("head_microbench.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
