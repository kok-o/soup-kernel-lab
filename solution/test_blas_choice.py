"""Cross-BLAS numerical checks with the same precision flags and dropout stream."""

import pytest

from solution import _speed_config


@pytest.mark.skipif(
    _speed_config.BLAS_LIBRARY != "cublaslt", reason="cross-backend gate for Lt candidate"
)
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_blas_real_nf4_lora_scaled_backward_and_update(scale):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit

    if not torch.cuda.is_available():
        pytest.skip("CUDA/bnb required")

    torch.manual_seed(712)
    base = bnb.nn.Linear4bit(
        1536,
        256,
        bias=False,
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type="nf4",
    ).cuda()
    base.weight.requires_grad_(False)
    layer = (
        Linear4bit(
            base,
            "default",
            r=16,
            lora_alpha=32,
            config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05),
        )
        .cuda()
        .train()
    )
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.01)
    parameters = [p for p in layer.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in parameters]
    source = torch.randn(4, 129, 1536, device="cuda")
    incoming = torch.randn(4, 129, 256, device="cuda") * (1e-5 * scale)
    rng = torch.cuda.get_rng_state()
    preference = torch.backends.cuda.preferred_blas_library()
    precision = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_fp16_accumulation,
    )

    def execute(backend):
        torch.backends.cuda.preferred_blas_library(backend)
        torch.cuda.set_rng_state(rng)
        for p, value in zip(parameters, initial, strict=True):
            p.data.copy_(value)
            p.grad = None
        x = source.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            output = layer(x)
        output.backward(incoming)
        gradients = [x.grad.detach().clone() / scale]
        for p in parameters:
            p.grad.div_(scale)
            gradients.append(p.grad.detach().clone())
        torch.optim.SGD(parameters, lr=2e-4).step()
        return (
            output.detach(),
            gradients,
            [p.detach().clone() for p in parameters],
            torch.cuda.get_rng_state(),
        )

    try:
        reference = execute("cublas")
        actual = execute("cublaslt")
        assert torch.equal(actual[3], reference[3])
        # Different native GEMM algorithms can round differently; exact equivalence
        # remains required in the cast/fork tests. These bounds are fixed before T4 runs.
        torch.testing.assert_close(actual[0], reference[0], rtol=3e-3, atol=2e-4)
        for got, expected in zip(actual[1], reference[1], strict=True):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, expected, rtol=5e-3, atol=2e-7)
        for got, expected in zip(actual[2], reference[2], strict=True):
            torch.testing.assert_close(got, expected, rtol=0, atol=2e-7)
        assert precision == (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_accumulation,
        )
    finally:
        torch.backends.cuda.preferred_blas_library(preference)
