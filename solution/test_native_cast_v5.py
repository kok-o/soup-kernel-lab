"""Native autograd parity, including the deliberate FP32 MLP fork reduction."""

import pytest

from solution import _speed_config


@pytest.mark.parametrize("mode", ["triton", "native", "native_autograd"])
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
@pytest.mark.parametrize("strided", [False, True])
def test_cast_exact_values_scaled_gradient(mode, dtype_name, scale, strided, monkeypatch):
    import torch

    from solution.dtype_cast import training_cast

    monkeypatch.setattr(_speed_config, "CAST_MODE", mode)
    dtype = getattr(torch, dtype_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = torch.randn(17, 33, device=device, dtype=dtype, requires_grad=True)
    if strided:
        source = source.t()
    target = torch.float32 if dtype == torch.float16 else torch.float16
    assert training_cast(source, dtype) is source
    incoming = torch.randn_like(source, dtype=target) * (1e-5 * scale)
    actual, expected = training_cast(source, target), source.to(target)
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual, source, incoming)[0],
        torch.autograd.grad(expected, source, incoming)[0],
    )


@pytest.mark.parametrize("mode", ["triton", "native", "native_autograd"])
def test_fork_still_widens_before_add(mode, monkeypatch):
    import torch

    from solution.fp16_mlp import _fork_cast

    monkeypatch.setattr(_speed_config, "CAST_MODE", mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = torch.randn(17, 128, device=device, requires_grad=True)
    gate, up = _fork_cast().apply(source)
    dg = torch.full_like(gate, 1000)
    du = torch.full_like(up, 0.25)
    (actual,) = torch.autograd.grad((gate, up), source, (dg, du))
    assert torch.equal(actual, dg.float() + du.float())
    assert not torch.equal(actual, (dg + du).float())


@pytest.mark.parametrize("mode", ["native", "native_autograd"])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_native_cast_real_nf4_lora_rng_updates(mode, scale, monkeypatch):
    import torch

    from solution.dtype_cast import patch_quantized_casts
    from solution.test_lora_fusion import test_real_nf4_lora_preserves_rng_gradients_and_update

    if not torch.cuda.is_available():
        pytest.skip("CUDA/bnb required")

    monkeypatch.setattr(_speed_config, "CAST_MODE", mode)
    assert patch_quantized_casts()
    test_real_nf4_lora_preserves_rng_gradients_and_update(scale)
