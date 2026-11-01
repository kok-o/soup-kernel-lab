"""Bitwise equivalence to Tensor.to, including scaled backwards."""

import pytest
import torch

from research.staged_cast.dtype_cast import training_cast


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("source_dtype,target_dtype", [
    (torch.float16, torch.float32), (torch.float32, torch.float16),
])
@pytest.mark.parametrize("shape", [(0,), (31, 127), (4, 19, 1536)])
def test_values_and_scaled_backward(device, source_dtype, target_dtype, shape):
    torch.manual_seed(97)
    source = torch.randn(shape, device=device, dtype=source_dtype, requires_grad=True)
    ref = source.detach().clone().requires_grad_()
    incoming = torch.randn(shape, device=device, dtype=target_dtype) * 128
    actual = training_cast(source, target_dtype)
    expected = ref.to(target_dtype)
    assert torch.equal(actual, expected)
    actual.backward(incoming)
    expected.backward(incoming)
    assert torch.equal(source.grad, ref.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/bnb required")
def test_actual_nf4_linear_values_and_gradients(monkeypatch):
    import bitsandbytes as bnb

    from research.staged_cast.dtype_cast import patch_quantized_casts

    # conftest may already have applied the solution. Restore the genuine bnb
    # implementation for the reference, otherwise both sides test our patch.
    original = bnb.nn.Linear4bit.forward
    while getattr(original, "_setww_training_casts", False):
        original = original.__wrapped__
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", original)
    torch.manual_seed(91)
    layer = bnb.nn.Linear4bit(128, 257, bias=True, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4").cuda()
    layer.weight.requires_grad_(False)
    layer.train()
    x = torch.randn(2, 19, 128, device="cuda", requires_grad=True)
    gradient = torch.randn(2, 19, 257, device="cuda") * 128
    with torch.autocast("cuda", dtype=torch.float16):
        reference = layer(x)
    reference_grad, = torch.autograd.grad(reference, x, gradient)
    assert patch_quantized_casts()
    patched = bnb.nn.Linear4bit.forward
    assert patch_quantized_casts() and bnb.nn.Linear4bit.forward is patched
    with torch.autocast("cuda", dtype=torch.float16):
        actual = layer(x)
    actual_grad, = torch.autograd.grad(actual, x, gradient)
    assert torch.equal(actual, reference)
    assert torch.equal(actual_grad, reference_grad)


def test_strided_fallback_and_same_dtype_identity():
    x = torch.randn(17, 11).T.requires_grad_()
    assert training_cast(x, x.dtype) is x
    y = training_cast(x, torch.float16)
    assert torch.equal(y, x.half())
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
