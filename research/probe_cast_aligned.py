"""Bitwise equivalence to Tensor.to, including scaled backwards."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/bnb required")
@pytest.mark.parametrize("width,outputs,bias", [(128,256,True),(1536,1536,True),(1536,8960,False)])
def test_actual_nf4_linear_values_and_gradients(monkeypatch,width,outputs,bias):
    import bitsandbytes as bnb

    from research.staged_cast.dtype_cast import patch_quantized_casts

    # conftest may already have applied the solution. Restore the genuine bnb
    # implementation for the reference, otherwise both sides test our patch.
    original = bnb.nn.Linear4bit.forward
    while getattr(original, "_setww_training_casts", False):
        original = original.__wrapped__
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", original)
    torch.manual_seed(91)
    layer = bnb.nn.Linear4bit(width, outputs, bias=bias, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4").cuda()
    layer.weight.requires_grad_(False)
    layer.train()
    x = torch.randn(2, 19, width, device="cuda", requires_grad=True)
    gradient = torch.randn(2, 19, outputs, device="cuda") * 128
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

