"""Compare the new frozen norm with real HF, including scaled gradients."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [7, 1536])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_frozen_norm_against_hf(monkeypatch, width, scale):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution.rmsnorm import patch_qwen_rmsnorm

    reference_forward = Qwen2RMSNorm.forward
    while getattr(reference_forward, "_setww_rmsnorm", False):
        reference_forward = reference_forward.__wrapped__
    monkeypatch.setattr(Qwen2RMSNorm, "forward", reference_forward)
    torch.manual_seed(523)
    norm = Qwen2RMSNorm(width, eps=1e-6).cuda().train()
    norm.weight.requires_grad_(False)
    with torch.no_grad():
        norm.weight.uniform_(0.4, 1.7)
    x = torch.randn(2, 19, width, device="cuda", requires_grad=True)
    incoming = torch.randn_like(x) * scale
    expected = norm(x)
    expected_dx, = torch.autograd.grad(expected, x, incoming)
    assert patch_qwen_rmsnorm()
    patched = Qwen2RMSNorm.forward
    assert patch_qwen_rmsnorm() and Qwen2RMSNorm.forward is patched
    actual = norm(x)
    actual_dx, = torch.autograd.grad(actual, x, incoming)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=2e-5, atol=scale * 3e-6)
    # Trainable norm and half input keep the genuine HF implementation.
    norm.weight.requires_grad_(True)
    y = norm(x)
    dw, = torch.autograd.grad(y.sum(), norm.weight)
    assert torch.isfinite(dw).all()
    norm.weight.requires_grad_(False)
    x_half = x.detach().half().requires_grad_(True)
    assert torch.equal(norm(x_half), reference_forward(norm, x_half))
