"""Packed real-token attention against the admitted dense, masked GQA path."""

import pytest


@pytest.mark.parametrize("tokens", [37, 512])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_varlen_two_groups_values_gradients_rng(tokens, scale):
    import torch

    from .gqa_native import _attention_bias, _autograd
    from .varlen_gqa import varlen_attention

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    lengths = torch.tensor([tokens, tokens*3//4, tokens//3, tokens//5], device="cuda",
                           dtype=torch.int32)
    valid = torch.arange(tokens, device="cuda")[None, :] < lengths[:, None]
    indices = valid.flatten().nonzero().flatten()
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    values = [torch.randn(4, tokens, heads, 128, device="cuda",
                          dtype=torch.float16 if heads == 12 else torch.float32)
              .transpose(1, 2).requires_grad_() for heads in (12, 2, 2)]
    positions = torch.arange(tokens, device="cuda")
    mask = valid[:, None, None, :] & (positions[None, :] <= positions[:, None])[None, None]
    incoming = torch.randn_like(values[0])*(1e-5*scale)
    incoming = incoming * valid[:, None, :, None]
    bias = _attention_bias(mask, torch.float16)
    expected = _autograd().apply(*values, bias, 128**-0.5)
    expected_gradients = torch.autograd.grad(expected, values, incoming)
    rng = torch.cuda.get_rng_state()
    actual = varlen_attention(*values, indices, cumulative, 128**-0.5)
    gradients = torch.autograd.grad(actual, values, incoming)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual.transpose(1, 2)[valid], expected.transpose(1, 2)[valid],
                               rtol=2e-3, atol=2e-4)
    assert not actual.transpose(1, 2)[~valid].count_nonzero()
    for left, right in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=5e-3, atol=max(2**-24, 5e-8*scale))
        assert torch.isfinite(left).all()
