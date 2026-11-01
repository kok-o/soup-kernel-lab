"""Fused rotary values and gradients through actual T4 SDPA/repeated FP32 keys."""

import pytest


@pytest.mark.parametrize("tokens", [7, 129])
@pytest.mark.parametrize("factor_batches", [1, 2])
@pytest.mark.parametrize("incoming_scale", [1.0, 65536.0])
def test_rotary_sdpa_rounding_and_gradient(tokens, factor_batches, incoming_scale):
    import torch
    from torch.nn import functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from solution.rope_fusion import fused_rotary

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    query = torch.randn(2, tokens, 12, 128, device="cuda").transpose(1, 2).requires_grad_()
    key = torch.randn(2, tokens, 2, 128, device="cuda").transpose(1, 2).requires_grad_()
    value = torch.randn_like(key)
    angle = torch.randn(factor_batches, tokens, 128, device="cuda")
    cosine, sine = angle.cos(), angle.sin()
    incoming = torch.randn_like(query, dtype=torch.float16) * (1e-5 * incoming_scale)
    rng = torch.cuda.get_rng_state()

    def rotate(x):
        return torch.cat((-x[..., 64:], x[..., :64]), dim=-1)

    qref = (query*cosine[:, None] + rotate(query)*sine[:, None]).half()
    kref = key*cosine[:, None] + rotate(key)*sine[:, None]
    qactual, kactual = fused_rotary(query, key, cosine, sine)
    torch.testing.assert_close(qactual, qref, rtol=0, atol=0)
    torch.testing.assert_close(kactual, kref, rtol=0, atol=0)
    key_incoming = torch.linspace(-0.5, 0.5, key.numel(), device="cuda").reshape(key.shape)
    key_incoming = key_incoming * (1e-5 * incoming_scale)
    direct_expected = torch.autograd.grad((qref, kref), (query, key),
                                         (incoming, key_incoming), retain_graph=True)
    direct_actual = torch.autograd.grad((qactual, kactual), (query, key),
                                       (incoming, key_incoming), retain_graph=True)
    for a, b in zip(direct_actual, direct_expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def attention(q, k):
        repeated_key = k[:, :, None].expand(-1, -1, 6, -1, -1).reshape(2, 12, tokens, 128)
        repeated_value = value[:, :, None].expand(-1, -1, 6, -1, -1)
        repeated_value = repeated_value.reshape(2, 12, tokens, 128)
        with torch.autocast("cuda"), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return functional.scaled_dot_product_attention(q, repeated_key, repeated_value,
                                                           is_causal=True)

    reference = attention(qref, kref)
    expected_grad = torch.autograd.grad(reference, (query, key), incoming)
    actual = attention(qactual, kactual)
    actual_grad = torch.autograd.grad(actual, (query, key), incoming)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        # T4 efficient-SDPA split-key Q reductions can differ by FP16 ULPs.
        # The rotary-only gradients above must nevertheless match exactly.
        ulp = max(2**-24, torch.finfo(torch.float16).eps * b.abs().max().item())
        torch.testing.assert_close(a, b, rtol=0, atol=2*ulp)
    assert torch.equal(rng, torch.cuda.get_rng_state())
