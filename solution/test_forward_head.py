"""AMP scale must be applied before half rounding; unexpected scales recompute."""

import pytest
import torch


@pytest.mark.parametrize("incoming", [65536.0, 1.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_gradient_scaled_and_fallback(incoming, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from solution.chunked_head import chunked_head_loss

    torch.manual_seed(992)
    dtype = torch.float64 if device == "cpu" else torch.float32
    hidden = torch.randn(2, 19, 32, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(101, 32, device=device, dtype=dtype) * 0.1
    labels = torch.randint(0, 101, (2, 19), device=device)
    labels[:, :7] = -100
    labels[:, -3:] = -100
    mask = torch.ones_like(labels)
    mask[:, -3:] = 0
    kwargs = dict(attention_mask=mask, chunk_size=11, compact=True)
    with torch.autocast(device, enabled=device == "cuda", dtype=torch.float16):
        expected, expected_stats = chunked_head_loss(hidden, weight, labels, **kwargs)
        actual, actual_stats = chunked_head_loss(
            hidden, weight, labels, precompute_scale=65536.0, **kwargs
        )
    expected_dx, = torch.autograd.grad(expected, hidden, expected.new_tensor(incoming))
    actual_dx, = torch.autograd.grad(actual, hidden, actual.new_tensor(incoming))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_stats.token_loss, expected_stats.token_loss,
                               rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_stats.token_entropy, expected_stats.token_entropy,
                               rtol=2e-5, atol=2e-5)
    if device == "cpu":
        torch.testing.assert_close(actual_dx, expected_dx, rtol=1e-11, atol=1e-10)
    else:
        torch.testing.assert_close(actual_dx, expected_dx, rtol=2e-3,
                                   atol=abs(incoming) * 2e-6)
