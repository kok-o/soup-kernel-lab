"""Exact branch-wise FP16 rounding and FP32 accumulation, including tails."""

import pytest


@pytest.mark.parametrize("tokens", [37, 129, 512])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_fork_sum_matches_independent_casts(tokens, scale):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from .fork_sum import fork_gradient_sum

    gradients = [torch.randn(4, tokens, 1536, device="cuda", dtype=torch.float16)
                 * (1e-5 * scale) for _ in range(2)]
    expected = gradients[0].float() + gradients[1].float()
    with torch.no_grad():
        actual = fork_gradient_sum(*gradients, torch.float32)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32
