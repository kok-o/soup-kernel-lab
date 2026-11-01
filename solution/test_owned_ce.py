"""Owned logits reuse must preserve AMP gradients and reject repeated consumption."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_owned_ce_linear_gradients_and_lifetime(scale):
    from solution.ce import aligned_ce, owned_aligned_ce

    torch.manual_seed(971)
    hidden = torch.randn(37, 64, device="cuda", requires_grad=True)
    weight = torch.randn(257, 64, device="cuda") * 0.1
    targets = torch.randint(0, 257, (1, 37), device="cuda")
    targets[:, ::5] = -100
    with torch.autocast("cuda", dtype=torch.float16):
        reference_logits = torch.nn.functional.linear(hidden, weight).unsqueeze(0)
        owned_logits = torch.nn.functional.linear(hidden, weight).unsqueeze(0)
        expected, expected_stats = aligned_ce(reference_logits, targets)
        actual, actual_stats = owned_aligned_ce(owned_logits, targets)
    original = reference_logits.detach().clone()
    expected_dx, = torch.autograd.grad(expected, hidden, expected.new_tensor(scale))
    actual_dx, = torch.autograd.grad(actual, hidden, actual.new_tensor(scale), retain_graph=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(actual_stats.entropy, expected_stats.entropy, rtol=0, atol=0)
    assert torch.equal(original, reference_logits)
    assert not torch.equal(original, owned_logits)
    with pytest.raises(RuntimeError, match="one backward"):
        torch.autograd.grad(actual, hidden)
