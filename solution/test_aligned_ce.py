"""Every chunk boundary retains its final real target and scaled derivative."""

import pytest
import torch
import torch.nn.functional as functional

from solution.ce import aligned_ce
from solution.chunked_head import adapt_aligned_ce, chunked_head_loss

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows", [1, 19])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_last_real_position_and_scaled_gradient(device, rows, dtype):
    torch.manual_seed(91)
    source = torch.randn(1, rows, 4103, device=device, dtype=dtype)
    actual_input = source.detach().clone().requires_grad_()
    baseline = source.detach().clone().requires_grad_()
    target = torch.randint(4103, (1, rows), device=device)
    if rows > 1:
        target[0, 0] = -100
    actual, stats = aligned_ce(actual_input, target, num_items_in_batch=17)
    reference = functional.cross_entropy(
        baseline.float().view(-1, 4103), target.flatten(), reduction="sum"
    ) / 17
    actual.backward(torch.tensor(8192.0, device=device))
    reference.backward(torch.tensor(8192.0, device=device))
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=2e-6)
    assert stats.valid[0, -1] and stats.token_loss[0, -1] > 0
    tolerance = 0.002 if dtype == torch.float16 else 3e-6
    torch.testing.assert_close(actual_input.grad, baseline.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_causal_sequences_split_across_arbitrary_chunks(device):
    torch.manual_seed(92)
    hidden = torch.randn(2, 7, 13, device=device, requires_grad=True)
    weight = torch.randn(37, 13, device=device)
    labels = torch.randint(37, (2, 7), device=device)
    labels[0, :4] = -100
    labels[1, :2] = -100
    reference = functional.cross_entropy(
        (hidden @ weight.t())[:, :-1].reshape(-1, 37), labels[:, 1:].reshape(-1)
    )
    expected_grad, = torch.autograd.grad(reference, hidden)
    for chunk in (1, 3, 5, 17):
        actual, _ = chunked_head_loss(
            hidden, weight, labels, chunk_size=chunk, backend=adapt_aligned_ce(aligned_ce)
        )
        gradient, = torch.autograd.grad(actual, hidden)
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=2e-6)
        torch.testing.assert_close(gradient, expected_grad, atol=2e-6, rtol=3e-6)
