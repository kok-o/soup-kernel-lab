"""Independent PyTorch comparisons; CUDA cases execute the actual Triton path."""

import pytest
import torch
import torch.nn.functional as functional

from solution.ce import causal_ce

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
))]


def reference(logits, labels, count=None):
    # Deliberately use the conventional S-1 view rather than the kernel's
    # padded S positions. Both must represent the same causal objective.
    losses = functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none",
    ).reshape(labels.shape[0], -1)
    denominator = (labels[:, 1:] != -100).sum() if count is None else count
    return losses.sum() / denominator, losses


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 7, 37), (2, 3, 4103), (1, 3, 151936)])
def test_loss_metrics_and_scaled_backward(device, dtype, shape):
    torch.manual_seed(71)
    source = torch.randn(shape, device=device, dtype=dtype)
    labels = torch.randint(shape[-1], shape[:2], device=device)
    labels[:, :2] = -100
    candidate = source.detach().clone().requires_grad_()
    baseline = source.detach().clone().requires_grad_()
    expected, token_loss = reference(baseline, labels)
    actual, stats = causal_ce(candidate, labels)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-6)
    torch.testing.assert_close(stats.token_loss[:, :-1], token_loss, atol=8e-6, rtol=2e-6)
    assert torch.equal(stats.token_loss[:, -1], torch.zeros_like(stats.token_loss[:, -1]))
    with torch.no_grad():
        probs = baseline.double().softmax(-1)
        entropy = -(probs * baseline.double().log_softmax(-1)).sum(-1).float()
        torch.testing.assert_close(stats.entropy, entropy, atol=8e-6, rtol=3e-6)
        correct = (source[:, :-1].argmax(-1) == labels[:, 1:]) & (labels[:, 1:] != -100)
        assert torch.equal(stats.correct[:, :-1], correct)
    scale = torch.tensor(32768.0, device=device)
    actual.backward(scale)
    expected.backward(scale)
    tolerance = 2e-3 if dtype == torch.float16 else 3e-6
    torch.testing.assert_close(candidate.grad, baseline.grad, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(candidate.grad[:, -1]) == 0
    assert not stats.token_loss.requires_grad and not stats.entropy.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_inputs_and_explicit_denominator(device):
    torch.manual_seed(82)
    # Neither token, vocabulary nor label layout may assume contiguous rows.
    source = torch.randn((2, 101, 9), device=device).transpose(1, 2)
    labels = torch.randint(101, (2, 18), device=device)[:, ::2]
    labels[1, 1:4] = -100
    candidate = source.detach().requires_grad_()
    baseline = source.detach().clone().requires_grad_()
    count = torch.tensor(53, device=device)
    actual, _ = causal_ce(candidate, labels, count)
    expected, _ = reference(baseline, labels, count)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(candidate.grad, baseline.grad, atol=2e-7, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_ties_large_offsets_empty_row_and_last_entropy(device):
    source = torch.zeros((3, 4, 4103), device=device)
    source[0] += 10000
    source[1, :, 4102] = 10000
    source[2, -1, 7] = 50
    labels = torch.tensor([[99, 0, 0, 0], [-100, 4102, 4102, 4102], [-100] * 4], device=device)
    loss, stats = causal_ce(source.requires_grad_(), labels)
    expected, _ = reference(source, labels)
    torch.testing.assert_close(loss, expected, atol=0.002, rtol=1e-5)
    assert stats.correct[0, :-1].all()  # first-index argmax across partitions
    assert stats.correct[1, :-1].all()
    means, counts = stats.row_losses()
    assert counts == [3, 3, 0] and means[2] == 0.0
    assert stats.entropy[2, -1] < stats.entropy[2, 0]
    assert not stats.correct[:, -1].any()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sequence", [1, 4])
def test_all_ignored_mean_and_zero_gradient(device, sequence):
    logits = torch.randn((2, sequence, 17), device=device, requires_grad=True)
    labels = torch.full((2, sequence), -100, device=device)
    loss, stats = causal_ce(logits, labels)
    assert torch.isnan(loss)
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert stats.row_losses() == ([0.0, 0.0], [0, 0])
    summed, _ = causal_ce(logits, labels, num_items_in_batch=1)
    assert summed.item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_real_lora_gradients_dropout_and_three_updates(device):
    torch.manual_seed(73)
    x = torch.randn((2, 5, 8), device=device)
    frozen_head = torch.randn((8, 41), device=device)
    labels = torch.randint(41, (2, 5), device=device)
    labels[:, :2] = -100
    initial = [torch.randn((8, 3), device=device) * 0.1,
               torch.randn((3, 8), device=device) * 0.1]
    runs = []
    for use_candidate in (False, True):
        parameters = [value.clone().requires_grad_() for value in initial]
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        history = []
        for step in range(3):
            torch.manual_seed(100 + step)
            dropped = functional.dropout(x, p=0.05, training=True)
            hidden = x + dropped @ parameters[0] @ parameters[1]
            logits = hidden @ frozen_head
            if use_candidate:
                loss, _ = causal_ce(logits, labels)
            else:
                loss, _ = reference(logits, labels)
            (loss * 8192).backward()
            gradients = [p.grad.detach().clone() / 8192 for p in parameters]
            for parameter in parameters:
                parameter.grad.div_(8192)
            assert all(torch.count_nonzero(g) > 0 and torch.isfinite(g).all() for g in gradients)
            optimizer.step()
            history.append((loss.detach(), gradients, [p.detach().clone() for p in parameters]))
            optimizer.zero_grad(set_to_none=True)
        runs.append(history)
    for base_step, candidate_step in zip(*runs, strict=True):
        torch.testing.assert_close(base_step[0], candidate_step[0], atol=2e-6, rtol=2e-6)
        for base, candidate in zip(
            base_step[1] + base_step[2], candidate_step[1] + candidate_step[2], strict=True
        ):
            torch.testing.assert_close(base, candidate, atol=2e-6, rtol=3e-6)


def test_validation():
    with pytest.raises(ValueError, match="expected logits"):
        causal_ce(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="int64"):
        causal_ce(torch.zeros(2, 4, 7), torch.zeros(2, 4))
    with pytest.raises(TypeError, match="FP16 and FP32"):
        causal_ce(torch.zeros(2, 4, 7, dtype=torch.float64), torch.zeros(2, 4, dtype=torch.int64))
