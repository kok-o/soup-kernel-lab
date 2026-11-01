"""Numerical gates for streamed head statistics, including tails and ties."""

import pytest
import torch

from .hosthead_stats import stream_head_stats


@pytest.mark.parametrize("rows,width,chunk,tail", [(7, 32, 64, 17), (17, 1536, 8192, 4480)])
def test_streamed_stats_matches_torch(rows, width, chunk, tail):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(1234)
    targets = torch.arange(rows, device="cuda", dtype=torch.int64) * 31
    targets[0] = -100
    targets[-1] = chunk + tail - 1

    def state():
        return [
            torch.full((rows,), -float("inf"), device="cuda"),
            torch.zeros(rows, device="cuda"),
            torch.zeros(rows, device="cuda"),
            torch.full((rows,), -float("inf"), device="cuda"),
            torch.zeros(rows, dtype=torch.int64, device="cuda"),
            torch.zeros(rows, device="cuda"),
            torch.zeros(rows, width, dtype=torch.float16, device="cuda"),
        ]

    reference, actual = state(), state()
    for start, cols in ((0, chunk), (chunk, tail)):
        z = torch.randn(rows, cols, device="cuda", dtype=torch.float16) * 3
        z[1, :2] = 20  # A tie within and across vocabulary chunks.
        w = torch.randn(cols, width, device="cuda", dtype=torch.float16) * 0.02
        m, s, ez, best, best_token, target_z, target_w = reference
        mx, arg = z.max(dim=-1)
        new_m = torch.maximum(m, mx.float())
        scale = torch.exp(m - new_m)
        e = torch.exp(z.float() - new_m[:, None])
        s.copy_(s * scale + e.sum(dim=-1))
        ez.copy_(ez * scale + (z.float() * e).sum(dim=-1))
        m.copy_(new_m)
        update = mx.float() > best
        best.copy_(torch.where(update, mx.float(), best))
        best_token.copy_(torch.where(update, arg + start, best_token))
        inside = (targets >= start) & (targets < start + cols)
        ids = inside.nonzero().flatten()
        target_z[ids] = z[ids, targets[ids] - start].float()
        target_w[ids] = w[targets[ids] - start]
        prob, actual_scale = stream_head_stats(z, w, targets, *actual, start)
        torch.testing.assert_close(actual_scale, scale, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(prob.float(), e.half().float(), rtol=1e-3, atol=6e-5)
        for a, b in zip(actual[:4], reference[:4]):
            torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-5)
        assert torch.equal(actual[4], reference[4])
        assert torch.equal(actual[5], reference[5])
        assert torch.equal(actual[6], reference[6])
    actual_entropy = actual[0] + actual[1].log() - actual[2] / actual[1]
    reference_entropy = reference[0] + reference[1].log() - reference[2] / reference[1]
    torch.testing.assert_close(actual_entropy, reference_entropy, rtol=2e-5, atol=2e-5)
