"""True product/backward comparisons; large case has the actual Qwen head shape."""

import pytest
import torch
import torch.nn.functional as functional

from research.rejected_head_gemm.head_linear import frozen_linear


def test_cpu_and_trainable_fallback():
    torch.manual_seed(28)
    x = torch.randn(3, 7, requires_grad=True)
    w = torch.randn(11, 7, requires_grad=True)
    actual = frozen_linear(x, w)
    expected = functional.linear(x, w)
    torch.testing.assert_close(actual, expected)
    dx, dw = torch.autograd.grad(actual.sum(), (x, w))
    rx, rw = torch.autograd.grad(expected.sum(), (x, w))
    torch.testing.assert_close(dx, rx)
    torch.testing.assert_close(dw, rw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA FP16")
@pytest.mark.parametrize("shape", [(13, 33, 71), (128, 1536, 4103), (31, 1536, 151936)])
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_products_scaled_backward(shape, autocast):
    torch.manual_seed(29)
    rows, width, vocab = shape
    dtype = torch.float32 if autocast else torch.float16
    x = torch.randn((width, rows), device="cuda", dtype=dtype).t().requires_grad_()
    w = torch.randn((vocab, width), device="cuda", dtype=dtype) * width ** -0.5
    baseline = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = frozen_linear(x, w)
        expected = functional.linear(baseline, w)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.004)
    grad = torch.randn_like(actual) * (0.01 / vocab ** 0.5) * 8192
    actual.backward(grad)
    expected.backward(grad)
    assert x.grad.dtype == dtype
    torch.testing.assert_close(x.grad.float() / 8192, baseline.grad.float() / 8192,
                               atol=2e-5, rtol=0.004)
