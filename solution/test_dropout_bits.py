"""Native dropout values/RNG, AMP gradients, tails, infinities and saved storage."""

import pytest


@pytest.mark.parametrize("tokens", [7, 193])
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
@pytest.mark.parametrize("incoming_scale", [1.0, 65536.0])
def test_native_dropout_rng_values_gradients_and_storage(tokens, dtype_name, incoming_scale):
    import torch
    from torch.nn import functional

    from solution.dropout_bits import packed_dropout

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.randn(4, tokens, 1536, device="cuda", dtype=getattr(torch, dtype_name))
    source.requires_grad_()
    incoming = torch.randn_like(source) * (1e-5 * incoming_scale)
    before = torch.cuda.get_rng_state()
    reference = functional.dropout(source, 0.05, training=True)
    after_reference = torch.cuda.get_rng_state()
    expected = torch.autograd.grad(reference, source, incoming)[0]
    torch.cuda.set_rng_state(before)
    saved = []

    def observe(tensor):
        saved.append((tensor.dtype, tensor.numel(), tensor.untyped_storage().nbytes()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(observe, lambda x: x):
        actual = packed_dropout(source, 0.05)
    actual_grad = torch.autograd.grad(actual, source, incoming)[0]
    assert torch.equal(torch.cuda.get_rng_state(), after_reference)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected, rtol=0, atol=0)
    assert saved == [(torch.uint8, (source.numel()+7)//8, (source.numel()+7)//8)]


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_masked_scale_tails_zero_sign_and_nonfinite_values(dtype_name):
    import torch

    from solution.dropout_bits import masked_gradient, pack_mask

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    incoming = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0,
                             0.0, -1.0, 1.0, 1e-5, 7.0], device="cuda",
                            dtype=getattr(torch, dtype_name)).repeat(9)[:73]
    mask = (torch.arange(73, device="cuda") % 3 != 0)
    scale = 1.0 / 0.95
    reference = torch.ops.aten.native_dropout_backward(incoming, mask, scale)
    actual = masked_gradient(incoming, pack_mask(mask), scale)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
    zero = reference == 0
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(reference[zero]))
