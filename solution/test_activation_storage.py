"""Storage error, saved-tensor lifetime, aliasing and unchanged-forward checks."""

import pytest
import torch

from solution.activation_storage import (
    pack_activation,
    saved_activation_hooks,
    unpack_activation,
)


def test_cpu_hooks_leave_values_and_gradients_exact():
    leaf = torch.randn(32, 64, requires_grad=True)
    value = leaf * 1.0
    pack, unpack = saved_activation_hooks(min_elements=1)
    assert pack(value) is value
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = value.square().sum()
    result.backward()
    torch.testing.assert_close(leaf.grad, 2 * leaf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(13, 37), (32, 1536), (257, 4096)])
def test_error_bound_and_noncontiguous_roundtrip(dtype, shape):
    torch.manual_seed(81)
    source = torch.randn(*reversed(shape), device="cuda", dtype=dtype).t()
    source[:, :2] = 0
    packet = pack_activation(source)
    actual = unpack_activation(packet)
    assert actual.shape == source.shape and actual.dtype == dtype
    assert packet.packed.numel() + packet.scales.numel() * 4 < source.numel() * 2
    groups = source.contiguous().flatten().float().split(256)
    error = (actual.float() - source.float()).flatten().split(256)
    for values, gap in zip(groups, error):
        bound = values.abs().max() / 254 + values.abs().max() * 0.001
        assert gap.abs().max() <= bound + 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hooks_deduplicate_restore_and_do_not_retain_original():
    import gc
    import weakref

    leaf = torch.randn(512, 1024, device="cuda", requires_grad=True)
    saved = leaf * 2
    pack, unpack = saved_activation_hooks()
    first = pack(saved)
    assert pack(saved) is first
    reference = weakref.ref(saved)
    del saved
    gc.collect()
    assert reference() is None
    restored = unpack(first)
    assert unpack(first) is restored
    torch.testing.assert_close(restored, leaf * 2, rtol=0, atol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_forward_exact_backward_uses_bounded_reconstruction():
    torch.manual_seed(82)
    leaf = torch.randn(512, 1024, device="cuda", requires_grad=True)
    source = leaf * 1.0
    pack, unpack = saved_activation_hooks()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual = source.square().sum()
    expected = source.square().sum()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert (leaf.grad - leaf * 2).norm() / (leaf * 2).norm() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mutating_a_live_saved_view_is_rejected():
    base = torch.randn(512, 1024, device="cuda")
    view = base.t()
    packet = pack_activation(view)
    del view
    base.add_(1)
    with pytest.raises(RuntimeError, match="modified in-place"):
        unpack_activation(packet)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_attention_and_parameters_keep_their_original_storage():
    parameter = torch.nn.Parameter(torch.randn(512, 1024, device="cuda"))
    attention = (parameter * 1).reshape(1, 4, 128, 1024)
    pack, unpack = saved_activation_hooks()
    assert pack(parameter) is parameter
    assert pack(attention) is attention
    assert unpack(attention) is attention
