"""Exact NF4 values, refresh after mutation, state preservation and cache lifetime."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/bnb")
@pytest.mark.parametrize("shape", [(257, 128), (8960, 1536)])
def test_exact_nested_dequant_and_version_refresh(monkeypatch, shape):
    import bitsandbytes.functional as functional

    from research.nested_scale_cache.nested_scales import patch_nested_scale_cache

    original = functional.dequantize_4bit
    while getattr(original, "_setww_nested_scales", False):
        original = original.__wrapped__
    monkeypatch.setattr(functional, "dequantize_4bit", original)
    torch.manual_seed(123)
    packed, state = functional.quantize_4bit(
        torch.randn(*shape, device="cuda", dtype=torch.float16),
        quant_type="nf4", compress_statistics=True,
    )
    expected = original(packed, state)
    assert patch_nested_scale_cache()
    patched = functional.dequantize_4bit
    assert patch_nested_scale_cache() and functional.dequantize_4bit is patched
    actual = patched(packed, state)
    assert torch.equal(actual, expected)
    first = patched._setww_scale_cache.get(state)
    assert patched._setww_scale_cache.get(state) is first
    assert state.nested and state.absmax.dtype == torch.uint8
    assert not first.nested and first.absmax.dtype == torch.float32
    assert torch.equal(patched(A=packed, quant_state=state), expected)
    state.absmax[0].bitwise_xor_(1)
    assert torch.equal(patched(packed, state), original(packed, state))
    assert patched._setww_scale_cache.get(state) is not first
    state.state2.absmax.mul_(0.99)
    assert torch.equal(patched(packed, state), original(packed, state))
    state.offset.add_(0.001)
    assert torch.equal(patched(packed, state), original(packed, state))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/bnb")
def test_cache_does_not_retain_canonical_state():
    import gc
    import weakref

    import bitsandbytes.functional as functional

    from research.nested_scale_cache.nested_scales import NestedScaleCache

    _, state = functional.quantize_4bit(
        torch.randn(256, 128, device="cuda", dtype=torch.float16),
        quant_type="nf4", compress_statistics=True,
    )
    cache = NestedScaleCache(functional)
    cache.get(state)
    identity = weakref.ref(state)
    assert len(cache.entries) == 1
    del state
    gc.collect()
    assert identity() is None and not cache.entries
