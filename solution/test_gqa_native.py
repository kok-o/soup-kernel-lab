"""Native GQA values, FP32 KV reductions and saved-buffer sizes on T4."""

import pytest


@pytest.mark.parametrize("tokens", [37, 512])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_shared_kv_native_matches_repeated_keys(tokens, scale):
    import torch
    from torch.nn import functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from solution.gqa_native import shared_kv_attention

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    values = [torch.randn(4, tokens, heads, 128, device="cuda").transpose(1, 2)
              .requires_grad_() for heads in (12, 2, 2)]
    query, key, value = values
    positions = torch.arange(tokens, device="cuda")
    mask = positions[None, :] <= positions[:, None]
    mask = mask[None, None].expand(4, 1, tokens, tokens)
    incoming = torch.randn_like(query, dtype=torch.float16) * (1e-5 * scale)
    kr = key[:, :, None].expand(-1, -1, 6, -1, -1).reshape(4, 12, tokens, 128)
    vr = value[:, :, None].expand(-1, -1, 6, -1, -1).reshape(4, 12, tokens, 128)
    with torch.autocast("cuda"), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        reference = functional.scaled_dot_product_attention(query, kr, vr, attn_mask=mask)
    expected = torch.autograd.grad(reference, values, incoming)
    saved = []

    def observe(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype, tensor.untyped_storage().nbytes()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(observe, lambda x: x):
        actual = shared_kv_attention(query, key, value, mask, 128**-0.5)
    actual_grad = torch.autograd.grad(actual, values, incoming)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # Native Q reduction may differ by one FP16 ULP when the head batch is split.
    torch.testing.assert_close(actual_grad[0], expected[0], rtol=2e-3,
                               atol=max(2**-24, 3e-8*scale))
    for a, b in zip(actual_grad[1:], expected[1:], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.dtype == torch.float32
    full_heads = [entry for entry in saved if entry[0] == (4, 12, tokens, 128)]
    assert len(full_heads) == 1, saved  # Query only, never repeated K/V or a full saved output.
    pieces = [entry for entry in saved if entry[0] == (4, 6, tokens, 128)]
    assert len(pieces) == 2, saved
    assert sum(entry[2] for entry in pieces) == query.numel() * 2
