"""Strict byte equivalence to two calls; failures are retained, not hidden by tolerance."""

import pytest


@pytest.fixture(autouse=True)
def deterministic_split_key_diagnostic():
    """Both native paths need fixed accumulation order for byte comparisons.

    The default split-key algorithm differs even from its own repeat in Q.
    Keep every original byte assertion; only tests disable that nondeterminism.
    Training does not alter this flag. Raw default-mode failures are retained.
    """
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize("lengths", [(127, 39, 191, 84), (512, 97, 58, 34),
                                    (317, 301, 248, 196), (511, 431, 293, 159)])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("incoming_scale", [1.0, 65536.0])
def test_one_call_values_all_qkv_gradient_bytes_and_rng(lengths, dense, incoming_scale):
    import torch

    from .grouped_ragged_attention import grouped_varlen_attention
    from .varlen_gqa import varlen_attention

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(1823)
    maximum, total = max(lengths), sum(lengths)
    lengths_cuda = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cumulative = torch.cat((lengths_cuda.new_zeros(1), lengths_cuda.cumsum(0, dtype=torch.int32)))
    valid = torch.arange(maximum, device="cuda")[None, :] < lengths_cuda[:, None]
    indices = valid.flatten().nonzero().flatten() if dense else None
    batch, tokens = (4, maximum) if dense else (1, total)
    inputs = [torch.randn(batch, tokens, heads, 128, device="cuda",
                          dtype=torch.float16 if heads == 12 else torch.float32)
              .transpose(1, 2).requires_grad_() for heads in (12, 2, 2)]
    incoming = torch.randn_like(inputs[0]) * (1e-5 * incoming_scale)
    before = torch.cuda.get_rng_state()
    expected = varlen_attention(*inputs, indices, cumulative, 128**-0.5, maximum)
    expected_grad = torch.autograd.grad(expected, inputs, incoming)
    after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(before)
    actual = grouped_varlen_attention(*inputs, indices, cumulative, 128**-0.5, maximum)
    actual_grad = torch.autograd.grad(actual, inputs, incoming)
    assert torch.equal(before, after) and torch.equal(after, torch.cuda.get_rng_state())
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))
    for result, reference in zip(actual_grad, expected_grad, strict=True):
        assert torch.equal(result.contiguous().view(torch.uint8),
                           reference.contiguous().view(torch.uint8))


def test_grouped_metadata_lives_only_in_current_layout_and_kv_expand_shares_storage():
    import torch

    from .compact_decoder import _current, _Layout
    from .grouped_ragged_attention import fold_groups, grouped_cumulative

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    indices = torch.arange(701, device="cuda")
    cumulative = torch.tensor([0, 512, 609, 667, 701], device="cuda", dtype=torch.int32)
    layout = _Layout((4, 512), indices, cumulative, packed=True)
    token = _current.set(layout)
    try:
        first = grouped_cumulative(cumulative, 701)
        assert grouped_cumulative(cumulative, 701) is first
        assert torch.equal(first, torch.tensor([0, 512, 609, 667, 701, 1213, 1310, 1368, 1402],
                                               device="cuda", dtype=torch.int32))
        key = torch.randn(1, 701, 2, 128, device="cuda", dtype=torch.float16)
        folded = fold_groups(key)
        expanded = folded.expand(-1, -1, 6, -1)
        assert expanded.stride(2) == 0
        assert expanded.untyped_storage().nbytes() == key.untyped_storage().nbytes()
    finally:
        _current.reset(token)
    second = grouped_cumulative(cumulative, 701)
    assert second is not first
