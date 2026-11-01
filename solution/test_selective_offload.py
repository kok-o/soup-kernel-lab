"""Exact codec parity across host transfer, layer boundaries and CUDA streams."""

import pytest


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
@pytest.mark.parametrize("tokens", [37, 129])
def test_selective_residency_and_streams(dtype_name, tokens):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from .activation_offload import _Offloaded
    from .activation_storage import _Packet, saved_activation_hooks
    from .selective_offload import _layer_index, patch_selective_offload, selective_packed_hooks

    patch_selective_offload(16)
    source = torch.randn(4, tokens, 8960, device="cuda", dtype=getattr(torch, dtype_name),
                         requires_grad=True)
    tensor = source * 0.25
    reference_pack, reference_unpack = saved_activation_hooks(bits=4, block=64, wide_only=True)
    expected = reference_unpack(reference_pack(tensor))
    pack, unpack = selective_packed_hooks(*saved_activation_hooks(
        bits=4, block=64, wide_only=True))
    token = _layer_index.set(0)
    try:
        transferred = pack(tensor)
    finally:
        _layer_index.reset(token)
    assert isinstance(transferred, _Offloaded)
    assert transferred.packet.packed.is_pinned()
    # Dedup must return the same wrapper even after crossing the residency boundary.
    token = _layer_index.set(27)
    try:
        assert pack(tensor) is transferred
        resident_tensor = source * 0.5
        resident = pack(resident_tensor)
    finally:
        _layer_index.reset(token)
    assert isinstance(resident, _Packet) and resident.packed.is_cuda
    token = _layer_index.set(0)
    try:
        assert pack(resident_tensor) is resident
    finally:
        _layer_index.reset(token)
    other_stream = torch.cuda.Stream()
    other_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other_stream):
        actual = unpack(transferred)
    torch.cuda.current_stream().wait_stream(other_stream)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert unpack(transferred) is actual
    assert transferred.packet.packed.device.type == "cpu"
    with torch.no_grad():
        tensor.add_(1)
    with pytest.raises(RuntimeError, match="modified in-place"):
        unpack(transferred)
