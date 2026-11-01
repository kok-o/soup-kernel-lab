"""Pinned offload must preserve the quantized values and saved-tensor lifetime."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 8])
def test_offload_matches_gpu_reconstruction_and_deduplicates(bits):
    import gc
    import weakref

    from solution.activation_offload import offload_packed_hooks
    from solution.activation_storage import (
        pack_activation,
        saved_activation_hooks,
        unpack_activation,
    )

    torch.manual_seed(781)
    leaf = torch.randn(512, 1024, device="cuda", requires_grad=True)
    source = leaf * 1
    expected = unpack_activation(pack_activation(source, bits=bits))
    pack, unpack = saved_activation_hooks(bits=bits)
    pack, unpack = offload_packed_hooks(pack, unpack)
    packet = pack(source)
    assert pack(source) is packet
    assert packet.packet.packed.device.type == "cpu" and packet.packet.packed.is_pinned()
    reference = weakref.ref(source)
    del source
    gc.collect()
    assert reference() is None
    # Also exercise consumption on a different CUDA stream.
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        actual = unpack(packet)
    torch.cuda.current_stream().wait_stream(consumer)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert unpack(packet) is actual
    del actual
    gc.collect()
    again = unpack(packet)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)
