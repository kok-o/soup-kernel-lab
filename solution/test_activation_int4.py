"""INT4 is an experimental backward-storage format; quality needs full training."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block", [64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(13, 37), (257, 1536)])
def test_int4_tails_strides_and_error_bound(dtype, shape, block):
    from solution.activation_storage import pack_activation, unpack_activation

    torch.manual_seed(918)
    source = torch.randn(*reversed(shape), device="cuda", dtype=dtype).t()
    packet = pack_activation(source, bits=4, block=block)
    actual = unpack_activation(packet)
    assert packet.bits == 4 and packet.packed.numel() == (source.numel() + 1) // 2
    assert packet.packed.numel() + packet.scales.numel() * 4 < source.numel()
    for values, errors in zip(source.contiguous().flatten().float().split(block),
                              (actual.float() - source.float()).flatten().split(block)):
        bound = values.abs().max() / 14 + values.abs().max() * 0.002
        assert errors.abs().max() <= bound + 1e-7
    assert torch.isfinite(actual).all()
    assert unpack_activation(packet) is actual
