"""Two-slot eviction/reload, pinned reuse, and cross-stream decoder reads."""

import gc

import pytest


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_two_slots_and_pinned_reuse(bits, dtype_name):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from .activation_storage import saved_activation_hooks
    from .pipelined_pool import _managers, pooled_offload_hooks

    device = torch.device("cuda", torch.cuda.current_device())
    # Isolate this test's pool without invalidating any live tensor from another test.
    _managers.clear()
    source = torch.randn(4, 129, 1536, device=device, dtype=getattr(torch, dtype_name),
                         requires_grad=True)
    pack, unpack = pooled_offload_hooks(*saved_activation_hooks(bits=bits, block=64),
                                       should_offload=lambda: True)
    tensors = [source * multiplier for multiplier in (0.25, 0.5, 0.75, 1.0)]
    reference_pack, reference_unpack = saved_activation_hooks(bits=bits, block=64)
    expected = [reference_unpack(reference_pack(tensor)) for tensor in tensors]
    packets = [pack(tensor) for tensor in tensors]
    manager = _managers[(device.type, device.index)]
    assert len(manager.slots) == 2
    pointers = [(slot.packed.data_ptr(), slot.scales.data_ptr()) for slot in manager.slots]
    other = torch.cuda.Stream()
    other.wait_stream(torch.cuda.current_stream())
    # Deliberately differ from reverse saving order to force speculative eviction.
    for index in (1, 3, 0, 2, 1):
        with torch.cuda.stream(other):
            actual = unpack(packets[index])
        torch.cuda.current_stream().wait_stream(other)
        torch.testing.assert_close(actual, expected[index], rtol=0, atol=0)
        del actual
    assert pointers == [(slot.packed.data_ptr(), slot.scales.data_ptr())
                        for slot in manager.slots]
    leases = len(manager.leases)
    packets.clear()
    gc.collect()
    second_pack, second_unpack = pooled_offload_hooks(
        *saved_activation_hooks(bits=bits, block=64), should_offload=lambda: True)
    again = second_pack(source * 0.25)
    torch.testing.assert_close(second_unpack(again), expected[0], rtol=0, atol=0)
    assert len(manager.leases) == leases
