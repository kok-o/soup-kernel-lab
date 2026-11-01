"""Bulk H2D must preserve bytes, independent graphs and cross-stream consumers."""
import gc
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('bits', [4, 8])
def test_layer_slab_bulk_copy_exact_and_live_graphs(bits, monkeypatch):
    from . import activation_offload as ao
    from .activation_storage import saved_activation_hooks
    torch.manual_seed(1829)
    ao.reset_offload_step()
    monkeypatch.setattr(ao, '_PINNED_LAYER_SLABS', [[] for _ in range(28)])
    monkeypatch.setattr(ao, '_HOST_SLAB_BYTES', 8 * 2**20)
    allocations = {}
    # Direct allocator test: independent generations cannot reuse live views.
    x = ao._alloc_layer_pinned(0, allocations, (17,), torch.float32, 68)
    y = ao._alloc_layer_pinned(0, {}, (17,), torch.float32, 68)
    assert x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr()
    del x, y, allocations
    gc.collect()
    leaves = [torch.randn(513, 1536, device='cuda', requires_grad=True) for _ in range(3)]
    sources = [t*0.25 for t in leaves]
    pack_ref, unpack_ref = saved_activation_hooks(bits=bits)
    expected = [unpack_ref(pack_ref(t)) for t in sources]
    pack, unpack = ao.offload_packed_hooks(*saved_activation_hooks(bits=bits), layer_getter=lambda: 0)
    packets = [pack(t) for t in sources]
    assert all(isinstance(p, ao._Offloaded) for p in packets)
    assert len({p.packet.packed.untyped_storage().data_ptr() for p in packets}) == 1
    assert all(p.packet.scales.is_pinned() for p in packets)
    copies = []
    original_copy = torch.Tensor.copy_
    def count_copy(dst, src, *args, **kwargs):
        if dst.is_cuda and src.device.type == 'cpu':
            copies.append(src.numel()*src.element_size())
        return original_copy(dst, src, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', count_copy)
    ao.prefetch_layer(0)
    assert len(copies) == 1, copies
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        actual = [unpack(p) for p in packets]
    torch.cuda.current_stream().wait_stream(consumer)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with torch.no_grad():
        sources[0].add_(1)
    with pytest.raises(RuntimeError, match='modified in-place'):
        unpack(packets[0])
    ao.reset_offload_step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_small_slab_overflow_and_raw_bytes(monkeypatch):
    from . import activation_offload as ao
    ao.reset_offload_step()
    monkeypatch.setattr(ao, '_PINNED_LAYER_SLABS', [[] for _ in range(28)])
    monkeypatch.setattr(ao, '_HOST_SLAB_BYTES', 65536)
    allocations = {}
    a = ao._alloc_layer_pinned(1, allocations, (40000,), torch.uint8, 40000)
    b = ao._alloc_layer_pinned(1, allocations, (40000,), torch.uint8, 40000)
    assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    a.copy_(torch.arange(40000, dtype=torch.int64).to(torch.uint8))
    b.copy_(255-a)
    class Raw:
        pass
    one, two = Raw(), Raw()
    one.host_tensor, two.host_tensor = a, b
    gpu = torch.empty(80128, dtype=torch.uint8, device='cuda')
    ao._copy_prefetch_views(gpu, [(one, gpu[:40000], None), (two, gpu[40128:], None)])
    torch.testing.assert_close(gpu[:40000].cpu(), a, atol=0, rtol=0)
    torch.testing.assert_close(gpu[40128:].cpu(), b, atol=0, rtol=0)
