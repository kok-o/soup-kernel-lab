"""The packed lease ends before dX GEMM, with bit-exact native gradients."""
from contextlib import contextmanager
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_backward_releases_packed_slot_before_matmul(monkeypatch):
    import bitsandbytes.functional as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from .stream_nf4 import HostNF4, NF4Slots, streamed_linear

    class ObservedSlots(NF4Slots):
        active = False

        @contextmanager
        def acquire(self, host):
            with super().acquire(host) as value:
                self.active = True
                try:
                    yield value
                finally:
                    self.active = False

    torch.manual_seed(36191)
    weight = torch.randn(256, 128, device='cuda', dtype=torch.float16)*0.02
    packed, state = bnb.quantize_4bit(weight, blocksize=64, compress_statistics=True, quant_type='nf4')
    host = HostNF4.capture(packed, state)
    pool = ObservedSlots(max_elements=256*128)
    data = torch.randn(71, 128, device='cuda', dtype=torch.float16)*0.02
    incoming = torch.randn(71, 256, device='cuda', dtype=torch.float16)*0.01
    reference_input = data.detach().requires_grad_()
    reference = MatMul4Bit.apply(reference_input, packed, None, None, state)
    reference.backward(incoming)
    actual_input = data.detach().requires_grad_()
    actual = streamed_linear(actual_input, host, pool)
    original = torch.matmul
    gemm_seen = []

    def observe(a, b, *args, **kwargs):
        gemm_seen.append(tuple(b.shape))
        assert not pool.active, 'Packed bytes must be reusable before dense dX GEMM'
        return original(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, 'matmul', observe)
    actual.backward(incoming)
    assert gemm_seen == [(256, 128)]
    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    assert torch.equal(actual_input.grad.view(torch.int16), reference_input.grad.view(torch.int16))
