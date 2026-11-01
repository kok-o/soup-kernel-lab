"""Grouped codec preserves the existing lossy format exactly, including tails."""

import pytest


@pytest.mark.parametrize("bits,block", [(4, 64), (4, 256), (8, 256)])
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
@pytest.mark.parametrize("elements", [137, 425*1536, 701*8960])
def test_grouped_codec_matches_original_bytes_scales_and_restoration(bits, block,
                                                                    dtype_name, elements):
    import torch
    import triton

    from . import _activation_kernels as old
    from . import _grouped_activation_kernels as new

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(716)
    source = torch.randn(elements, device="cuda", dtype=getattr(torch, dtype_name))
    source[:min(elements, block)].zero_()
    count = triton.cdiv(elements, block)
    packed = [torch.empty(elements if bits == 8 else triton.cdiv(elements, 2),
                          device="cuda", dtype=torch.int8 if bits == 8 else torch.uint8)
              for _ in range(2)]
    scales = [torch.empty(count, device="cuda", dtype=torch.float32) for _ in range(2)]
    restored = [torch.empty_like(source) for _ in range(2)]
    pack = old.activation_pack_kernel if bits == 8 else old.activation_pack4_kernel
    unpack = old.activation_unpack_kernel if bits == 8 else old.activation_unpack4_kernel
    pack[(count,)](source, packed[0], scales[0], elements, block=block, num_warps=4)
    unpack[(count,)](packed[0], scales[0], restored[0], elements, block=block, num_warps=4)
    grouped = (new.grouped_activation_pack8_kernel if bits == 8
               else new.grouped_activation_pack4_kernel)
    recover = (new.grouped_activation_unpack8_kernel if bits == 8
               else new.grouped_activation_unpack4_kernel)
    rng = torch.cuda.get_rng_state()
    for groups in (4, 8, 16, 32):
        grouped[(triton.cdiv(count, groups),)](source, packed[1], scales[1], elements,
                                               block=block, groups=groups, num_warps=4)
        recover[(triton.cdiv(elements, 1024),)](packed[1], scales[1], restored[1], elements,
                                                block=block, tile=1024, num_warps=4)
        torch.testing.assert_close(packed[1], packed[0], rtol=0, atol=0)
        torch.testing.assert_close(scales[1], scales[0], rtol=0, atol=0)
        torch.testing.assert_close(restored[1], restored[0], rtol=0, atol=0)
    assert torch.equal(rng, torch.cuda.get_rng_state())
