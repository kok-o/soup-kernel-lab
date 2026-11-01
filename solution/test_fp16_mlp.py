"""Actual NF4 MLP parity, preserving separate FP16-to-FP32 gradient casts."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/bnb required")
def test_nf4_mlp_forward_scaled_gradient_and_rng(monkeypatch):
    import bitsandbytes as bnb
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution.fp16_mlp import patch_fp16_mlp

    original = Qwen2MLP.forward
    while (getattr(original, "_setww_unpadded_mlp", False)
           or getattr(original, "_setww_fp16_mlp", False)):
        original = original.__wrapped__
    monkeypatch.setattr(Qwen2MLP, "forward", original)
    # This ablates the FP16 carrier patch against its actual prerequisite:
    # starter SwiGLU. Native torch SiLU has a different arithmetic schedule.
    # Explicit setup makes standalone `pytest solution` identical to the
    # combined suite, whose conftest already calls solution.apply(). Keep both
    # forward and AMP-scaled backward exact, without loosening tolerances.
    from solution.swiglu import patch_qwen2_mlp

    patch_qwen2_mlp()
    torch.manual_seed(961)
    config = Qwen2Config(hidden_size=1536, intermediate_size=8960, hidden_act="silu")
    layer = Qwen2MLP(config).cuda().train()
    for name, ins, outs in (("gate_proj", 1536, 8960), ("up_proj", 1536, 8960),
                           ("down_proj", 8960, 1536)):
        quantized = bnb.nn.Linear4bit(ins, outs, bias=False, compute_dtype=torch.float16,
                                     compress_statistics=True, quant_type="nf4").cuda()
        quantized.weight.requires_grad_(False)
        setattr(layer, name, quantized)
    hidden = torch.randn(2, 7, 1536, device="cuda", requires_grad=True)
    incoming = torch.randn_like(hidden) * 128
    rng = torch.cuda.get_rng_state()
    with torch.autocast("cuda", dtype=torch.float16):
        expected = layer(hidden)
    expected_dx, = torch.autograd.grad(expected, hidden, incoming)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert patch_fp16_mlp()
    with torch.autocast("cuda", dtype=torch.float16):
        actual = layer(hidden)
    actual_dx, = torch.autograd.grad(actual, hidden, incoming)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
    assert torch.equal(rng, torch.cuda.get_rng_state())
