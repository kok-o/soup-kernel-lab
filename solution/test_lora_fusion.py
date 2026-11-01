"""AMP rounding, dropout RNG, true NF4 gradients and real adapter updates."""

import pytest


@pytest.mark.parametrize("scale", [0.3, 2.0])
@pytest.mark.parametrize("incoming_scale", [1.0, 65536.0])
def test_scale_add_rounding_and_backward(scale, incoming_scale):
    import torch

    from solution.lora_fusion import fused_lora_add

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    base = torch.randn(513, 256, device="cuda", requires_grad=True)
    delta = torch.randn(513, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    incoming = torch.randn_like(base) * (1e-5 * incoming_scale)
    expected = base.clone() + delta * scale
    expected_grad = torch.autograd.grad(expected, (base, delta), incoming)
    actual = fused_lora_add(base, delta, scale)
    actual_grad = torch.autograd.grad(actual, (base, delta), incoming)
    assert actual.data_ptr() != base.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("incoming_scale", [1.0, 65536.0])
def test_real_nf4_lora_preserves_rng_gradients_and_update(incoming_scale):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit

    from solution.lora_fusion import patch_vanilla_nf4_lora

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    base = bnb.nn.Linear4bit(1536, 256, bias=True, compute_dtype=torch.float16,
                            compress_statistics=True, quant_type="nf4").cuda()
    base.weight.requires_grad_(False)
    base.bias.requires_grad_(False)
    layer = Linear4bit(base, "default", r=16, lora_alpha=32,
                      config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)).cuda().train()
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.01)
    parameters = [p for p in layer.parameters() if p.requires_grad]
    saved_parameters = [p.detach().clone() for p in parameters]
    original = Linear4bit.forward
    while hasattr(original, "__wrapped__") and getattr(original, "_setww_lora_scale_add", False):
        original = original.__wrapped__
    assert patch_vanilla_nf4_lora()
    source = torch.randn(4, 129, 1536, device="cuda")
    incoming = torch.randn(4, 129, 256, device="cuda") * (1e-5 * incoming_scale)
    rng = torch.cuda.get_rng_state()

    def execute(function):
        torch.cuda.set_rng_state(rng)
        for parameter, value in zip(parameters, saved_parameters, strict=True):
            parameter.data.copy_(value)
            parameter.grad = None
        x = source.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            output = function(layer, x)
        after = torch.cuda.get_rng_state()
        output.backward(incoming)
        gradients = [x.grad.detach().clone(), *[p.grad.detach().clone() for p in parameters]]
        torch.optim.SGD(parameters, lr=2e-4).step()
        return output.detach(), after, gradients, [p.detach().clone() for p in parameters]

    expected = execute(original)
    actual = execute(Linear4bit.forward)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert torch.equal(actual[1], expected[1])
    for a, b in zip(actual[2] + actual[3], expected[2] + expected[3], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
