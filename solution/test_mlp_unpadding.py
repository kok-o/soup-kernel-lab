"""Token MLP sparsification must preserve real-token gradients and LoRA RNG."""

import pytest
import torch


def test_compact_mlp_masked_value_and_gradient():
    from solution.mlp_unpadding import compact_mlp

    torch.manual_seed(431)
    layer = torch.nn.Sequential(torch.nn.Linear(13, 27), torch.nn.SiLU(),
                                torch.nn.Linear(27, 13))
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(2, 7, 13, requires_grad=True)
    ids = torch.tensor([0, 1, 2, 7, 8, 9, 10])
    expected = layer(x).reshape(-1, 13).index_select(0, ids)
    reference_gradient, = torch.autograd.grad(expected.square().sum(), x)
    result = compact_mlp(lambda module, hidden: module(hidden), layer, x, ids)
    selected = result.reshape(-1, 13).index_select(0, ids)
    actual_gradient, = torch.autograd.grad(selected.square().sum(), x)
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(actual_gradient, reference_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA Qwen/PEFT required")
def test_real_qwen_lora_gradients_and_rng(monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution.mlp_unpadding import patch_tokenwise_mlp

    original_mlp = Qwen2MLP.forward
    while getattr(original_mlp, "_setww_unpadded_mlp", False):
        original_mlp = original_mlp.__wrapped__
    original_model = Qwen2ForCausalLM.forward
    while getattr(original_model, "_setww_unpadded_mlp", False):
        original_model = original_model.__wrapped__
    monkeypatch.setattr(Qwen2MLP, "forward", original_mlp)
    monkeypatch.setattr(Qwen2ForCausalLM, "forward", original_model)
    torch.manual_seed(417)
    config = Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "sdpa"
    model = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )).cuda().train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]], device="cuda")
    tokens = torch.randint(0, 101, mask.shape, device="cuda")
    labels = tokens.clone().masked_fill(mask.eq(0), -100)
    labels[:, 0] = -100
    rng = torch.cuda.get_rng_state()
    reference = model(input_ids=tokens, attention_mask=mask, labels=labels)
    reference.loss.backward()
    expected_rng = torch.cuda.get_rng_state()
    expected = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    assert patch_tokenwise_mlp()
    actual = model(input_ids=tokens, attention_mask=mask, labels=labels)
    actual.loss.backward()
    assert torch.equal(expected_rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual.loss, reference.loss, rtol=2e-6, atol=2e-6)
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=3e-4, atol=2e-6)
    assert torch.isfinite(actual.loss)
