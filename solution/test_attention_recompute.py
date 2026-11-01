"""Selective attention recomputation preserves actual LoRA gradients and RNG."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("autocast", [False, True])
def test_attention_lora_gradients_and_rng(monkeypatch, autocast):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    from solution.attention_recompute import patch_attention_recompute

    original = Qwen2Attention.forward
    while getattr(original, "_setww_attention_recompute", False):
        original = original.__wrapped__
    monkeypatch.setattr(Qwen2Attention, "forward", original)
    torch.manual_seed(734)
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
    cpu_rng = torch.get_rng_state()
    with torch.autocast("cuda", enabled=autocast, dtype=torch.float16):
        reference = model(input_ids=tokens, attention_mask=mask, labels=labels)
    reference.loss.backward()
    expected_rng = torch.cuda.get_rng_state()
    expected_cpu_rng = torch.get_rng_state()
    expected = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    torch.set_rng_state(cpu_rng)
    assert patch_attention_recompute()
    with torch.autocast("cuda", enabled=autocast, dtype=torch.float16):
        actual = model(input_ids=tokens, attention_mask=mask, labels=labels)
    actual.loss.backward()
    assert torch.equal(expected_rng, torch.cuda.get_rng_state())
    assert torch.equal(expected_cpu_rng, torch.get_rng_state())
    torch.testing.assert_close(actual.loss, reference.loss, rtol=2e-6, atol=2e-6)
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=3e-4, atol=2e-6)
