"""Actual Qwen/LoRA forwards must pack after dense dropout and retain its RNG."""


def test_qwen_lora_uses_varlen_after_dense_dropout(monkeypatch):
    import pytest
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from . import gqa_native, varlen_gqa
    from .mlp_unpadding import patch_tokenwise_mlp

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(714)
    config = Qwen2Config(vocab_size=103, hidden_size=1536, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                        tie_word_embeddings=True, use_cache=False, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = get_peft_model(Qwen2ForCausalLM(config).cuda(), LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"))
    model.train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.002)
    assert patch_tokenwise_mlp() and gqa_native.patch_qwen_shared_kv()
    tokens = torch.randint(0, config.vocab_size, (4, 37), device="cuda")
    lengths = torch.tensor([37, 28, 13, 8], device="cuda")
    valid = torch.arange(37, device="cuda")[None, :] < lengths[:, None]
    labels = tokens.masked_fill(~valid, -100)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial_rng = torch.cuda.get_rng_state()
    monkeypatch.setattr(gqa_native, "_varlen_enabled", False)
    with torch.autocast("cuda", dtype=torch.float16):
        reference = model(input_ids=tokens, attention_mask=valid, labels=labels).loss
    reference.backward()
    reference_rng = torch.cuda.get_rng_state()
    expected = [parameter.grad.clone() for parameter in parameters]
    model.zero_grad(set_to_none=True)
    entered = []
    original = varlen_gqa.varlen_attention

    def observed(query, key, value, *args):
        entered.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
        return original(query, key, value, *args)

    monkeypatch.setattr(varlen_gqa, "varlen_attention", observed)
    monkeypatch.setattr(gqa_native, "_varlen_enabled", True)
    torch.cuda.set_rng_state(initial_rng)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = model(input_ids=tokens, attention_mask=valid, labels=labels).loss
    actual.backward()
    assert entered == [((4, 12, 37, 128), (4, 2, 37, 128), (4, 2, 37, 128))]*2
    assert torch.equal(reference_rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual, reference, rtol=5e-4, atol=2e-5)
    for parameter, gradient in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-2, atol=5e-5)
