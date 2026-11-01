"""Exact embedding/head values, original dtypes, input gradients and tied identity."""

import pytest
import torch

from solution.frozen_storage import compact_frozen_tied_embedding


def tiny_model(device):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(212)
    config = Qwen2Config(
        vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config).half().float().to(device)
    model.requires_grad_(False)
    model.enable_input_require_grads()
    model.train()
    return model


def test_cpu_keeps_original_parameter_storage():
    model = tiny_model("cpu")
    weight = model.get_input_embeddings().weight
    assert not compact_frozen_tied_embedding(model)
    assert weight.dtype == torch.float32
    assert model.get_output_embeddings().weight is weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("autocast", [False, True])
def test_exact_forward_and_input_gradients_with_original_dtype(autocast):
    model = tiny_model("cuda")
    embedding = model.get_input_embeddings()
    weight = embedding.weight
    expected_weight = weight.detach().clone()
    captured = {}

    def capture(_module, _inputs, output):
        captured["output"] = output
        output.retain_grad()

    embedding.register_forward_hook(capture)
    ids = torch.randint(37, (2, 7), device="cuda")
    labels = ids.clone()
    labels[:, :2] = -100
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        reference = model(input_ids=ids, labels=labels, use_cache=False)
    reference.loss.mul(8192).backward()
    expected_input = captured["output"].detach().clone()
    expected_gradient = captured["output"].grad.detach().clone()
    assert compact_frozen_tied_embedding(model)
    assert compact_frozen_tied_embedding(model)
    assert embedding.weight is weight and model.get_output_embeddings().weight is weight
    assert weight.dtype == torch.float16 and not weight.requires_grad
    assert torch.equal(weight.float(), expected_weight)
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = model(input_ids=ids, labels=labels, use_cache=False)
    actual.loss.mul(8192).backward()
    assert captured["output"].dtype == expected_input.dtype == torch.float32
    assert torch.equal(captured["output"], expected_input)
    assert torch.equal(actual.logits, reference.logits)
    assert torch.equal(captured["output"].grad, expected_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nonrepresentable_weight_is_not_changed():
    model = tiny_model("cuda")
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        weight[0, 0] = 0.123456789
    source = weight.detach().clone()
    assert not compact_frozen_tied_embedding(model)
    assert not compact_frozen_tied_embedding(model)
    assert weight.dtype == torch.float32 and torch.equal(weight, source)
