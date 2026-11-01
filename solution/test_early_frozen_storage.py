"""No target upcast; all other PEFT preparation and FP32 embedding semantics match."""

import copy

import pytest


@pytest.mark.parametrize("checkpointing", [False, True])
def test_early_tied_storage_and_gradient_boundary(monkeypatch, checkpointing):
    import peft
    import torch
    from peft.utils import other
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from .early_frozen_storage import patch_early_frozen_storage
    from .frozen_storage import compact_frozen_tied_embedding

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        tie_word_embeddings=True, use_cache=False)
    model = Qwen2ForCausalLM(config).half().cuda()
    model.is_loaded_in_4bit = True
    reference = copy.deepcopy(model)
    native_prepare = other.prepare_model_for_kbit_training
    while getattr(native_prepare, "_setww_early_frozen_storage", False):
        native_prepare = native_prepare.__wrapped__
    native_prepare(reference, use_gradient_checkpointing=checkpointing)
    assert compact_frozen_tied_embedding(reference)
    assert patch_early_frozen_storage()
    target = model.get_input_embeddings().weight
    identity, pointer = id(target), target.data_ptr()
    native_to = torch.Tensor.to
    casts = []

    def observe(tensor, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if (tensor.is_cuda and tensor.data_ptr() == pointer and dtype == torch.float32):
            casts.append(True)
        return native_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observe)
    peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing)
    assert not casts
    assert model._setww_retained_frozen_half
    assert id(model.get_input_embeddings().weight) == identity
    assert model.get_output_embeddings().weight is target
    assert target.dtype == torch.float16 and target.data_ptr() == pointer
    assert "parameters" not in model.__dict__
    assert not any(parameter.requires_grad for parameter in model.parameters())
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name], rtol=0, atol=0)
        if parameter is not target:
            assert parameter.dtype == torch.float32
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]], device="cuda")
    actual = model.get_input_embeddings()(tokens)
    expected = reference.get_input_embeddings()(tokens)
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.requires_grad == expected.requires_grad
    assert torch.equal(actual, expected)
    if checkpointing:
        outputs = [module.get_output_embeddings()(hidden) for module, hidden in
                   ((model, actual), (reference, expected))]
        gradients = [torch.autograd.grad(output.square().sum(), hidden)[0]
                     for output, hidden in zip(outputs, (actual, expected), strict=True)]
        assert torch.equal(*gradients)
