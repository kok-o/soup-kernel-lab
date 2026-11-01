"""Compact decoder: native padded dropout, varlen gradients and actual LoRA updates."""

import pytest


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_compact_dropout_matches_padded_native_values_gradients_and_rng(dtype_name, requires_grad):
    import torch

    import solution

    from .compact_decoder import _current, _Layout

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    solution.apply()
    torch.manual_seed(675)
    dtype = getattr(torch, dtype_name)
    source = torch.randn(4, 67, 1536, device="cuda", dtype=dtype, requires_grad=requires_grad)
    lengths = torch.tensor([67, 53, 37, 29], device="cuda", dtype=torch.int32)
    valid = torch.arange(67, device="cuda")[None] < lengths[:, None]
    indices = valid.flatten().nonzero().flatten()
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    compact = source.reshape(-1, 1536).index_select(0, indices).unsqueeze(0)
    dropout = torch.nn.Dropout(0.05).train()
    incoming = torch.randn_like(compact)
    state = torch.cuda.get_rng_state()
    expected = dropout(source)
    expected_rng = torch.cuda.get_rng_state()
    if requires_grad:
        incoming_full = torch.zeros_like(source).view(-1, 1536)
        incoming_full.index_copy_(0, indices, incoming[0])
        expected_gradient = torch.autograd.grad(expected, source, incoming_full.view_as(source))[0]
    torch.cuda.set_rng_state(state)
    token = _current.set(_Layout((4, 67), indices, cumulative, packed=True))
    try:
        actual = dropout(compact)
    finally:
        _current.reset(token)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual[0], expected[valid], rtol=0, atol=0)
    if requires_grad:
        gradient = torch.autograd.grad(actual, compact, incoming)[0]
        torch.testing.assert_close(gradient[0], expected_gradient[valid], rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_already_packed_varlen_matches_gathered_path(scale):
    import torch

    from .varlen_gqa import varlen_attention

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    lengths = torch.tensor([67, 53, 37, 29], device="cuda", dtype=torch.int32)
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    tokens = 186
    values = [torch.randn(1, tokens, heads, 128, device="cuda",
                          dtype=torch.float16 if heads == 12 else torch.float32)
              .transpose(1, 2).requires_grad_() for heads in (12, 2, 2)]
    indices = torch.arange(tokens, device="cuda")
    incoming = torch.randn_like(values[0])*(1e-5*scale)
    rng = torch.cuda.get_rng_state()
    reference = varlen_attention(*values, indices, cumulative, 128**-0.5, max_sequence=67)
    expected = torch.autograd.grad(reference, values, incoming)
    actual = varlen_attention(*values, None, cumulative, 128**-0.5, max_sequence=67)
    gradients = torch.autograd.grad(actual, values, incoming)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for left, right in zip(gradients, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=5e-3, atol=max(2**-24, 5e-8*scale))


def test_compact_dropout_saves_only_mask_bits_and_no_float_scatter_sources():
    import torch

    from .compact_dropout import compact_native_dropout

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.randn(1, 701, 1536, device="cuda", requires_grad=True)*1.0
    indices = torch.arange(701, device="cuda")
    saved = []

    def observe(value):
        saved.append((value.dtype, value.shape))
        return value

    with torch.autograd.graph.saved_tensors_hooks(observe, lambda value: value):
        output = compact_native_dropout(source, indices, (4, 512), 0.05)
    assert saved == [(torch.uint8, torch.Size((701*1536//8,)))]
    gradient, = torch.autograd.grad(output, source, torch.ones_like(output))
    assert torch.isfinite(gradient).all()


def _model():
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=103, hidden_size=1536, intermediate_size=8960,
                        num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
                        tie_word_embeddings=True, use_cache=False, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    base = Qwen2ForCausalLM(config).half().cuda()
    for layer in base.model.layers:
        for container, name in ((layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
                                (layer.self_attn, "v_proj"), (layer.self_attn, "o_proj"),
                                (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                                (layer.mlp, "down_proj")):
            dense = getattr(container, name)
            linear = bnb.nn.Linear4bit(dense.in_features, dense.out_features,
                                      bias=dense.bias is not None, compute_dtype=torch.float16,
                                      compress_statistics=True, quant_type="nf4")
            linear.weight = bnb.nn.Params4bit(dense.weight.detach(), requires_grad=False,
                                             compress_statistics=True, quant_type="nf4",
                                             module=linear)
            if dense.bias is not None:
                linear.bias = torch.nn.Parameter(dense.bias.detach(), requires_grad=False)
            setattr(container, name, linear.cuda())
    base.is_loaded_in_4bit = True
    prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    return get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                          target_modules=["q_proj", "v_proj"],
                                          task_type="CAUSAL_LM")).train()


@pytest.mark.parametrize("embedding_grad", [False, True])
def test_full_nf4_qwen_lora_lossless_geometry_rng_and_finite_updates(monkeypatch, embedding_grad):
    import torch

    import solution

    from . import activation_storage, compact_decoder

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    solution.apply()
    # Isolate layout/autograd correctness from the existing lossy activation codec.
    # Changing GEMM M changes FP16 rounding; INT4 thresholds can amplify that
    # change in backward. Actual compressed training needs the full quality gate.
    monkeypatch.setattr(activation_storage, "saved_activation_hooks",
                        lambda **_kwargs: (lambda value: value, lambda value: value))
    torch.manual_seed(843)
    model = _model()
    if embedding_grad:
        model.enable_input_require_grads()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.002)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = [parameter.detach().clone() for parameter in parameters]
    tokens = torch.randint(1, 103, (4, 193), device="cuda")
    lengths = torch.tensor([193, 140, 100, 40], device="cuda")
    valid = torch.arange(193, device="cuda")[None] < lengths[:, None]
    labels = tokens.masked_fill(~valid, -100)
    labels[:, :5] = -100
    rng = torch.cuda.get_rng_state()
    layer_shapes, root_outputs = [], []

    def record_shape(_module, _args, output):
        layer_shapes.append(output.shape)

    def record_output(_module, _args, output):
        root_outputs.append(output.last_hidden_state.detach().clone())

    for layer in model.get_base_model().model.layers:
        layer.register_forward_hook(record_shape)
    model.get_base_model().model.register_forward_hook(record_output)

    def execute(enabled):
        monkeypatch.setattr(compact_decoder, "_enabled", enabled)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter, start in zip(parameters, initial, strict=True):
                parameter.copy_(start)
        optimizer = torch.optim.AdamW(parameters, lr=2e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=65536.0)
        torch.cuda.set_rng_state(rng)
        layer_shapes.clear()
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(input_ids=tokens, attention_mask=valid, labels=labels, use_cache=False)
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad.detach().clone() for parameter in parameters]
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        updates = [parameter.detach()-start
                   for parameter, start in zip(parameters, initial, strict=True)]
        return (output.loss.detach(), gradients, updates, torch.cuda.get_rng_state(),
                list(layer_shapes))

    reference = execute(False)
    actual = execute(True)
    assert torch.equal(reference[3], actual[3])
    assert reference[4] == [torch.Size((4, 193, 1536))]*2
    assert actual[4] == [torch.Size((473, 1536))]*2
    assert all(output.shape == (4, 193, 1536) for output in root_outputs)
    hidden_error = (root_outputs[1][valid]-root_outputs[0][valid]).norm()
    assert hidden_error <= 0.002*root_outputs[0][valid].norm()
    torch.testing.assert_close(actual[0], reference[0], rtol=1e-3, atol=2e-4)
    for observed, expected in zip(actual[1], reference[1], strict=True):
        error = (observed-expected).norm()
        assert error <= 0.005*expected.norm()+1e-7
    for observed, expected in zip(actual[2], reference[2], strict=True):
        assert expected.norm() > 0
        assert torch.isfinite(observed).all() and observed.norm() > 0
