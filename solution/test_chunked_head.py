"""R2: loss, metrics, AMP gradient and frozen-head storage contracts.

CPU tests validate semantics. CUDA cases are required before accepting a T4 gain.
"""

import copy

import pytest
import torch
from torch.nn import functional


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference(hidden, weight, labels, denominator=None):
    target = functional.pad(labels, (0, 1), value=-100)[:, 1:]
    logits = functional.linear(hidden, weight)
    wide = logits if logits.dtype == torch.float64 else logits.float()
    token_loss = functional.cross_entropy(
        wide.flatten(0, 1), target.flatten(), reduction="none", ignore_index=-100
    ).view_as(target)
    logp = wide.log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    correct = (wide.argmax(-1) == target) & target.ne(-100)
    if denominator is None:
        denominator = target.ne(-100).sum()
    return token_loss.sum() / denominator, token_loss, entropy, correct


@pytest.mark.parametrize("chunk", [1, 5, 128, 256])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("scale", [0.125, 7.0, 65536.0])
def test_loss_metrics_and_scaled_gradient(chunk, compact, scale):
    from solution.chunked_head import chunked_head_loss

    torch.manual_seed(2)
    # Transposed storage tests non-contiguous hidden input.
    hidden = torch.randn(3, 11, 7, dtype=torch.float64).transpose(1, 2).requires_grad_()
    weight = torch.randn(37, 11, dtype=torch.float64)
    labels = torch.randint(37, (3, 7))
    labels[:, :2] = -100
    labels[1] = -100
    mask = torch.ones_like(labels)
    mask[0, -2:] = 0
    labels[0, -2:] = -100
    denominator = torch.tensor(23)
    ref_loss, ref_tokens, ref_entropy, ref_correct = reference(hidden, weight, labels, denominator)
    ref_grad, = torch.autograd.grad(ref_loss * scale, hidden)

    loss, step = chunked_head_loss(
        hidden, weight, labels, attention_mask=mask, chunk_size=chunk,
        num_items_in_batch=denominator, compact=compact,
    )
    grad, = torch.autograd.grad(loss * scale, hidden)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-12, atol=1e-10)
    torch.testing.assert_close(step.token_loss, ref_tokens, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        step.token_entropy[mask.bool()], ref_entropy[mask.bool()], rtol=1e-12, atol=1e-12
    )
    assert int(step.correct) == int(ref_correct.sum())
    assert int(step.labelled) == int(labels[:, 1:].ne(-100).sum())
    assert not step.token_loss.requires_grad


def test_gradcheck():
    from solution.chunked_head import chunked_head_loss

    torch.manual_seed(4)
    hidden = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 3, dtype=torch.float64)
    labels = torch.tensor([[-100, 1, 3, 2], [-100, -100, -100, -100]])
    assert torch.autograd.gradcheck(
        lambda h: chunked_head_loss(h, weight, labels, chunk_size=3)[0], (hidden,)
    )


def test_chunks_bound_logits_and_only_supervised_positions_recompute():
    from solution.chunked_head import chunked_head_loss, reference_terms

    seen = []
    def backend(logits, target, *, need_metrics):
        seen.append((tuple(logits.shape), need_metrics, int(target.ne(-100).sum())))
        return reference_terms(logits, target, need_metrics=need_metrics)

    hidden = torch.randn(2, 7, 4, requires_grad=True)
    weight = torch.randn(19, 4)
    labels = torch.tensor([[-100, -100, 2, 3, -100, -100, -100],
                           [-100, -100, -100, 4, 5, 6, 7]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    saved_shapes = []
    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss, _ = chunked_head_loss(
            hidden, weight, labels, attention_mask=mask, chunk_size=3,
            compact=True, backend=backend,
        )
    assert all(shape[-1:] != (19,) for shape in saved_shapes)
    loss.backward()
    assert all(shape[0] <= 3 for shape, _, _ in seen)
    assert sum(shape[0] for shape, metrics, _ in seen if metrics) == int(mask.sum())
    assert sum(shape[0] for shape, metrics, _ in seen if not metrics) == 6


@pytest.mark.parametrize("chunk", [128, 256])
def test_cpu_autocast_and_adapter_updates(chunk):
    from solution.chunked_head import chunked_head_loss

    torch.manual_seed(12)
    adapter = torch.nn.Sequential(torch.nn.Linear(8, 3, bias=False), torch.nn.Dropout(0.05),
                                  torch.nn.Linear(3, 8, bias=False))
    candidate = copy.deepcopy(adapter)
    weight = torch.randn(41, 8)
    x = torch.randn(2, 9, 8)
    labels = torch.randint(41, (2, 9))
    labels[:, :3] = -100
    labels[0, -2:] = -100
    ref_opt = torch.optim.AdamW(adapter.parameters(), lr=2e-4)
    opt = torch.optim.AdamW(candidate.parameters(), lr=2e-4)
    for _ in range(3):
        rng = torch.random.get_rng_state()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ref_hidden = x + adapter(x)
            ref_loss, *_ = reference(ref_hidden, weight, labels)
        (ref_loss * 65536).backward()
        torch.random.set_rng_state(rng)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            hidden = x + candidate(x)
            loss, _ = chunked_head_loss(hidden, weight, labels, chunk_size=chunk)
        loss.mul(65536).backward()
        for p, ref_p in zip(candidate.parameters(), adapter.parameters()):
            torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=1e-4)
            p.grad.div_(65536)
            ref_p.grad.div_(65536)
        opt.step()
        ref_opt.step()
        opt.zero_grad()
        ref_opt.zero_grad()
        for p, ref_p in zip(candidate.parameters(), adapter.parameters()):
            torch.testing.assert_close(p, ref_p, rtol=1e-6, atol=1e-7)


def test_cpu_fp16_recompute_receives_scale_before_rounding():
    from solution.chunked_head import chunked_head_loss

    torch.manual_seed(19)
    hidden = torch.randn(2, 9, 8, requires_grad=True)
    weight = torch.randn(257, 8) * 0.01
    labels = torch.randint(257, (2, 9))
    labels[:, :2] = -100
    denominator = 2 ** 22
    with torch.autocast("cpu", dtype=torch.float16):
        ref_loss, *_ = reference(hidden, weight, labels, denominator)
    scaled, = torch.autograd.grad(ref_loss * 65536, hidden)
    with torch.autocast("cpu", dtype=torch.float16):
        loss, _ = chunked_head_loss(hidden, weight, labels, chunk_size=5,
                                   num_items_in_batch=denominator)
    actual, = torch.autograd.grad(loss * 65536, hidden)
    assert torch.count_nonzero(scaled) > 0
    torch.testing.assert_close(actual, scaled, rtol=2e-3, atol=1e-7)


def test_ignored_batch_with_explicit_denominator():
    from solution.chunked_head import chunked_head_loss

    h = torch.randn(2, 4, 3, requires_grad=True)
    loss, step = chunked_head_loss(h, torch.randn(11, 3), torch.full((2, 4), -100),
                                   num_items_in_batch=7, chunk_size=2)
    loss.backward()
    assert loss == 0 and step.correct == 0 and step.labelled == 0
    assert torch.count_nonzero(h.grad) == 0
    assert torch.count_nonzero(step.token_loss) == 0


def test_default_denominator_is_shifted_target_count_and_shift_override():
    from solution.chunked_head import chunked_head_loss

    h = torch.randn(2, 4, 3, requires_grad=True)
    w = torch.randn(11, 3)
    labels = torch.tensor([[-100, -100, 3, 4], [-100, 5, 6, -100]])
    loss, _ = chunked_head_loss(h, w, labels, chunk_size=3)
    ref_loss, *_ = reference(h, w, labels)
    torch.testing.assert_close(loss, ref_loss)
    shifted = functional.pad(labels, (0, 1), value=-100)[:, 1:]
    same, _ = chunked_head_loss(h, w, labels, shift_labels=shifted, chunk_size=3)
    torch.testing.assert_close(loss, same)


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_invalid_chunk_rejected(chunk):
    from solution.chunked_head import chunked_head_loss

    with pytest.raises(ValueError, match="chunk"):
        chunked_head_loss(torch.randn(1, 2, 3), torch.randn(5, 3),
                          torch.zeros(1, 2, dtype=torch.long), chunk_size=chunk)


def test_trainable_head_rejected():
    from solution.chunked_head import chunked_head_loss

    with pytest.raises(ValueError, match="frozen"):
        chunked_head_loss(torch.randn(1, 2, 3), torch.randn(5, 3, requires_grad=True),
                          torch.zeros(1, 2, dtype=torch.long))


@pytest.mark.parametrize("chunk", [1, 5, 128, 256])
def test_actual_r1_causal_api_keeps_every_chunk_tail(chunk):
    from solution.chunked_head import adapt_causal_ce, chunked_head_loss
    causal_ce = pytest.importorskip("solution.ce", reason="R1 branch not integrated").causal_ce

    torch.manual_seed(23)
    h = torch.randn(2, 9, 8, requires_grad=True)
    w = torch.randn(37, 8)
    labels = torch.randint(37, (2, 9))
    labels[:, :2] = -100
    loss, step = chunked_head_loss(
        h, w, labels, chunk_size=chunk, backend=adapt_causal_ce(causal_ce),
        num_items_in_batch=53,
    )
    ref_loss, tokens, entropy, correct = reference(h, w, labels, 53)
    actual, = torch.autograd.grad(loss * 65536, h)
    expected, = torch.autograd.grad(ref_loss * 65536, h)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual / 65536, expected / 65536, rtol=3e-5, atol=2e-7)
    torch.testing.assert_close(step.token_loss, tokens)
    torch.testing.assert_close(step.token_entropy, entropy)
    assert step.correct == correct.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FP16")
@pytest.mark.parametrize("chunk", [128, 256])
def test_cuda_fp16_scaled_gradient(chunk):
    from solution.chunked_head import chunked_head_loss

    torch.manual_seed(21)
    h = torch.randn(2, 131, 64, device="cuda", requires_grad=True)
    w = torch.randn(1025, 64, device="cuda")
    labels = torch.randint(1025, (2, 131), device="cuda")
    labels[:, :19] = -100
    with torch.autocast("cuda", dtype=torch.float16):
        ref_loss, *_ = reference(h, w, labels)
    ref_grad, = torch.autograd.grad(ref_loss * 65536, h)
    with torch.autocast("cuda", dtype=torch.float16):
        loss, _ = chunked_head_loss(h, w, labels, chunk_size=chunk)
    grad, = torch.autograd.grad(loss * 65536, h)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(grad / 65536, ref_grad / 65536, rtol=3e-3, atol=2e-5)


@pytest.mark.parametrize("chunk", [5, 128, 256])
@pytest.mark.parametrize("backend", ["reference", "r1"])
def test_real_qwen_lora_training_metrics_rewind_and_rng(tmp_path, monkeypatch, chunk, backend):
    """Actual HF/TRL/PEFT training and Soup recorder, without model downloads."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from trl import SFTConfig, SFTTrainer

    from solution import chunked_head
    from solution.chunked_head import patch_chunked_head

    monkeypatch.setattr(chunked_head, "_ce_backend", chunked_head.reference_terms)
    if backend == "r1":
        causal_ce = pytest.importorskip("solution.ce", reason="R1 branch not integrated").causal_ce
        chunked_head.register_ce_backend(chunked_head.adapt_causal_ce(causal_ce))

    # Restore the class patches even if a numerical assertion fails.
    monkeypatch.setattr(Qwen2ForCausalLM, "forward", Qwen2ForCausalLM.forward)
    monkeypatch.setattr(SFTTrainer, "compute_loss", SFTTrainer.compute_loss)
    # Also obtain a real unpatched reference when another test called apply().
    while getattr(Qwen2ForCausalLM.forward, "_setww_r2", False):
        Qwen2ForCausalLM.forward = Qwen2ForCausalLM.forward.__wrapped__
    while getattr(SFTTrainer.compute_loss, "_setww_r2", False):
        SFTTrainer.compute_loss = SFTTrainer.compute_loss.__wrapped__
    torch.manual_seed(31)
    cfg = Qwen2Config(vocab_size=37, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     pad_token_id=0, eos_token_id=1, tie_word_embeddings=True,
                     attention_dropout=0.0, attn_implementation="eager")
    model = get_peft_model(
        Qwen2ForCausalLM(cfg),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                   target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    candidate = copy.deepcopy(model)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()
               if value.requires_grad}
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"t{i}": i for i in range(3, 37)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    dataset = Dataset.from_dict({
        "input_ids": [[3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13]],
        "labels": [[-100, -100, 5, 6, 7, 8, 9], [-100, -100, 12, 13]],
        "attention_mask": [[1] * 7, [1] * 4],
    })

    class Sink:
        def __init__(self):
            self.records = []
        def record_batch(self, **record):
            self.records.append(record)

    def trainer_for(which_model, name):
        args = SFTConfig(
            output_dir=str(tmp_path / name), use_cpu=True, bf16=False, fp16=False,
            max_steps=3, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            learning_rate=2e-4, lr_scheduler_type="cosine", optim="adamw_torch",
            gradient_checkpointing=False, logging_steps=1, save_strategy="no",
            report_to="none", disable_tqdm=True, seed=1,
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=7,
        )
        trainer = make_rewind_trainer_class(SFTTrainer)(
            model=which_model, args=args, train_dataset=dataset, processing_class=tokenizer,
        )
        sink = Sink()
        recorder = attach_rewind_state(trainer, sink)
        return trainer, sink, recorder

    baseline, base_sink, base_recorder = trainer_for(model, "baseline")
    torch.manual_seed(313)
    baseline.train()
    expected_rng = torch.random.get_rng_state()
    assert patch_chunked_head(chunk, compact=True)
    first_forward, first_compute = Qwen2ForCausalLM.forward, SFTTrainer.compute_loss
    assert patch_chunked_head(chunk, compact=True)
    assert Qwen2ForCausalLM.forward is first_forward
    assert SFTTrainer.compute_loss is first_compute
    ours, sink, recorder = trainer_for(candidate, "candidate")
    seen_logits = []
    hook = candidate.register_forward_hook(
        lambda _model, _inputs, outputs: seen_logits.append(outputs.logits)
    )
    torch.manual_seed(313)
    ours.train()
    hook.remove()
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    assert len(seen_logits) == 3 and all(logits is None for logits in seen_logits)
    assert ours._soup_chunked_step is not None
    assert not recorder.failed and recorder.dropped == 0
    assert not base_recorder.failed and base_recorder.dropped == 0
    assert len(sink.records) == len(base_sink.records) == 3
    for actual, expected in zip(sink.records, base_sink.records):
        for key in ("step", "micro", "rows", "row_tokens"):
            assert actual[key] == expected[key]
        torch.testing.assert_close(torch.tensor(actual["row_loss"]),
                                   torch.tensor(expected["row_loss"]), rtol=1e-6, atol=1e-6)
    base_logs = [log for log in baseline.state.log_history if "loss" in log]
    our_logs = [log for log in ours.state.log_history if "loss" in log]
    assert len(base_logs) == len(our_logs) == 3
    for actual, expected in zip(our_logs, base_logs):
        for key in ("loss", "entropy", "mean_token_accuracy", "num_tokens"):
            assert actual[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-5)
    base_parameters = dict(model.named_parameters())
    assert set(base_parameters) == set(dict(candidate.named_parameters()))
    for name, value in candidate.named_parameters():
        torch.testing.assert_close(value, base_parameters[name], rtol=1e-5, atol=2e-7)
    assert any(not torch.equal(value, base_parameters[name]) for name, value in initial.items())
    causal = candidate.get_base_model()
    assert causal.lm_head.weight is causal.model.embed_tokens.weight
    assert not causal.lm_head.weight.requires_grad
    # A training-mode call outside the SFT scope still returns ordinary logits.
    ids = torch.tensor([[3, 4, 5]])
    assert candidate(input_ids=ids).logits.shape == (1, 3, 37)
    candidate.eval()
    assert candidate(input_ids=ids, labels=ids).logits.shape == (1, 3, 37)


def test_patch_handles_fake_modules(monkeypatch):
    import sys
    import types

    from solution.chunked_head import patch_chunked_head

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace())
    assert patch_chunked_head() is False
