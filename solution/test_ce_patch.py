"""Metric semantics and actual CUDA Qwen/TRL integration without downloads."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from solution.ce import causal_ce
from solution.ce_patch import record_metrics


def fake_trainer():
    return SimpleNamespace(
        accelerator=SimpleNamespace(gather_for_metrics=lambda x: x, num_processes=1),
        _metrics=defaultdict(lambda: defaultdict(list)),
        _total_train_tokens=0,
    )


def test_entropy_uses_prompt_and_final_position_while_accuracy_does_not():
    logits = torch.zeros(2, 4, 5)
    logits[0, 0, 2] = 5
    logits[0, -1, 3] = 7
    labels = torch.tensor([[-100, -100, 0, 0], [-100] * 4])
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    _, stats = causal_ce(logits, labels)
    trainer = fake_trainer()
    record_metrics(trainer, stats, attention)
    p = logits.softmax(-1)
    expected_entropy = (-(p * logits.log_softmax(-1)).sum(-1) * attention).sum() / attention.sum()
    assert trainer._metrics["train"]["entropy"] == pytest.approx([expected_entropy.item()])
    assert trainer._metrics["train"]["mean_token_accuracy"] == [1.0]
    assert trainer._metrics["train"]["num_tokens"] == [6]
    record_metrics(trainer, stats, attention)
    assert trainer._metrics["train"]["num_tokens"] == [12]
    assert stats.row_losses()[1] == [2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and pinned Soup/TRL")
def test_qwen_trainer_metrics_rewind_and_updates(monkeypatch):
    import copy

    from soup_cli.trainer import rewind_hf
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import SFTTrainer

    from solution.ce_patch import patch_sft_ce

    saved_compute = SFTTrainer.compute_loss
    saved_rows = rewind_hf.row_losses
    # monkeypatch restores the real classes after this test, including if it fails.
    monkeypatch.setattr(SFTTrainer, "compute_loss", saved_compute)
    monkeypatch.setattr(rewind_hf, "row_losses", saved_rows)
    original_compute = getattr(saved_compute, "__wrapped__", saved_compute)
    original_rows = getattr(saved_rows, "__wrapped__", saved_rows)
    config = Qwen2Config(vocab_size=71, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    torch.manual_seed(72)
    baseline = Qwen2ForCausalLM(config).cuda().train()
    candidate = copy.deepcopy(baseline)
    labels = torch.randint(71, (2, 5), device="cuda")
    labels[:, :2] = -100
    ids = torch.randint(71, (2, 5), device="cuda")
    attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device="cuda")
    labels[1, 3:] = -100

    def trainer(model):
        result = object.__new__(SFTTrainer)
        for name, value in vars(fake_trainer()).items():
            setattr(result, name, value)
        result.model = model
        result.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll",
                                      average_tokens_across_devices=False, n_gpu=1)
        result.model_accepts_loss_kwargs = True
        result.label_smoother = None
        result.compute_loss_func = None
        result.num_virtual_tokens = 0
        result.aux_loss_enabled = False
        result.padding_free = False
        return result

    base_trainer, candidate_trainer = trainer(baseline), trainer(candidate)
    inputs = {"input_ids": ids, "labels": labels, "attention_mask": attention}
    count = (labels[:, 1:] != -100).sum()
    base_loss, base_out = original_compute(base_trainer, baseline, dict(inputs), True, count)
    patch_sft_ce()
    patched = SFTTrainer.compute_loss
    patch_sft_ce()
    assert SFTTrainer.compute_loss is patched
    loss, outputs = candidate_trainer.compute_loss(candidate, dict(inputs), True, count)
    torch.testing.assert_close(loss, base_loss, atol=3e-6, rtol=3e-6)
    for key in ("entropy", "num_tokens", "mean_token_accuracy"):
        assert candidate_trainer._metrics["train"][key] == pytest.approx(
            base_trainer._metrics["train"][key], abs=2e-6
        )
    expected_rows, counts = original_rows(base_out.logits, labels)
    actual_rows, actual_counts = rewind_hf.row_losses(outputs.logits, labels)
    assert counts == actual_counts
    assert actual_rows == pytest.approx(expected_rows, abs=3e-6)
    (loss * 8192).backward()
    (base_loss * 8192).backward()
    for base, ours in zip(baseline.parameters(), candidate.parameters(), strict=True):
        if base.grad is not None:
            torch.testing.assert_close(ours.grad, base.grad, atol=0.02, rtol=3e-4)


def test_cpu_model_and_fake_modules_use_original_path():
    from solution.ce_patch import _eligible

    trainer = fake_trainer()
    model = torch.nn.Linear(2, 2)
    assert not _eligible(trainer, model, {"labels": torch.zeros(1, 2, dtype=torch.int64)})
    assert not _eligible(trainer, SimpleNamespace(training=True), {})
