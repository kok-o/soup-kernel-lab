import pytest
import torch
import torch.nn.functional as functional

from solution.host_head import HostHeadFunction, patch_host_head


def test_embedding_cpu_gather_cuda_dma():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    patch_host_head()

    vocab_size = 1000
    hidden_size = 64
    batch_size = 2
    seq_len = 16

    cpu_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float16, pin_memory=True)
    gpu_weight = cpu_weight.cuda()

    emb = torch.nn.Embedding(vocab_size, hidden_size)
    emb.weight.data = cpu_weight

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    out_cand = emb(input_ids)
    out_ref = functional.embedding(input_ids, gpu_weight)

    assert out_cand.is_cuda
    assert torch.equal(out_cand, out_ref)


def test_host_head_autograd_against_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 16
    hidden_size = 64
    vocab_size = 512
    chunk_size = 128
    scale = 65536.0

    cpu_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float16, pin_memory=True) * 0.05
    gpu_weight = cpu_weight.to(device)

    hidden_ref = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.float16, requires_grad=True
    )
    hidden_cand = hidden_ref.detach().clone().requires_grad_(True)

    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels[:, :4] = -100
    labels[:, -2:] = -100
    mask = torch.ones_like(labels)
    mask[:, -2:] = 0

    supervised = labels.ne(-100)
    denominator = supervised.sum().float()

    # Reference
    with torch.autocast("cuda", dtype=torch.float16):
        ref_logits = functional.linear(hidden_ref, gpu_weight)
        ref_loss = (
            functional.cross_entropy(
                ref_logits.float().reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / denominator
        )
        ref_logp = ref_logits.float().log_softmax(-1)
        ref_entropy = -(ref_logp.exp() * ref_logp).sum(-1)
        ref_correct = (ref_logits.argmax(-1) == labels) & supervised

    ref_loss.backward(torch.tensor(scale, device=device))
    ref_grad = hidden_ref.grad

    # Candidate
    cand_loss, cand_token_loss, cand_entropy, cand_correct = HostHeadFunction.apply(
        hidden_cand, cpu_weight, labels, mask, denominator, chunk_size, scale
    )
    cand_loss.backward(torch.tensor(scale, device=device))
    cand_grad = hidden_cand.grad

    torch.testing.assert_close(cand_loss, ref_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        cand_entropy[supervised], ref_entropy[supervised], rtol=1e-3, atol=1e-3
    )
    assert (cand_correct == ref_correct).all()
    torch.testing.assert_close(cand_grad, ref_grad, atol=0.5, rtol=1e-2)
