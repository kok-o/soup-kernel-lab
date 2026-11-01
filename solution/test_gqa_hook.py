"""The actual Qwen2 attention forward must enter our registered shared-KV path."""


def test_qwen_attention_forward_uses_compact_saved_kv(monkeypatch):
    import torch
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    from . import gqa_native

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")
    config = Qwen2Config(hidden_size=1536, num_attention_heads=12, num_key_value_heads=2,
                        head_dim=128, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    attention = Qwen2Attention(config, layer_idx=0).cuda().train()
    assert gqa_native.patch_qwen_shared_kv()
    original = gqa_native.shared_kv_attention
    entered = []

    def observed(query, key, value, *args):
        entered.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
        return original(query, key, value, *args)

    monkeypatch.setattr(gqa_native, "shared_kv_attention", observed)
    hidden = torch.randn(4, 37, 1536, device="cuda", requires_grad=True)
    positions = torch.arange(37, device="cuda")
    causal = positions[None, :] <= positions[:, None]
    mask = torch.zeros((4, 1, 37, 37), device="cuda").masked_fill_(~causal, float("-inf"))
    angle = torch.randn(4, 37, 128, device="cuda")
    saved = []

    def record(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        with torch.autocast("cuda", dtype=torch.float16):
            output, _ = attention(
                hidden_states=hidden, position_embeddings=(angle.cos(), angle.sin()),
                attention_mask=mask)
    assert entered == [((4, 12, 37, 128), (4, 2, 37, 128), (4, 2, 37, 128))]
    # Only query is saved with twelve heads; native outputs retain six each.
    assert saved.count((4, 12, 37, 128)) == 1
    assert saved.count((4, 6, 37, 128)) == 2
    derivative, = torch.autograd.grad(output, hidden, torch.ones_like(output) * 1e-3)
    assert torch.isfinite(derivative).all()
