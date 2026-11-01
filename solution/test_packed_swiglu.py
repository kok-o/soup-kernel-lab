"""Exact parity with unpack-then-starter backward and original version checks."""

import pytest


@pytest.mark.parametrize("positions", [37, 129, 831])
@pytest.mark.parametrize("incoming_scale", [1.0, 65536.0])
def test_packed_swiglu_matches_existing_codec_and_starter(positions, incoming_scale):
    import torch

    from solution import swiglu
    from solution.activation_storage import _active_codec_hooks, saved_activation_hooks
    from solution.packed_swiglu import patch_packed_swiglu

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(positions, 8960, device="cuda", dtype=torch.float16,
                       requires_grad=True) * 1
    up = torch.randn_like(gate, requires_grad=True) * 1
    incoming = torch.randn_like(gate) * (1e-4 * incoming_scale)
    assert patch_packed_swiglu()

    def execute(function):
        hooks = saved_activation_hooks(bits=4, block=64, wide_only=True)
        token = _active_codec_hooks.set(hooks)
        try:
            with torch.autograd.graph.saved_tensors_hooks(*hooks):
                output = function(gate, up)
        finally:
            _active_codec_hooks.reset(token)
        return output.detach(), torch.autograd.grad(output, (gate, up), incoming)

    reference = execute(swiglu.SwiGLU.apply)
    actual = execute(swiglu.swiglu)
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
    for a, b in zip(actual[1], reference[1], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("offload", [False, True])
def test_packed_swiglu_detects_mutated_original(offload):
    import torch

    from solution import swiglu
    from solution.activation_storage import _active_codec_hooks, saved_activation_hooks
    from solution.packed_swiglu import patch_packed_swiglu

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(37, 8960, device="cuda", dtype=torch.float16, requires_grad=True) * 1
    up = torch.randn_like(gate, requires_grad=True) * 1
    patch_packed_swiglu()
    hooks = saved_activation_hooks(bits=4, block=64, wide_only=True)
    if offload:
        from solution.activation_offload import offload_packed_hooks

        hooks = offload_packed_hooks(*hooks, min_bytes=0)
    token = _active_codec_hooks.set(hooks)
    try:
        with torch.autograd.graph.saved_tensors_hooks(*hooks):
            output = swiglu.swiglu(gate, up)
    finally:
        _active_codec_hooks.reset(token)
    with torch.no_grad():
        gate.add_(1)
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.sum().backward()


@pytest.mark.parametrize("positions", [37, 129, 831])
@pytest.mark.parametrize("incoming_scale", [1.0, 65536.0])
def test_offloaded_swiglu_keeps_packed_path_and_exact_gradient(positions, incoming_scale):
    import torch

    from solution import swiglu
    from solution.activation_offload import offload_packed_hooks
    from solution.activation_storage import _active_codec_hooks, saved_activation_hooks
    from solution.packed_swiglu import patch_packed_swiglu

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(positions, 8960, device="cuda", dtype=torch.float16,
                       requires_grad=True) * 1
    up = torch.randn_like(gate, requires_grad=True) * 1
    incoming = torch.randn_like(gate) * (1e-4 * incoming_scale)
    patch_packed_swiglu()

    def execute(function, packed):
        hooks = offload_packed_hooks(*saved_activation_hooks(bits=4, block=64, wide_only=True),
                                    min_bytes=0)
        token = _active_codec_hooks.set(hooks)
        try:
            with torch.autograd.graph.saved_tensors_hooks(*hooks):
                output = function(gate, up)
        finally:
            _active_codec_hooks.reset(token)
        if packed:
            assert output.grad_fn.__class__.__name__ == "_PackedSwiGLUBackward"
            buffers = output.grad_fn.saved_tensors
            assert len(buffers) == 4 and all(value.device.type == "cpu" for value in buffers)
            assert all(value.is_pinned() for value in buffers)
            assert all(value.dtype in (torch.uint8, torch.float32) for value in buffers)
        return output.detach(), torch.autograd.grad(output, (gate, up), incoming)

    reference = execute(swiglu.SwiGLU.apply, False)
    actual = execute(swiglu.swiglu, True)
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
    for left, right in zip(actual[1], reference[1], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
