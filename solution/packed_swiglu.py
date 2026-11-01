"""Decode saved gate/up inside starter-derived backward instead of two full buffers.

The unchanged starter forward is used. Its rewritten backward is explicitly
excluded from author quota. Quantization bytes and FP16 reconstruction match the
existing INT4/64 codec. Offloaded gate/up return as compressed packets only;
no complete FP16 gate/up is materialized in this backward.
"""

from functools import lru_cache, wraps


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    class _PackedSwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up, gate_packet, up_packet):
            from .activation_offload import _Offloaded
            from .swiglu import BLOCK, example_swiglu_fwd_kernel

            output = torch.empty_like(gate)
            with torch.cuda.device(gate.device):
                example_swiglu_fwd_kernel[((gate.numel()+BLOCK-1)//BLOCK,)](
                    gate, up, output, gate.numel(), BLOCK=BLOCK,
                )
            values = (gate_packet, up_packet)
            packets = [value.packet if isinstance(value, _Offloaded) else value
                       for value in values]
            ctx.save_for_backward(packets[0].packed, packets[0].scales,
                                  packets[1].packed, packets[1].scales)
            ctx.originals = [(packet.original, packet.version) for packet in packets]
            ctx.transfers = [(value.device, value.event) if isinstance(value, _Offloaded)
                             else None for value in values]
            ctx.packets = values  # Keep packet identity dedup until backward.
            ctx.shape, ctx.elements = gate.shape, gate.numel()
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            from .activation_offload import _Offloaded
            from ._packed_swiglu_kernels import example_swiglu_packed_bwd_kernel

            for reference, version in ctx.originals:
                original = reference()
                if original is not None and original._version != version:
                    raise RuntimeError("a packed saved activation was modified in-place")
            storage = list(ctx.saved_tensors)
            for index, transfer in enumerate(ctx.transfers):
                if transfer is None:
                    continue
                device, ready = transfer
                value = ctx.packets[index] if ctx.packets is not None else None
                with torch.cuda.device(device):
                    if isinstance(value, _Offloaded) and value.prefetched_packed is not None:
                        torch.cuda.current_stream(device).wait_event(value.prefetch_event)
                        storage[2 * index] = value.prefetched_packed
                        storage[2 * index + 1] = value.prefetched_scales
                    else:
                        torch.cuda.current_stream(device).wait_event(ready)
                        storage[2 * index] = storage[2 * index].to(device, non_blocking=True)
                        storage[2 * index + 1] = storage[2 * index + 1].to(
                            device, non_blocking=True)
            gate_bytes, gate_scale, up_bytes, up_scale = storage
            incoming = incoming.to(torch.float16).contiguous()
            gate_grad = torch.empty(ctx.shape, device=incoming.device, dtype=torch.float16)
            up_grad = torch.empty_like(gate_grad)
            with torch.cuda.device(incoming.device):
                example_swiglu_packed_bwd_kernel[((ctx.elements+1023)//1024,)](
                    incoming, gate_bytes, gate_scale, up_bytes, up_scale,
                    gate_grad, up_grad, ctx.elements, block=1024, num_warps=4,
                )
            ctx.packets = None
            return gate_grad, up_grad, None, None

    return _PackedSwiGLU


def patch_packed_swiglu():
    import torch

    from . import swiglu as helper
    from .activation_offload import _Offloaded
    from .activation_storage import _active_codec_hooks, _Packet

    original = helper.swiglu
    if getattr(original, "_setww_packed_swiglu", False):
        return True

    @wraps(original)
    def forward(gate, up):
        hooks = _active_codec_hooks.get()
        supported = (hooks is not None and torch.is_grad_enabled() and gate.is_cuda
                     and up.device == gate.device and gate.shape == up.shape
                     and gate.dtype == up.dtype == torch.float16
                     and gate.shape[-1] == 8960 and gate.is_contiguous() and up.is_contiguous()
                     and gate.requires_grad and up.requires_grad and gate.numel() >= 262144)
        if not supported:
            return original(gate, up)
        gate_packet, up_packet = hooks[0](gate), hooks[0](up)
        def supported_packet(value):
            packet = value.packet if isinstance(value, _Offloaded) else value
            return (isinstance(packet, _Packet) and packet.bits == 4 and packet.block == 64
                    and (packet.packed.is_cuda or isinstance(value, _Offloaded)))

        if not all(supported_packet(value) for value in (gate_packet, up_packet)):
            return original(gate, up)
        return _autograd().apply(gate, up, gate_packet, up_packet)

    forward._setww_packed_swiglu = True
    helper.swiglu = forward
    return True
