"""Keep native dropout RNG/forward, save only one bit per mask element.

Reference operation order: PyTorch v2.11.0 aten/src/ATen/native/cuda/Dropout.cu.
Only the task's out-of-place training dropout p=0.05 on contiguous [B,T,1536]
is patched. Masks are generated on the original full padded shape.
"""

from functools import lru_cache, wraps


def pack_mask(mask):
    import torch

    from ._dropout_kernels import dropout_mask_pack_kernel

    packed = torch.empty((mask.numel() + 7) // 8, device=mask.device, dtype=torch.uint8)
    if mask.numel():
        with torch.cuda.device(mask.device):
            dropout_mask_pack_kernel[((mask.numel() + 1023) // 1024,)](
                mask, packed, mask.numel(), groups=128, num_warps=4,
            )
    return packed


def masked_gradient(incoming, packed, scale):
    import torch

    from ._dropout_kernels import dropout_mask_gradient_kernel

    incoming = incoming.contiguous()
    output = torch.empty_like(incoming)
    if incoming.numel():
        with torch.cuda.device(incoming.device):
            dropout_mask_gradient_kernel[((incoming.numel() + 1023) // 1024,)](
                incoming, packed, output, incoming.numel(), scale=scale, block=1024,
                num_warps=4, enable_fp_fusion=False,
            )
    return output


@lru_cache(maxsize=1)
def _autograd():
    import torch
    from torch.autograd.function import once_differentiable

    class _PackedDropout(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source, probability):
            output, mask = torch.ops.aten.native_dropout(source, probability, True)
            ctx.save_for_backward(pack_mask(mask))
            ctx.scale = 1.0 / (1.0 - probability)
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, incoming):
            packed, = ctx.saved_tensors
            return masked_gradient(incoming, packed, ctx.scale), None

    return _PackedDropout


def packed_dropout(source, probability=0.05):
    return _autograd().apply(source, probability)


def patch_lora_dropout_masks():
    import torch

    original = torch.nn.Dropout.forward
    if getattr(original, "_setww_dropout_bits", False):
        return True

    @wraps(original)
    def forward(self, source):
        supported = (self.training and not self.inplace and self.p == 0.05
                     and torch.is_grad_enabled() and source.requires_grad and source.is_cuda
                     and source.dtype in (torch.float16, torch.float32)
                     and source.ndim == 3 and source.shape[-1] == 1536
                     and source.is_contiguous() and source.numel() > 0)
        return packed_dropout(source, self.p) if supported else original(self, source)

    forward._setww_dropout_bits = True
    torch.nn.Dropout.forward = forward
    return True
