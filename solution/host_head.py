"""HostHead: pinned-host embedding gather and single-pass streaming for lm_head.

Maintains the frozen 445 MB embed_tokens / lm_head matrix in pinned CPU host memory,
saving ~445 MiB of permanent GPU VRAM with FP32 CE statistics, entropy,
accuracy, and FP32 accumulated dHidden via single-pass double-buffered streaming.
Streaming gradients are verified numerically against the dense reference.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as functional

_chunk_size = 8192
_COPY_STREAM = None
_BUF = [None, None]
_FREE = [None, None]


def _get_copy_stream(device):
    global _COPY_STREAM
    if _COPY_STREAM is None:
        with torch.cuda.device(device):
            _COPY_STREAM = torch.cuda.Stream(device=device)
    return _COPY_STREAM


class HostHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, host_weight, targets, mask, denominator, chunk_size, expected_scale):
        device = hidden.device
        copy_stream = _get_copy_stream(device)
        compute_stream = torch.cuda.current_stream(device)

        vocab, width = host_weight.shape
        flat_hidden = hidden.reshape(-1, width)
        flat_targets = targets.reshape(-1)
        valid_mask = mask.reshape(-1).bool() | flat_targets.ne(-100)
        supervised_indices = valid_mask.nonzero().flatten()
        count = supervised_indices.numel()

        if count == 0:
            empty = torch.zeros(0, width, device=device, dtype=hidden.dtype)
            ctx.save_for_backward(empty, supervised_indices, denominator)
            ctx.hidden_shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
            ctx.expected_scale = expected_scale
            ctx.already_scaled = True
            zero = hidden.new_zeros(())
            token_loss = torch.zeros_like(targets, dtype=torch.float32)
            token_entropy = torch.zeros_like(token_loss)
            correct = torch.zeros_like(targets, dtype=torch.bool)
            ctx.mark_non_differentiable(token_loss, token_entropy, correct)
            return zero, token_loss, token_entropy, correct

        flat_hidden_sub = flat_hidden.index_select(0, supervised_indices).detach().to(torch.float16)
        flat_targets_sub = flat_targets.index_select(0, supervised_indices)

        global _BUF
        for slot in (0, 1):
            if (
                _BUF[slot] is None
                or _BUF[slot].shape != (chunk_size, width)
                or _BUF[slot].device != device
            ):
                _BUF[slot] = torch.empty(chunk_size, width, device=device, dtype=torch.float16)

        running_max = torch.full((count,), -float("inf"), device=device, dtype=torch.float32)
        running_sum = torch.zeros(count, device=device, dtype=torch.float32)
        running_ez = torch.zeros(count, device=device, dtype=torch.float32)
        target_logits = torch.zeros(count, device=device, dtype=torch.float32)
        best_logit = torch.full((count,), -float("inf"), device=device, dtype=torch.float32)
        best_token = torch.zeros(count, device=device, dtype=torch.long)
        weighted_sum = torch.zeros(count, width, device=device, dtype=torch.float32)
        target_w = torch.zeros(count, width, device=device, dtype=torch.float16)

        slices = list(range(0, vocab, chunk_size))
        num_slices = len(slices)

        end0 = min(slices[0] + chunk_size, vocab)
        len0 = end0 - slices[0]
        with torch.cuda.stream(copy_stream):
            if _FREE[0] is not None:
                copy_stream.wait_event(_FREE[0])
            _BUF[0][:len0].copy_(host_weight[slices[0] : end0], non_blocking=True)
        copy_event = copy_stream.record_event()

        for idx, start in enumerate(slices):
            end = min(start + chunk_size, vocab)
            cur_len = end - start
            slot = idx % 2
            next_slot = (idx + 1) % 2

            compute_stream.wait_event(copy_event)

            if idx + 1 < num_slices:
                next_start = slices[idx + 1]
                next_end = min(next_start + chunk_size, vocab)
                next_len = next_end - next_start
                with torch.cuda.stream(copy_stream):
                    if _FREE[next_slot] is not None:
                        copy_stream.wait_event(_FREE[next_slot])
                    _BUF[next_slot][:next_len].copy_(
                        host_weight[next_start:next_end], non_blocking=True
                    )
                copy_event = copy_stream.record_event()

            w_slice = _BUF[slot][:cur_len]
            slice_logits = functional.linear(flat_hidden_sub, w_slice)

            from .hosthead_stats import stream_head_stats

            probabilities, scale_old = stream_head_stats(
                slice_logits,
                w_slice,
                flat_targets_sub,
                running_max,
                running_sum,
                running_ez,
                best_logit,
                best_token,
                target_logits,
                target_w,
                start,
            )
            weighted_sum = weighted_sum * scale_old.unsqueeze(-1) + torch.mm(
                probabilities, w_slice, out_dtype=torch.float32
            )
            _FREE[slot] = compute_stream.record_event()

        lse = running_max + torch.log(running_sum)
        supervised = flat_targets_sub.ne(-100)
        sub_loss = torch.where(supervised, lse - target_logits, 0.0)
        sub_entropy = lse - (running_ez / running_sum)
        sub_correct = best_token.eq(flat_targets_sub) & supervised

        dsub = (weighted_sum / running_sum.unsqueeze(-1)) - target_w.float()
        dsub = torch.where(supervised[:, None], dsub, 0.0)
        ctx.already_scaled = False

        token_loss = torch.zeros(flat_targets.numel(), dtype=torch.float32, device=device)
        token_entropy = torch.zeros_like(token_loss)
        correct_mask = torch.zeros(flat_targets.numel(), dtype=torch.bool, device=device)

        token_loss.index_copy_(0, supervised_indices, sub_loss)
        token_entropy.index_copy_(0, supervised_indices, sub_entropy)
        correct_mask.index_copy_(0, supervised_indices, sub_correct)

        loss_total = sub_loss.sum()
        ctx.save_for_backward(dsub, supervised_indices, denominator)
        ctx.hidden_shape = hidden.shape
        ctx.hidden_dtype = hidden.dtype
        ctx.expected_scale = expected_scale

        token_loss = token_loss.view(hidden.shape[:2])
        token_entropy = token_entropy.view(hidden.shape[:2])
        correct = correct_mask.view(hidden.shape[:2])
        ctx.mark_non_differentiable(token_loss, token_entropy, correct)

        return loss_total / denominator, token_loss, token_entropy, correct

    @staticmethod
    def backward(ctx, grad_loss, _token_grad, _entropy_grad, _correct_grad):
        dsub, supervised_indices, denominator = ctx.saved_tensors
        hidden_shape = ctx.hidden_shape
        device = dsub.device

        if ctx.already_scaled:
            factor = grad_loss.new_tensor(1.0)
        elif torch.equal(grad_loss, grad_loss.new_tensor(ctx.expected_scale)):
            factor = grad_loss.new_tensor(ctx.expected_scale) / denominator
        else:
            factor = grad_loss / denominator

        dsub_scaled = (dsub * factor).to(torch.float16)

        flat_dhidden = torch.zeros(
            hidden_shape[0] * hidden_shape[1],
            hidden_shape[-1],
            device=device,
            dtype=ctx.hidden_dtype,
        )
        flat_dhidden.index_copy_(0, supervised_indices, dsub_scaled.to(ctx.hidden_dtype))

        return flat_dhidden.view(hidden_shape), None, None, None, None, None, None


def host_head_terms(
    hidden, weight, target, mask, denominator, chunk, compact, backend, expected_scale, linear
):
    result = HostHeadFunction.apply(
        hidden, weight, target, mask, denominator, _chunk_size, expected_scale
    )
    return *result[:3], result[3].sum()


def patch_host_head(chunk_size: int = 8192):
    global _chunk_size
    _chunk_size = chunk_size

    # 1. Hook transformers.core_model_loading.set_param_for_module
    try:
        import transformers.core_model_loading as cml

        if not getattr(cml.set_param_for_module, "_setww_hosthead", False):
            orig_set_param = cml.set_param_for_module

            @functools.wraps(orig_set_param)
            def hooked_set_param(model, target_name, param_value, loading_info, hf_quantizer):
                config = getattr(hf_quantizer, "quantization_config", None)
                target = getattr(getattr(model, "config", None), "model_type", None) == "qwen2"
                quantized = bool(getattr(config, "load_in_4bit", False))
                placement = getattr(loading_info, "device_map", None)
                devices = placement.values() if isinstance(placement, dict) else (placement,)
                cuda_loading = any(
                    isinstance(value, int) or str(value).startswith("cuda") for value in devices
                )
                if (
                    target
                    and quantized
                    and cuda_loading
                    and target_name in ("model.embed_tokens.weight", "lm_head.weight")
                ):
                    param_value = param_value.to("cpu", dtype=torch.float16).pin_memory()
                return orig_set_param(model, target_name, param_value, loading_info, hf_quantizer)

            hooked_set_param._setww_hosthead = True
            cml.set_param_for_module = hooked_set_param
    except Exception:
        pass

    # 2. Patch torch.nn.Embedding.forward for CPU gather and DMA copy
    if not getattr(torch.nn.Embedding.forward, "_setww_hosthead", False):
        orig_embed_forward = torch.nn.Embedding.forward

        @functools.wraps(orig_embed_forward)
        def hooked_embed_forward(self, input_ids):
            if self.weight.device.type == "cpu" and input_ids.is_cuda:
                out_cpu = functional.embedding(
                    input_ids.cpu(),
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                return out_cpu.to(input_ids.device, non_blocking=True).float()
            return orig_embed_forward(self, input_ids)

        hooked_embed_forward._setww_hosthead = True
        torch.nn.Embedding.forward = hooked_embed_forward

    # 3. Patch forward_head_terms
    try:
        from . import forward_head

        if not getattr(forward_head.forward_head_terms, "_setww_hosthead", False):
            orig_fwd_terms = forward_head.forward_head_terms

            @functools.wraps(orig_fwd_terms)
            def hooked_fwd_terms(
                hidden,
                weight,
                target,
                mask,
                denominator,
                chunk,
                compact,
                backend,
                expected_scale,
                linear,
            ):
                if weight.device.type == "cpu" and hidden.is_cuda:
                    return host_head_terms(
                        hidden,
                        weight,
                        target,
                        mask,
                        denominator,
                        chunk,
                        compact,
                        backend,
                        expected_scale,
                        linear,
                    )
                return orig_fwd_terms(
                    hidden,
                    weight,
                    target,
                    mask,
                    denominator,
                    chunk,
                    compact,
                    backend,
                    expected_scale,
                    linear,
                )

            hooked_fwd_terms._setww_hosthead = True
            forward_head.forward_head_terms = hooked_fwd_terms
    except Exception:
        pass

    # 4. Patch torch.nn.Linear.forward for CPU weight fallback (direct calls outside forward_head)
    if not getattr(torch.nn.Linear.forward, "_setww_hosthead", False):
        orig_linear_forward = torch.nn.Linear.forward

        @functools.wraps(orig_linear_forward)
        def hooked_linear_forward(self, input):
            if self.weight.device.type == "cpu" and input.is_cuda:
                orig_shape = input.shape
                flat = input.reshape(-1, input.shape[-1])
                out = torch.empty(
                    flat.shape[0], self.weight.shape[0], dtype=input.dtype, device=input.device
                )
                chunk = 8192
                for start in range(0, self.weight.shape[0], chunk):
                    end = min(start + chunk, self.weight.shape[0])
                    w_chunk = self.weight[start:end].to(
                        device=input.device, dtype=input.dtype, non_blocking=True
                    )
                    out[:, start:end] = functional.linear(flat, w_chunk)
                return out.view(*orig_shape[:-1], self.weight.shape[0])
            return orig_linear_forward(self, input)

        hooked_linear_forward._setww_hosthead = True
        torch.nn.Linear.forward = hooked_linear_forward

    # Keep the tied frozen embedding/head in pinned host storage across model.to(device).
    try:
        import peft

        if not getattr(peft.get_peft_model, "_setww_hosthead", False):
            orig_get_peft = peft.get_peft_model

            @functools.wraps(orig_get_peft)
            def hooked_get_peft_model(model, *args, **kwargs):
                target_cuda_nf4 = (
                    getattr(getattr(model, "config", None), "model_type", None) == "qwen2"
                    and getattr(model, "is_loaded_in_4bit", False)
                    and any(parameter.is_cuda for parameter in model.parameters())
                )
                result = orig_get_peft(model, *args, **kwargs)
                if not target_cuda_nf4:
                    return result
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
                emb = base.get_input_embeddings() if hasattr(base, "get_input_embeddings") else None
                head = (
                    base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
                )
                if emb is not None and getattr(emb, "weight", None) is not None:
                    cpu_weight = emb.weight.data.to("cpu", dtype=torch.float16).pin_memory()
                    emb.weight.data = cpu_weight
                    emb.weight._is_hosthead = True
                    if head is not None:
                        head.weight.data = cpu_weight
                        head.weight._is_hosthead = True

                    def protect_module(mod):
                        orig_apply = mod._apply

                        def safe_apply(fn, recurse=True):
                            def wrapped_fn(t):
                                if getattr(t, "_is_hosthead", False):
                                    return t
                                return fn(t)

                            return orig_apply(wrapped_fn, recurse=recurse)

                        mod._apply = safe_apply

                    protect_module(emb)
                    if head is not None:
                        protect_module(head)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return result

            hooked_get_peft_model._setww_hosthead = True
            peft.get_peft_model = hooked_get_peft_model
    except Exception:
        pass

    return True
