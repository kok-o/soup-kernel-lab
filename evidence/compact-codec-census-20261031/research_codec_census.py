"""Count exact saved-view duplicates without retaining any GPU tensor/storage."""

import json
import os
import traceback
import weakref
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from pathlib import Path


def main():
    import torch

    import research_speed_profile
    import solution
    from solution import selective_offload
    from solution.activation_offload import _Offloaded
    from solution.activation_storage import _Packet

    original_factory = selective_offload.selective_packed_hooks
    counts = Counter()
    categories = Counter()
    callers = Counter()
    boundary_saves = Counter()
    boundaries = Counter()
    scope = ContextVar("census_consumer", default=())
    duplicate_samples = []
    batches = []
    live_duplicates = []
    peak_live_duplicate_gpu_bytes = 0

    def factory(pack, unpack):
        original_pack, original_unpack = original_factory(pack, unpack)
        seen = {}
        batch = {"pack_requests": 0, "eligible_requests": 0,
                 "same_packet_hits": 0, "distinct_view_duplicates": 0,
                 "possible_gpu_bytes": 0, "possible_host_bytes": 0}
        batches.append(batch)

        def observe(tensor):
            nonlocal peak_live_duplicate_gpu_bytes
            batch["pack_requests"] += 1
            result = original_pack(tensor)
            packet = result.packet if isinstance(result, _Offloaded) else result
            if "VARLEN_SAVED_QKV_OUTPUTS" in scope.get():
                boundary_saves[(tuple(tensor.shape), str(tensor.dtype),
                                isinstance(packet, _Packet))] += 1
            if not isinstance(packet, _Packet):
                return result
            batch["eligible_requests"] += 1
            # A weak live root plus a matching storage/view prevents allocator reuse
            # from being interpreted as aliasing. No strong GPU reference is stored.
            root = tensor._base if tensor._base is not None else tensor
            storage_identity = tensor.untyped_storage()._cdata
            placement = "host" if isinstance(result, _Offloaded) else "gpu"
            size = packet.packed.numel()*packet.packed.element_size()+packet.scales.numel()*4
            layer = selective_offload._layer_index.get()
            stack = traceback.extract_stack(limit=12)
            consumer = next((f"{Path(frame.filename).name}:{frame.name}:{frame.lineno}"
                            for frame in reversed(stack[:-1])
                            if "/solution/" in frame.filename), "native/framework")
            producer = type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None
            callers[(scope.get(), consumer, producer, tuple(tensor.shape),
                     str(tensor.dtype), placement)] += 1
            view = (str(tensor.device), storage_identity, tensor.storage_offset(),
                    tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
                    tensor._version, id(root), packet.bits, packet.block, placement)
            previous = seen.get(view)
            row = (weakref.ref(root), weakref.ref(packet), id(tensor))
            if previous is not None and previous[0]() is root:
                other = previous[1]()
                if other is packet:
                    batch["same_packet_hits"] += 1
                elif other is not None:
                    batch["distinct_view_duplicates"] += 1
                    batch["possible_"+("host" if placement == "host" else "gpu")+"_bytes"] += size
                    category = (layer, tuple(tensor.shape), tuple(tensor.stride()),
                                str(tensor.dtype), packet.bits, packet.block, placement)
                    categories[category] += 1
                    live_duplicates.append((weakref.ref(packet), placement, size))
                    if len(duplicate_samples) < 100:
                        duplicate_samples.append({"batch": len(batches)-1, "layer": layer,
                            "shape": list(tensor.shape), "stride": list(tensor.stride()),
                            "storage_offset": tensor.storage_offset(), "dtype": str(tensor.dtype),
                            "version": tensor._version, "bits": packet.bits,
                            "block": packet.block, "placement": placement,
                            "packet_bytes": size, "same_tensor_object": id(tensor) == previous[2],
                            "consumer_scope": scope.get(), "producer": producer,
                            "stack": [f"{Path(frame.filename).name}:{frame.name}:{frame.lineno}"
                                      for frame in stack]})
                else:
                    seen[view] = row
            else:
                seen[view] = row
            live_duplicates[:] = [entry for entry in live_duplicates if entry[0]() is not None]
            live_gpu = sum(entry[2] for entry in live_duplicates if entry[1] == "gpu")
            peak_live_duplicate_gpu_bytes = max(peak_live_duplicate_gpu_bytes, live_gpu)
            return result

        return observe, original_unpack

    selective_offload.selective_packed_hooks = factory
    from solution import varlen_gqa

    native_varlen = varlen_gqa.varlen_attention

    @wraps(native_varlen)
    def varlen(*args, **kwargs):
        boundaries[tuple((tuple(tensor.shape), str(tensor.dtype)) for tensor in args[:3])] += 1
        token = scope.set((*scope.get(), "VARLEN_SAVED_QKV_OUTPUTS"))
        try:
            return native_varlen(*args, **kwargs)
        finally:
            scope.reset(token)

    varlen_gqa.varlen_attention = varlen
    original_apply = solution.apply

    def apply():
        original_apply()
        from transformers import Trainer, TrainerCallback

        initialize = Trainer.__init__

        class Stop(TrainerCallback):
            def on_step_end(self, _args, state, control, **_kwargs):
                limit = int(os.environ.get("SETWW_CENSUS_STEPS", "0"))
                if limit and state.global_step >= limit:
                    control.should_training_stop = True

        @wraps(initialize)
        def init(self, *args, **kwargs):
            initialize(self, *args, **kwargs)
            for name, module in self.model.named_modules():
                if name.endswith(tuple("self_attn."+projection+"_proj" for projection in "qkvo")):
                    def before(_module, _args, label=name):
                        scope.set((*scope.get(), label))

                    def after(_module, _args, _output):
                        scope.set(scope.get()[:-1])

                    module.register_forward_pre_hook(before)
                    module.register_forward_hook(after, always_call=True)
            self.add_callback(Stop())

        Trainer.__init__ = init

    solution.apply = apply
    try:
        research_speed_profile.main()
    finally:
        for batch in batches:
            counts.update(batch)
        report = {"diagnostic_only": True, "runtime_commit": "c369cc7986127330b9aedeba655757d322b64559",
            "note": "Potential savings counts describe overlapping compressed packets, not measured NVIDIA peak savings. Same storage with different roots is conservatively excluded.",
            "batches": batches, "totals": dict(counts),
            "peak_live_duplicate_gpu_packet_mib": peak_live_duplicate_gpu_bytes/2**20,
            "eligible_pack_callers": [{"scope": key[0], "consumer": key[1], "producer": key[2],
                "shape": key[3], "dtype": key[4], "placement": key[5], "count": count}
                for key, count in callers.most_common()],
            "actual_varlen_input_boundaries": [{"tensors": key, "count": count}
                                               for key, count in boundaries.items()],
            "actual_varlen_saved_tensors": [{"shape": key[0], "dtype": key[1],
                "compressed": key[2], "count": count} for key, count in boundary_saves.items()],
            "duplicate_categories": [{"layer": key[0], "shape": key[1], "stride": key[2],
                "dtype": key[3], "bits": key[4], "block": key[5], "placement": key[6],
                "count": count} for key, count in categories.most_common()],
            "samples": duplicate_samples}
        Path(os.environ["SETWW_CODEC_CENSUS_REPORT"]).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
