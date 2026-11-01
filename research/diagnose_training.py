"""Two-step failure isolation. This is diagnostic, never a scored benchmark."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    import yaml
    from rich.console import Console

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from harness.common import ALLOC_CONF, OFFLINE_ENV

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", choices=(
        "off", "all", "mlp", "nonattention", "contiguous4d", "preserve4d"
    ), required=True)
    args = parser.parse_args()
    out = root / "diagnostics" / args.storage
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(OFFLINE_ENV)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
    os.environ["SOUP_DB_PATH"] = str(out / "experiments.db")
    os.environ["TRITON_CACHE_DIR"] = str(out / "triton_cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor_cache")
    config = yaml.safe_load((root / "runs_activation_pilot/base_1/task.yaml").read_text())
    config["output"] = str(out / "adapter")
    config_path = out / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    import torch
    from transformers import Trainer, TrainerCallback

    from solution import activation_storage

    console = Console()
    details = {"mode": args.storage, "bad_sources": [], "bad_decoded": [],
               "bad_gradients": [], "pack_groups": {}, "logs": []}
    current_module = []
    current_step = [0]
    original_factory = activation_storage.saved_activation_hooks

    if args.storage == "off":
        activation_storage.patch_activation_storage = lambda: False
    else:
        def monitored_hooks():
            pack, unpack = original_factory()

            def checked_pack(tensor):
                attention = (
                    tensor.is_cuda and tensor.ndim == 4 and tensor.requires_grad
                    and not tensor.is_leaf and tensor.numel() >= 262144
                    and tensor.dtype in (torch.float16, torch.float32)
                )
                if args.storage == "contiguous4d":
                    if attention:
                        key = f"layout-only {list(tensor.shape)} {list(tensor.stride())}"
                        details["pack_groups"][key] = details["pack_groups"].get(key, 0) + 1
                        return tensor.contiguous()
                    return tensor
                if args.storage == "preserve4d" and attention:
                    if any(size > 1 and stride == 0
                           for size, stride in zip(tensor.shape, tensor.stride())):
                        return tensor
                    key = f"preserve {list(tensor.shape)} {list(tensor.stride())}"
                    details["pack_groups"][key] = details["pack_groups"].get(key, 0) + 1
                    return (activation_storage.pack_activation(tensor), tensor.stride())
                if args.storage == "nonattention" and tensor.ndim not in (2, 3):
                    return tensor
                if args.storage == "mlp" and not (
                    tensor.ndim == 3 and tensor.shape[-1] == 8960
                ):
                    return tensor
                packet = pack(tensor)
                if isinstance(packet, activation_storage._Packet):
                    where = current_module[-1] if current_module else "?"
                    key = f"{where} {list(tensor.shape)} {tensor.dtype}"
                    details["pack_groups"][key] = details["pack_groups"].get(key, 0) + 1
                    if not torch.isfinite(tensor).all().item():
                        details["bad_sources"].append(key)
                    restored = unpack(packet)
                    if not torch.isfinite(restored).all().item():
                        details["bad_decoded"].append(key)
                return packet

            def checked_unpack(value):
                if isinstance(value, tuple):
                    packet, strides = value
                    dense = activation_storage.unpack_activation(packet)
                    restored = torch.empty_strided(
                        packet.shape, strides, dtype=packet.dtype, device=packet.packed.device
                    )
                    restored.copy_(dense)
                    return restored
                return unpack(value)

            return checked_pack, checked_unpack

        activation_storage.saved_activation_hooks = monitored_hooks

    class StopAfterTwo(TrainerCallback):
        def on_step_begin(self, _args, state, _control, **_kwargs):
            current_step[0] = state.global_step

        def on_step_end(self, _args, state, control, **_kwargs):
            if state.global_step >= 2:
                control.should_training_stop = True

        def on_log(self, _args, _state, _control, logs=None, **_kwargs):
            details["logs"].append(logs)

    original_init = Trainer.__init__

    def trainer_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        self.add_callback(StopAfterTwo())
        for name, module in self.model.named_modules():
            def enter(_module, _inputs, name=name):
                current_module.append(name)

            def leave(_module, _inputs, _output):
                current_module.pop()

            module.register_forward_pre_hook(enter)
            module.register_forward_hook(leave)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            def check_gradient(gradient, name=name):
                if not torch.isfinite(gradient).all().item():
                    details["bad_gradients"].append([current_step[0], name])
                return gradient

            parameter.register_hook(check_gradient)

    Trainer.__init__ = trainer_init
    sys.argv = ["_train.py", "--config", str(config_path),
                "--metrics", str(out / "metrics.json"), "--solution"]
    try:
        runpy.run_path(str(root / "harness/_train.py"), run_name="__main__")
    finally:
        (out / "diagnosis.json").write_text(json.dumps(details, indent=2))
        console.print({"mode": args.storage,
                       "bad_sources": details["bad_sources"][:5],
                       "bad_decoded": details["bad_decoded"][:5],
                       "bad_gradients": details["bad_gradients"][:5],
                       "saved_groups": len(details["pack_groups"]),
                       "logs": details["logs"]}, markup=False)


if __name__ == "__main__":
    main()
