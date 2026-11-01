"""Observe actual training masks without changing official files or solution code.

Run as a diagnostic wrapper of harness/_train.py with its ordinary CLI arguments.
Instrumentation adds synchronization, so its wall time is never a score claim.
"""

import json
import os
import runpy
from functools import wraps
from pathlib import Path


def main():
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    repo = Path(__file__).absolute().parents[1]
    destination = Path(os.environ["SETWW_PADDING_REPORT"])
    original = Qwen2Model.forward
    observations = []

    @wraps(original)
    def observe(self, *args, **kwargs):
        mask = kwargs.get("attention_mask")
        if (self.training and torch.is_grad_enabled() and mask is not None
                and mask.ndim == 2):
            lengths = mask.ne(0).sum(-1).detach().cpu().tolist()
            observations.append({
                "batch": len(observations), "shape": list(mask.shape),
                "valid": sum(lengths), "total": mask.numel(), "lengths": lengths,
            })
        return original(self, *args, **kwargs)

    Qwen2Model.forward = observe
    try:
        runpy.run_path(str(repo / "harness" / "_train.py"), run_name="__main__")
    finally:
        Qwen2Model.forward = original
        valid = sum(row["valid"] for row in observations)
        total = sum(row["total"] for row in observations)
        report = {
            "diagnostic_only": True, "source_commit": "bef6d7a99779593bcd80d43df013ce34561d55b9",
            "batches": len(observations), "valid_tokens": valid, "padded_slots": total,
            "padding_fraction": 1 - valid / total if total else None,
            "observations": observations,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
