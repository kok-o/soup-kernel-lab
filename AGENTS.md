# SetWW / Soup Kernel Challenge

This is kok-o's independent post-hackathon research repository. Never modify the original hackathon repository or its Kaggle notebooks. The historical baseline is preserved in BASELINE_PROVENANCE.json. GPU runs require the user's current resource authorization.
Read README.md, docs/TEAM.md and docs/RESEARCH.md before changing the solution.

- Runtime and all implementation dependencies needed by the jury must live in `solution/`;
  the jury copies only that directory into the official starter.
- Preserve `harness/`, `task/`, `tests/` exactly as imported. Do not change benchmark settings.
- The source of truth for this fixed task is `task/task.yaml` plus the official starter README
  preserved in `docs/STARTER_GUIDE.md`. Do not change model, data, batches/order, steps, LoRA,
  optimizer, allocator or built-in Soup switches.
- R1: CE/metrics. R2: chunked head. R3: RMSNorm. Keep work in separate branches and modules.
  Coordinate edits to `solution/__init__.py`; `apply()` must be narrow and idempotent.
- Keep entropy, token accuracy, shifted assistant loss and per-row rewind semantics.
  Check backward and real parameter updates, not loss alone.
- Write original Triton kernels; cite idea sources. Do not copy forbidden libraries.
  The starter SwiGLU and its rewrites do not count toward author kernels or the в‰Ґ5% quota.
- Use the official notebook on Kaggle T4 FP16 for accepted measurements.
  Do not fabricate results or count microbench/GPU time as full-command time.
- Required checks: `python -m pytest solution -q`, `python harness/run_task.py --repeats 2`,
  `python harness/check_rules.py` (without skip-tests); all ten official checks must pass.
- Save measured JSON and provenance under `evidence/`; keep caches, models, adapters and secrets out.
- The dependency install and full run instructions are in `docs/RUNBOOK.md`.
