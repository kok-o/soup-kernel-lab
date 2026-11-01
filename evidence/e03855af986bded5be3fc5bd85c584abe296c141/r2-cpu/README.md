# R2 CPU correctness evidence

Tested R2 source: `e03855af986bded5be3fc5bd85c584abe296c141`. R1 compatibility source: `00d00d6b678548851b4725c390bbd91152f411a3`.

- Standalone R2: **119 passed, 13 skipped**. Seven skips require R1 files; six require CUDA.
- Isolated R1 + R2 integration: **140 passed, 18 skipped**. All skips require CUDA.
- Both runs include all 79 unchanged Soup regression tests.
- Raw JUnit reports are preserved; package versions and source SHA-256 are in `provenance.json`.

The integration copy used unmodified R1 files plus the three R2 Python files.
R1 files were not committed into the R2 branch. Tests exercised actual tiny Qwen +
PEFT LoRA + TRL training and Soup rewind on CPU, including the R1 API adapter.

These are CPU correctness results, not T4 benchmarks. No GPU kernels, official
ABBA, nvidia-smi peak, held-out quality or >=5% own-time admission was measured.
No Kaggle notebook/version applies to this local CPU run.
