# R2 + R1 + original NF4 casts: admitted pilot (not final ABBA)

Tested source `80fb6e53b2c2d3dcc68c51ca43109e5d3e0816f1`, chunk 128.
Notebook https://www.kaggle.com/code/narimanabdikarim/setww-r2-abba-3527e32/versions/4
Kaggle Tesla T4, Torch 2.11.0+cu128, Triton 3.6.0.

- 93 CUDA/unit tests passed, including actual NF4 forward/backward parity.
- Single baseline/solution: 86.05 → 80.63 seconds; 14345 → 8305 MiB peak.
- Speed ratio 1.0672; memory ratio 1.7273; eval delta −0.001%.
- All 10 official checks passed, including 79 Soup regressions.
- Own-time share 10.5%, supplied SwiGLU excluded.

This is a one-pair pilot; the required selected-candidate two-repeat ABBA is
performed separately. New training_dtype_cast replaces real existing bnb
input/output conversions. It does not add redundant copies, busy work or change
compute dtype/precision. The model, task, metrics and adapter are unchanged.

Raw result/checks SHA binding verified. Raw outputs and manifest retained.
Optional incompatible torchao 0.10.0 is removed during environment preparation,
before both baseline and solution measurements. No packages change during runs.
