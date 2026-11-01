# R2 + R1 v2: official T4 ABBA, negative admission result

Source: `3527e32abe8bdd92c447232250bb1a622fc7541e`.
Notebook: https://www.kaggle.com/code/narimanabdikarim/setww-r2-abba-3527e32/versions/2
Kaggle kernel ID 141689692, saved version 2. Torch 2.11.0+cu128, Triton 3.6.0.

The pinned dependency install was followed by removal of preinstalled optional
`torchao==0.10.0` BEFORE any measured runs: PEFT 0.20.0 rejects that old version.
Same environment for baseline and solution. Official harness/task/tests unchanged.

CUDA unit tests: **79 passed**. Full ABBA, two runs per side:

| Metric | Baseline | R1+R2 |
|---|---:|---:|
| Full command seconds, median | 85.82 | 81.75 |
| nvidia-smi peak MiB, max | 14345 | 8305 |
| Eval loss | 2.04964 | 2.04967 |

Speedup 1.0498x (under 1.05, so no speed points); memory ratio 1.7273x;
eval delta +0.001%. **9/10 official checks passed**; own-kernel share 0.8%
FAILS the >=5% admission requirement. All output metrics/rewind, loss curve,
quality, measurement and 79 Soup regressions passed. No overall admission claimed.

The source manifest matches the tested source. Raw result/checks are unmodified;
checks.result_sha256 was verified against the raw result.json. Raw logs and
rewind/trainer states are retained in r2-evidence.zip. No weights/data/caches.
This is the combination with the provided SwiGLU helper, not an isolated R2 gain.
