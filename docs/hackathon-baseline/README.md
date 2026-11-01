# SetWW — Soup Kernel Challenge

**61.40 seconds full cold training · 729 MiB GPU peak · 10/10 official checks**

An optimized Qwen2.5-1.5B-Instruct LoRA training implementation for the Soup Kernel Challenge.
The final result was measured on Kaggle Tesla T4 with the original 2+2 ABBA harness.

[Public Kaggle notebook](https://www.kaggle.com/code/koko667/setww-final-slotviews?scriptVersionId=364323222) · [Submission details](SUBMISSION.md) · [Raw evidence](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/tree/5d10568bd847d2586bb67ad01852e77a28b4d6fd/evidence/kaggle-slotviews-20261031) · [Exact measured source](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/commit/652334f3c63764d8ca998ef3c9ae1c378465cf63)

## Measured result

| Original Kaggle T4 ABBA | Base Soup | Solution | Ratio |
|---|---:|---:|---:|
| Full cold soup train wall, s | 84.34 | 61.40 | 1.3736x |
| External nvidia-smi peak, MiB | 14345 | 729 | 19.6776x |
| Training loop only, s | 57.44 | 38.33 | separate from full cold |
| Public held-out eval loss | 2.04965 | 2.05214 | +0.121% |

All 10 original checks pass. Patched tests: 370 passed, 2 original skips; official check 10 passes 79 regressions.
Counted own kernels: 8.3% of the training time (lowest of 2 runs, steps 3+). Raw JSON/logs/source: [evidence](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/tree/5d10568bd847d2586bb67ad01852e77a28b4d6fd/evidence/kaggle-slotviews-20261031).
Raw ZIP SHA256: `f9f33be19b21c093fa74855e38e4c898ecd87d589955ac5ab9ddbab7ff531b7e`. Canonical solution SHA256: `3b8e9eff479025995004eb34dab2db10f27fcd3f440f015021f8af3db1a78f6a`.


The two solution runs were **61.38 s and 61.42 s**, both at **729 MiB**.
The headline is the entire cold training process, including imports, model loading and JIT compilation.
The 38.33 s training loop is reported separately. Relative to the same-session baseline,
the solution is **1.374x faster** and uses **19.68x less peak GPU memory**.

## How it works

1. **Stream frozen NF4 weights.** All 196 frozen projections and their exact quantization state remain in pinned host memory. Two fixed GPU buffers stage weights when needed, with CUDA events protecting transfers and readers.
2. **Reuse metadata, not resident weights.** A bounded cache reuses QuantState objects and tensor views that alias those two buffers. Every requested weight is still transferred; this removes repeated Python setup without allocating extra GPU weight storage.
3. **Bound vocabulary-head memory.** The shared embedding/head storage stays on the host; chunked vocabulary computation preserves the loss, entropy, accuracy and input-gradient outputs.
4. **Balance activation memory and transfer time.** Activations from the first 20 decoder layers are offloaded, while eight late layers remain resident. Packed activation and dropout-mask storage reduce live GPU memory.
5. **Reduce redundant work.** Compact valid tokens inside each existing batch, grouped ragged attention, and authored Triton kernels handle normalization, RoPE, activation codecs and LoRA scale/add operations.

The original dataset/order, 64 steps, dropout/RNG, LoRA configuration, optimizer, metrics and official harness remain unchanged. GPU tests check native NF4 forward/backward parity, buffer aliasing and mutation detection. No prebuilt compiler caches or saved model outputs are shipped.

Frozen NF4 weights retain their exact native values. Some saved activations use INT4/INT8 storage and are reconstructed for backward, so the complete training trajectory is not claimed to be bitwise identical to the baseline. The original quality checks pass, including the measured +0.121% eval-loss difference.

## Repository guide

| Path | Purpose |
|---|---|
| `solution/` | Complete implementation copied by the jury; all runtime helpers are included |
| `notebooks/final-qualification/` | Exact submitted notebook and 96-file source hash manifest |
| `evidence/kaggle-slotviews-20261031/` | Final raw archive, results, official checks and source verification |
| `harness/`, `task/`, `tests/` | Unmodified official measurement, task and regression files |
| `docs/submission/` | Measured environment, dependency audit, authored-kernel quota and readiness |
| `SUBMISSION.md` | Exact source/notebook binding and remaining team-supplied materials |

## Reproduce

```bash
git clone https://github.com/BAITC-Hacks/hack-3ef0581f-setww.git
cd hack-3ef0581f-setww
git checkout --detach 652334f3c63764d8ca998ef3c9ae1c378465cf63
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install "triton==3.6.0" "datasets==4.8.5"
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
SOUP_APPLY_SOLUTION=1 python -m pytest solution tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Use the official Kaggle notebook with Tesla T4 x2 selected, Internet enabled and only GPU0 used.
Install dependencies before measurements, never from solution.apply(). Each command starts a fresh
process and empty compiler caches; ABBA is base, solution, solution, base in one session.
Only solution/ is copied by the jury; all runtime helpers are inside it. The repository's 15 official
harness/task/test files remain byte-identical to starter `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.

Measured environment: python=3.13.15; soup-cli=0.77.0; torch=2.11.0+cu128; triton=3.6.0; transformers=5.16.1; peft=0.20.0; trl=0.29.1; bitsandbytes=0.50.2; accelerate=1.14.0; datasets=4.8.5. GPU driver/T4 inventory is in PROFILE.json/result.json.


## Source and reproducibility

Measured runtime: `652334f3c63764d8ca998ef3c9ae1c378465cf63`. The final `solution/`, `harness/`, `task/` and `tests/` match that commit byte for byte.
Later commits add only documentation, the saved notebook and evidence; they do not claim a new measurement.
The original harness records cold process time and external `nvidia-smi` peak. Its meter, allocator settings and checks are unchanged.

This qualification covers the public task and seed 1. Hidden rows and seeds were not tested for this exact runtime; the jury's hidden run and code review determine final acceptance.

## Authorship and licenses

The team wrote the counted Triton implementations. Documented contributors are
[@kok-o](https://github.com/kok-o) (CE, integration and new kernels) and
[@Nari-Ab](https://github.com/Nari-Ab) (head/casts).
Reference operation definitions and ideas: [Qwen2 in Transformers5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py),
[PEFT LoRA0.20.0](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py),
[PyTorch native dropout2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu),
and [Soup0.77.0 chunked loss](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/utils/stream_loss_chunk.py).
Installed native cuBLAS/bitsandbytes/PyTorch SDPA are not claimed as authored Triton kernels.
Starter SwiGLU and every renamed/packed derivative are explicitly excluded from the quota.


Apache-2.0 applies to the repository; retain [LICENSE](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/blob/652334f3c63764d8ca998ef3c9ae1c378465cf63/LICENSE) and [NOTICE](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/blob/652334f3c63764d8ca998ef3c9ae1c378465cf63/NOTICE) from the starter. The five
Soup v0.77.0 regression test files are attributed in NOTICE. Model Qwen/Qwen2.5-1.5B-Instruct:
Apache-2.0. Task data AmanMussa/kazakh-instruction-v2: MIT on its card; translated Stanford Alpaca
has CC BY-NC4.0 provenance, as disclosed by the official starter. Data/model/adapter/compiler caches
are downloaded or generated for the run and are not bundled in the submitted implementation.



## Submission materials

The code, public notebook, measured results and verification evidence are ready. [Team presentation](https://docs.google.com/presentation/d/1tCZmI-LPZFjENUWKEz_xOEdSMxerElHM/edit) is linked; public viewing access has not been independently verified. The team has not provided a video. Full team/contact details and platform delivery are tracked in [SUBMISSION.md](SUBMISSION.md).
