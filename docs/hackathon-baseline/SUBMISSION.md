# Submission: SetWW

**Kaggle Tesla T4: 61.40 s full cold / 729 MiB, all 10 original checks passed.**

## Links and exact source

- Repository, measured runtime commit: [652334f3c63764d8ca998ef3c9ae1c378465cf63](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/commit/652334f3c63764d8ca998ef3c9ae1c378465cf63).
- Kaggle notebook, same link as README: [Kaggle](https://www.kaggle.com/code/koko667/setww-final-slotviews?scriptVersionId=364323222). Public saved version, pinned to the measured source. The Share Public setting was checked in Kaggle; independent anonymous access was not verified.
- Notebook literal `COMMIT = '652334f3c63764d8ca998ef3c9ae1c378465cf63'` is verified in its source.
- Evidence and reproduction: [source-bound raw/results/checks](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/tree/5d10568bd847d2586bb67ad01852e77a28b4d6fd/evidence/kaggle-slotviews-20261031).
- Evidence archive commit: `5d10568bd847d2586bb67ad01852e77a28b4d6fd`; no new runtime measurement is claimed for a later wrapper.
- Demo video, duration at most 3 minutes: **NOT PROVIDED** (confirmed by the team).
- Presentation: [Team presentation](https://docs.google.com/presentation/d/1tCZmI-LPZFjENUWKEz_xOEdSMxerElHM/edit). Link supplied by the team; public viewing access has not been independently verified.

## Team, at most 4 people

Recorded contribution handles are listed below. The team will complete names, membership, Kaggle handles and captain contact.

| Name | Role | GitHub | Kaggle | Captain contact |
|---|---|---|---|---|
| PENDING_HUMAN_NAME (recorded handle kok-o) | CE/integration/new kernels | [kok-o](https://github.com/kok-o) | PENDING_HUMAN_CONFIRM | PENDING_HUMAN_CAPTAIN_CONTACT |
| PENDING_HUMAN_NAME (recorded handle Nari-Ab) | Head/casts | [Nari-Ab](https://github.com/Nari-Ab) | PENDING_HUMAN_CONFIRM | — |

Complete membership confirmation: **PENDING_HUMAN_TEAM_CONFIRMATION**.

## Our kernels

The columns show the exact two measured solution runs, steps 3+ against their recorded wall-time
denominators. Kernels not launched usefully in both runs are omitted. No per-kernel causal speedup is claimed.

| Triton kernel | File | Operation replaced | Run 1 / Run 2 share |
|---|---|---|---:|
| `_hosthead_row_stats` | `solution/hosthead_stats.py` | FP32 streamed vocabulary row statistics for CE, entropy, accuracy and dHidden | 0.4412% / 0.4408% |
| `activation_pack_kernel` | `solution/_activation_kernels.py` | INT8/256 saved activation encoding | 0.9356% / 0.9332% |
| `activation_unpack_kernel` | `solution/_activation_kernels.py` | INT8/256 saved activation reconstruction | 0.7691% / 0.7528% |
| `dropout_mask_gradient_kernel` | `solution/_dropout_kernels.py` | Original dropout gradient from the packed native mask | 0.6592% / 0.6497% |
| `dropout_mask_pack_kernel` | `solution/_dropout_kernels.py` | Pack original native dropout mask; preserve RNG consumption | 0.2612% / 0.2671% |
| `fork_gradient_sum_kernel` | `solution/fork_sum.py` | Widen independently rounded gate/up gradients before FP32 addition | 0.2646% / 0.2646% |
| `grouped_activation_pack4_kernel` | `solution/_grouped_activation_kernels.py` | INT4/64 wide MLP saved activation encoding | 1.5034% / 1.5430% |
| `lora_scale_add_kernel` | `solution/_lora_kernels.py` | Preserve LoRA FP16 scale and FP32 output addition | 0.6896% / 0.6669% |
| `lora_scale_gradient_kernel` | `solution/_lora_kernels.py` | Original LoRA scaled backward rounding | 0.3723% / 0.3717% |
| `rmsnorm_backward_kernel` | `solution/_rmsnorm_kernels.py` | Frozen-weight FP32 RMSNorm input gradient | 0.7343% / 0.7238% |
| `rmsnorm_forward_kernel` | `solution/_rmsnorm_kernels.py` | Frozen-weight FP32 RMSNorm forward | 0.6607% / 0.6605% |
| `rotary_backward_kernel` | `solution/_rope_kernels.py` | Original FP32 RoPE input gradient | 0.3542% / 0.3533% |
| `rotary_forward_kernel` | `solution/_rope_kernels.py` | Original RoPE formula and query FP16 cast boundary | 0.6830% / 0.6716% |

The team wrote the counted Triton implementations. Documented contributors are
[@kok-o](https://github.com/kok-o) (CE, integration and new kernels) and
[@Nari-Ab](https://github.com/Nari-Ab) (head/casts).
Reference operation definitions and ideas: [Qwen2 in Transformers5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py),
[PEFT LoRA0.20.0](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py),
[PyTorch native dropout2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu),
and [Soup0.77.0 chunked loss](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/utils/stream_loss_chunk.py).
Installed native cuBLAS/bitsandbytes/PyTorch SDPA are not claimed as authored Triton kernels.
Starter SwiGLU and every renamed/packed derivative are explicitly excluded from the quota.


## Results on Kaggle T4

Measured runtime: [`652334f3c63764d8ca998ef3c9ae1c378465cf63`](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/commit/652334f3c63764d8ca998ef3c9ae1c378465cf63). Evidence archive commit: `5d10568bd847d2586bb67ad01852e77a28b4d6fd`.
The results below belong to this exact runtime. Notebook `COMMIT` matches it; the separate evidence commit stores the raw results and reproduction artifacts.
[Kaggle notebook](https://www.kaggle.com/code/koko667/setww-final-slotviews?scriptVersionId=364323222). Public saved version, pinned to the measured source. The Share Public setting was checked in Kaggle; independent anonymous access was not verified.

| Original Kaggle T4 ABBA | Base Soup | Solution | Ratio |
|---|---:|---:|---:|
| Full cold soup train wall, s | 84.34 | 61.40 | 1.3736x |
| External nvidia-smi peak, MiB | 14345 | 729 | 19.6776x |
| Training loop only, s | 57.44 | 38.33 | separate from full cold |
| Public held-out eval loss | 2.04965 | 2.05214 | +0.121% |

All 10 original checks pass. Patched tests: 370 passed, 2 original skips; official check 10 passes 79 regressions.
Counted own kernels: 8.3% of the training time (lowest of 2 runs, steps 3+). Raw JSON/logs/source: [evidence](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/tree/5d10568bd847d2586bb67ad01852e77a28b4d6fd/evidence/kaggle-slotviews-20261031).
Raw ZIP SHA256: `f9f33be19b21c093fa74855e38e4c898ecd87d589955ac5ab9ddbab7ff531b7e`. Canonical solution SHA256: `3b8e9eff479025995004eb34dab2db10f27fcd3f440f015021f8af3db1a78f6a`.

## Reproduction and fixed task

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

Task: Qwen2.5-1.5B-Instruct NF4 double quant; original 256 training rows in their original order;
batch 4, max 512 tokens, 1 epoch, 64 steps; LoRA r=16, alpha=32, dropout=0.05, q_proj/v_proj; native
adamw_torch,cosine schedule, LR=2e-4. Same random seed/RNG consumption, allocator
expandable_segments:True, adapter names/shapes/settings, entropy/accuracy/shiftedloss/rewind.
Public eval rows 40000..40255; jury uses other rows and seed. No hidden data were read.

- [x] All 10 original checks passed for the measured source above, without skip-tests.
- [x] Source/evidence/runtime identity, dependencies, results and limits documented.
- [x] Pinned saved-version URL, COMMIT and Share Public setting checked; anonymous access not independently verified.
- [x] Presentation link supplied and included.
- [ ] Video supplied, complete team/contact data entered and platform delivery confirmed.

## Presentation covers

Operations and why they matter; kernels and dataflow; original cold wall, external peak and
quality against same-session base; exact source/notebook/evidence; limitations and remaining work.
The result reported here is the complete Kaggle run above; separate Brev screens are not used as qualification.

## Limitations and licenses

Frozen NF4 weights retain their exact native values; some saved activations use INT4/INT8 storage with reconstruction for backward. Complete training is not claimed to be bitwise identical to the baseline. Acceptance here refers to all 10 original checks and their quality tolerances, including the measured +0.121% eval-loss difference.

This exact source is the user's final selected result. The earlier research target of 52 s
was not reached and is not claimed. The reported 61.40 s is full cold wall time; 38.33 s is only the
training loop. Results apply to public rows and seed 1 on Kaggle T4 FP16. Hidden rows/seeds remain
untested for this runtime; other variants' seed checks are not transferred. No Brev training run was executed for this source; this qualification was run directly on Kaggle. Separate 304 or d070 Brev timings are not transferred.
The original ABBA measures this complete source against its own same-session base; it does not
establish the causal speedup of each component. Eval is outside timed training. PyTorch memory
diagnostics and CPU byte budgets do not replace nvidia-smi peak. The jury's hidden run and code
review remain the final acceptance authority.

Apache-2.0 applies to the repository; retain [LICENSE](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/blob/652334f3c63764d8ca998ef3c9ae1c378465cf63/LICENSE) and [NOTICE](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/blob/652334f3c63764d8ca998ef3c9ae1c378465cf63/NOTICE) from the starter. The five
Soup v0.77.0 regression test files are attributed in NOTICE. Model Qwen/Qwen2.5-1.5B-Instruct:
Apache-2.0. Task data AmanMussa/kazakh-instruction-v2: MIT on its card; translated Stanford Alpaca
has CC BY-NC4.0 provenance, as disclosed by the official starter. Data/model/adapter/compiler caches
are downloaded or generated for the run and are not bundled in the submitted implementation.
