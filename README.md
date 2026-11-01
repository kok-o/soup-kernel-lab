# Soup Kernel Lab

Personal research repository for experimenting with Soup training performance and GPU memory.
Independent repository, not a fork or a new competition submission.

## Start on Kaggle

- [New research notebook](https://www.kaggle.com/code/koko667/soup-kernel-lab-independent-experiments) (private to its owner initially).
- [Notebook source](kaggle/soup-kernel-lab.ipynb).
- Enable Internet, select a Tesla T4 accelerator and set `RUN_TRAINING = True`.
- Set `REF` to your pushed branch or exact commit SHA. Default `main` fetches the latest main at session start.
- Start a fresh session and Run All. Tests, original 2+2 ABBA, all official checks and a source-bound evidence ZIP follow.

The notebook downloads code from **this repository**. It resolves the selected ref once, checks out the exact SHA and records source hashes. No edits to the old notebook are needed. `RUN_TRAINING = False` only checks source checkout; it performs no training or dependency installs.

## Test a hypothesis

```bash
git clone https://github.com/kok-o/soup-kernel-lab.git
cd soup-kernel-lab
git switch -c experiment/my-hypothesis
# Change solution/, preserving the task and official measurements.
git add solution
git commit -m "Describe the hypothesis"
git push -u origin experiment/my-hypothesis
```

In Kaggle set `REF = 'experiment/my-hypothesis'` and run a fresh session. Each completed run exports `soup-lab-<commit>-evidence.zip`. Compare full cold time, external peak, and quality; do not equate training-loop time with full process time.

Changing `solution/` only requires pushing the selected branch. If you change the notebook itself, import the new `kaggle/soup-kernel-lab.ipynb` into the separate research notebook. The saved Kaggle notebook does not automatically refresh its own cells from GitHub.

## Preserved baseline

Historical result: **61.40 s full cold /729 MiB; 10/10 checks**, measured on Kaggle T4 for original runtime `652334f3c63764d8ca998ef3c9ae1c378465cf63`. This is a reference, not a performance claim for subsequent edits.

- `solution/`: complete runtime implementation and tests.
- `harness/`, `task/`, `tests/`: unchanged official comparison protocol.
- `research/`, `docs/`, `evidence/`: imported research tools and available evidence.
- `BASELINE_PROVENANCE.json`: baseline source hashes and original attribution.
- `notebooks/final-qualification/`: byte-preserved original embedded-source notebook, for archival reproduction only.
- `docs/hackathon-baseline/`: original README and submission record.

Frozen NF4 weights keep their native values; some saved activations use INT4/INT8 storage. The complete training trajectory is not claimed to be bitwise identical to baseline. Keep the original quality checks enabled.

Dependencies and model/data downloads require Internet. Models, datasets, adapters and compiler caches are not included. Training needs a GPU; the initial notebook publication is a CPU checkout check only. Kaggle GPU quota is not used until you explicitly enable training and an accelerator.

## Attribution

Based on BAITC-Hacks/hack-3ef0581f-setww, imported tree `086a7eebb88caa3386d2b8167c030998a06bfd73`. Original licenses, notices, attribution and evidence are retained. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [historical README](docs/hackathon-baseline/README.md).
