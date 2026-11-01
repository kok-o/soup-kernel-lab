# Original Kaggle T4 qualification

**66.84 s full cold / 729 MiB external peak; all 10 original checks pass.**

Measured runtime: `304722792b5b9b44d9dae454487e27f728ff2e8f`. [Saved Kaggle notebook](https://www.kaggle.com/code/koko667/setww-final-offload20?scriptVersionId=364320899).
Later documentation commits do not represent new runtime measurements.

| Run | Full cold seconds | External peak MiB |
|---|---:|---:|
| base_1 | 82.56 | 14345 |
| base_2 | 82.89 | 14345 |
| solution_1 | 66.95 | 729 |
| solution_2 | 66.74 | 729 |

Base median 82.72 s / 14345 MiB. Solution training loop only:
42.25 s. Full cold time includes normal imports, JIT, loading, training and exit.
Public eval 2.05209 versus 2.04965 (+0.119%).
Minimum counted own-kernel share: ['8.2% of the training time (lowest of 2 runs, steps 3+)'].
The official sampler, allocator, JITFunction.run meter, data, RNG, LoRA and all 64 steps remain intact.

Verification covers 138 raw manifest entries, 95 source files,
15 unchanged official files, result/checker/source bindings and four complete 64-step runs.
The original checker ran without skip-tests. See VERIFIED.json for patched test results and versions.
Raw SHA256: `b7f33443e4bdc699d299c36152537443e662da005a25aec2eedb60919c52cbc2`.

Reproduce with the pinned versions in VERIFIED.json, `python task/prepare_data.py`,
`python harness/run_task.py --repeats 2`, then `python harness/check_rules.py`.
No weights, adapters, compiler caches or credentials are in this evidence archive.
This result must not be replaced with a Brev screen or a training-loop-only timing.
The human target of approximately 52 s / 700 MiB remains an unachieved target unless these measured values satisfy it.
