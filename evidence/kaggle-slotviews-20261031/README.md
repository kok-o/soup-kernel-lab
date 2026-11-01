# Original Kaggle T4 qualification

**61.40 s full cold / 729 MiB external peak; all 10 original checks pass.**

Measured runtime: `652334f3c63764d8ca998ef3c9ae1c378465cf63`. [Saved Kaggle notebook](https://www.kaggle.com/code/koko667/setww-final-slotviews?scriptVersionId=364323222).
Later documentation commits do not represent new runtime measurements.

| Run | Full cold seconds | External peak MiB |
|---|---:|---:|
| base_1 | 87.24 | 14345 |
| base_2 | 81.43 | 14345 |
| solution_1 | 61.38 | 729 |
| solution_2 | 61.42 | 729 |

Base median 84.34 s / 14345 MiB. Solution training loop only:
38.33 s. Full cold time includes normal imports, JIT, loading, training and exit.
Public eval 2.05214 versus 2.04965 (+0.121%).
Minimum counted own-kernel share: ['8.3% of the training time (lowest of 2 runs, steps 3+)'].
The official sampler, allocator, JITFunction.run meter, data, RNG, LoRA and all 64 steps remain intact.

Verification covers 139 raw manifest entries, 96 source files,
15 unchanged official files, result/checker/source bindings and four complete 64-step runs.
The original checker ran without skip-tests. See VERIFIED.json for patched test results and versions.
Raw SHA256: `f9f33be19b21c093fa74855e38e4c898ecd87d589955ac5ab9ddbab7ff531b7e`.

Reproduce with the pinned versions in VERIFIED.json, `python task/prepare_data.py`,
`python harness/run_task.py --repeats 2`, then `python harness/check_rules.py`.
No weights, adapters, compiler caches or credentials are in this evidence archive.
This result must not be replaced with a Brev screen or a training-loop-only timing.
The human target of approximately 52 s / 700 MiB remains an unachieved target unless these measured values satisfy it.
