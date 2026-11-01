# Original Kaggle T4 qualification

**68.58 s full cold / 495 MiB external peak; all 10 original checks pass.**

Measured runtime: `d070881332c9925744871bfbd38987f7fb3f3130`. [Saved Kaggle notebook](https://www.kaggle.com/code/koko667/setww-final-stream-slab?scriptVersionId=364313153).
Later documentation commits do not represent new runtime measurements.

| Run | Full cold seconds | External peak MiB |
|---|---:|---:|
| base_1 | 87.23 | 14345 |
| base_2 | 79.97 | 14345 |
| solution_1 | 68.90 | 495 |
| solution_2 | 68.25 | 495 |

Base median 83.60 s / 14345 MiB. Solution training loop only:
43.45 s. Full cold time includes normal imports, JIT, loading, training and exit.
Public eval 2.05216 versus 2.04963 (+0.123%).
Minimum counted own-kernel share: ['7.8% of the training time (lowest of 2 runs, steps 3+)'].
The official sampler, allocator, JITFunction.run meter, data, RNG, LoRA and all 64 steps remain intact.

Verification covers 138 raw manifest entries, 95 source files,
15 unchanged official files, result/checker/source bindings and four complete 64-step runs.
The original checker ran without skip-tests. See VERIFIED.json for patched test results and versions.
Raw SHA256: `15a0975db104cc2c04b41265f05feb84f668a756b9df2a0ee3cd8de2358fae21`.

Reproduce with the pinned versions in VERIFIED.json, `python task/prepare_data.py`,
`python harness/run_task.py --repeats 2`, then `python harness/check_rules.py`.
No weights, adapters, compiler caches or credentials are in this evidence archive.
This result must not be replaced with a Brev screen or a training-loop-only timing.
The human target of approximately 52 s / 700 MiB remains an unachieved target unless these measured values satisfy it.
