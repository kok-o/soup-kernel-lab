# All 28 activation layers offloaded with the final slot-view cache

**Kaggle T4: 65.33 s full cold /495 MiB external peak; all 10 original checks pass.**

This is an independent personal post-hackathon measurement. The historical competition submission and notebook were not changed.

| Original 2+2 ABBA in the same Kaggle session | 20 layers: control | 28 layers: candidate |
|---|---:|---:|
| Full cold median | 63.88 s | 65.33 s |
| External nvidia-smi peak | 729 MiB | 495 MiB |
| Solution run 1 | 63.17 s | 65.23 s |
| Solution run 2 | 64.58 s | 65.43 s |
| Train loop median | 40.70 s | 41.97 s |
| Original Soup full cold median | 88.06 s | 86.44 s |
| Speedup versus each suite's Soup base | 1.3785x | 1.3231x |
| Solution eval loss | 2.05214 | 2.05210 |
| Eval loss relative to each Soup base | +0.121% | +0.121% |
| Minimum counted own-kernel share | 8.1% | 7.8% |
| Patched suite | 370 passed, 2 original skips | 370 passed, 2 original skips |
| Official checks | 10/10 | 10/10 |

The memory target below 500 MiB was reached in both candidate runs. The observed full cold median is approximately 1.5 s slower while external peak is 234 MiB (32.1%) lower. This is a memory/time tradeoff, not a speed improvement.

The two complete original suites ran sequentially on one Kaggle T4 session with separate source checkouts and processes. The four stock Soup times span 3.15 s; this does not establish an exact causal speed penalty independent of machine drift. Historical 61.40 s /729 MiB is preserved as a previous measurement, not used as today's control.

## Source and protocol

- Control: `60e973d310ad4b25ecf2eb3d18df284f7a563fcc`; runtime bytes match historical `652334f3`.
- Candidate: `06dc4b1957fbe1e23b4b32fa420d3205890fa553`.
- Only runtime change: `solution/__init__.py`, `patch_selective_offload(layers=20)` to `layers=28`.
- The same bounded slot metadata cache, two GPU weight slots, native NF4 values, head chunk8192, activation codec, task, RNG, LoRA and optimizer are retained.
- [Kaggle saved version](https://www.kaggle.com/code/koko667/soup-lab-offload28-compare?scriptVersionId=364496225) is private to the owner. Public raw evidence and verification are included here.
- GPU selector: `NvidiaTeslaT4`; only GPU0 used by the original harness; other GPU idle. No Brev machine was started.
- Eight complete 64-step train commands: two base and two solution runs per variant. Original allocator,100ms external sampler and TritonWatch retained. No skip-tests.

Local verifier checks every archive manifest entry, all 96 source files against the pinned Git commit, 15 unchanged official files, checker/result/solution bindings, all 10 checks, 64 steps and rewind for every run, one T4/GPU0, allocator and matching train/eval data hashes. Both raw archives include tests, metrics, logs, result, checks, source and provenance. No model weights, adapters, compiler caches or credentials are bundled in them.

Some saved activations use INT4/INT8 storage. No complete bitwise training-trajectory equivalence is claimed; native weight tests and official quality tolerances apply.

## Reproduce

The pinned notebook is in `kaggle/offload28-comparison/`. Submit through the CLI:

```bash
kaggle kernels push -p kaggle/offload28-comparison --accelerator NvidiaTeslaT4 --timeout 2100
```

Each variant runs `python -m pytest solution tests -q`, `python harness/run_task.py --repeats 2`, `python harness/check_rules.py` and `python harness/score.py runs/result.json`. Installs and data preparation occur outside timed train processes, as in the original protocol.

Verify this published evidence on CPU:

```bash
python research/verify_offload_comparison.py --downloads evidence/kaggle-offload28-slotviews-20261101 --out verified-offload28.json
```

## Raw hashes

- Control ZIP: `dd875b89f7133e099777dc3c517793316e53d40c1551d3bc30acecfd1581d2c5`.
- Candidate ZIP: `6f3400316df2b274ac35a54c74846dd5f248862d62a48f0b8ff2151be6fae09c`.

Torch diagnostics are secondary: control allocated494.6/reserved562.0 MiB; candidate allocated275.1/reserved326.0 MiB. The headline uses external729/495 MiB, not allocated memory.

## Next evidence needed

The candidate spends41.97 s inside training and23.36 s outside it; `apply()` is about5.18 s. The unallocated outside-loop time must be attributed with an actual phase trace before changing import/loading/save/exit code. The next speed investigation should profile transfers and waits on this cache-enabled stack. Old gate-to-up prefetch and head16k regressions are recorded in `research/OFFLOAD28_EXPERIMENT.md`; do not relabel them as untested or retry them without a new measurable reason.
