# NF4 backward early packed-slot release: rejected

**No speed win. Keep the original lease in the production baseline.**

| Same-session original 2+2 ABBA | Bulk-H2D control | Early-release candidate |
|---|---:|---:|
| Commit | `77cfc4d` | `f4aca2b` |
| Full cold median | 67.47 s | 68.00 s |
| Solution runs | 67.55 /67.39 s | 67.81 /68.20 s |
| External GPU peak, both repeats | 495 MiB | 495 MiB |
| Training-loop median | 43.03 s | 43.20 s |
| Stock Soup median | 88.37 s | 88.19 s |
| Eval loss | 2.05210 | 2.05217 |
| Relative eval delta vs own stock base | +0.119% | +0.122% |
| Minimum counted own-kernel share | 7.9% | 7.9% |
| Patched suite | 373 passed, 2 skips | 374 passed, 2 skips |
| All original checks | 10/10 | 10/10 |

The candidate is 0.53 s slower in observed median despite a slightly faster
stock base. Four stock runs span 0.97 s. This is not evidence that the lease
end was a useful critical-path bottleneck. Do not retry the same change or
promote it just because parity and ten checks passed. The earlier 73.17 versus
68.15 s single screens overstated the benefit and are not qualification.

The experiment retains identical existing decode and matmul operations but
records the packed slot's last-reader event between them. The dense decoded
weight owns separate storage. The new test verifies that the lease is released
before GEMM and checks native forward/dX bit parity. It is independent of the
previously rejected speculative gate-to-up predictor.

Sources and reproduction:

- Full control: `77cfc4d7ef293b1d45e2dab06b5428d178f1faa1`.
- Full candidate: `f4aca2b2b1d8b3d6a5b6f8a082ab83731d151cca`.
- [Rejected code commit](https://github.com/kok-o/soup-kernel-lab/commit/f4aca2b2b1d8b3d6a5b6f8a082ab83731d151cca)
  remains available; it is not installed by the final baseline.
- [Free T4 notebook](https://www.kaggle.com/code/koko667/soup-lab-nf4-early-release)
  version1 is private to the owner. Pinned source is in `kaggle/nf4-early-release/`.
- Both complete original suites, untouched meter/allocator/TritonWatch and all
  ten checks; no skip-tests, paid GPU or historical submission change.
- `VERIFIED.json` checks all 97/98 source files, original15files, data,
  result/checker/source bindings, all eight 64-step runs and rewind outputs.
- Raw control SHA256: `d6867cf8504ec0e37d6eb61b5631f3f4277dcf53bef47913f77d760a569bc15d`.
- Raw candidate SHA256: `4b18c2f15c672887a7740a7fe4c00e24621aec7de1f7c590b650dcdc240d542c`.

```bash
kaggle kernels push -p kaggle/nf4-early-release --accelerator NvidiaTeslaT4 --timeout 1800
python research/verify_offload_comparison.py \
  --downloads evidence/kaggle-nf4-early-release-20261101 --out verified-nf4-release.json
```

No claim is made by comparing these times to another T4 session's 62.55 s.
