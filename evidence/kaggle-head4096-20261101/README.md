# Head chunk 4096: memory reduction, no speed win

**Both pass 10/10. Keep the original 8192-row head for the speed baseline.**

| Same-session original 2+2 ABBA | 8192-row control | 4096-row candidate |
|---|---:|---:|
| Commit | `77cfc4d` | `b758b60` |
| Full cold median | 64.80 s | 64.95 s |
| Solution runs | 64.57 /65.03 s | 64.63 /65.27 s |
| External GPU peak, both repeats | 495 MiB | 475 MiB |
| Training-loop median | 41.41 s | 41.59 s |
| Stock Soup median | 84.52 s | 84.15 s |
| Eval loss | 2.05210 | 2.05220 |
| Relative eval delta vs own stock base | +0.118% | +0.125% |
| Minimum counted own-kernel share | 7.9% | 7.9% |
| Patched suite | 373 passed, 2 skips | 373 passed, 2 skips |
| All original checks | 10/10 | 10/10 |

The candidate saves 20 MiB externally but is 0.15 s slower in observed median;
both time ranges overlap. The control/candidate stock bases also differ.
There is no demonstrated speed improvement. At the requested <=500 MiB limit,
the original base already fits. Do not promote a different chunk solely to
show a code change. The 4096 variant remains an optional measured memory
tradeoff in its isolated branch, not the selected runtime.

The runtime changes only head chunk 8192 ->4096 and its matching full/tail cold
compile layouts. No ready cache, datatype/RNG/data/settings or required output
was changed. FP32 row accumulation has a different number of chunks; quality,
loss, entropy, accuracy, rewind and backward checks remain enabled.

- Control: `77cfc4d7ef293b1d45e2dab06b5428d178f1faa1`.
- Candidate: `b758b6030ba0215df77e6a13c59fe94391014aff`.
- [Candidate code](https://github.com/kok-o/soup-kernel-lab/commit/b758b6030ba0215df77e6a13c59fe94391014aff)
  and `kaggle/head4096/` preserve reproduction.
- [Free Kaggle T4 notebook](https://www.kaggle.com/code/koko667/soup-lab-head4096),
  version1, is private to the owner; raw source-bound results here are public.
- Original two complete 2+2 suites, eight full64-step commands, original
  allocator,100ms external sampler and TritonWatch, no skip-tests.
- Local `VERIFIED.json` validates all97 source files for each snapshot,15
  official files, source/result/checker bindings, matching data and all8 runs.
- Allocator diagnostics: control275.1/326MiB, candidate251.1/306MiB allocated/
  reserved. Headline memory uses external495/475MiB.
- Raw control SHA256: `c189f29c3ac5667e159a62b7381352794e5a0b1fab17ad44d0a8111608df6129`.
- Raw candidate SHA256: `b1e969c7fee79f04e26c2da74531fe20357cb371c2d3cc56aba20239af9aec35`.

```bash
kaggle kernels push -p kaggle/head4096 --accelerator NvidiaTeslaT4 --timeout 1800
python research/verify_offload_comparison.py \
  --downloads evidence/kaggle-head4096-20261101 --out verified-head4096.json
```

No paid GPU or original hackathon repository/notebook was changed.
