# R2 final T4 ABBA · saved Kaggle version 5

- Source commit: `b0dcbfe10d7d8fbd47aa0d8fe7d69f978e26d62c`; chunk **256**; `compact=False`.
- [Kaggle notebook](https://www.kaggle.com/code/narimanabdikarim/setww-r2-abba-3527e32), saved version **5**, kernel ID 141689692.
  Notebook is private; the JSON/logs here and the source-embedded
  [reproduction notebook](../../../templates/r2_verified_notebook.ipynb) are the handoff.
- Measured 2026-10-30 on Kaggle Tesla T4, FP16; one visible GPU per measured process.
- R1 v2 CE + R2 head + original dtype casts + starter SwiGLU helper.
  This is a combined candidate, not a component ablation. SwiGLU is excluded from own share.

| Metric | Base | Solution |
|---|---:|---:|
| Full command time, median seconds | 83.06 | 79.97 |
| nvidia-smi peak, maximum MiB | 14345 | 8445 |
| Public eval loss | 2.04968 | 2.04968 |
| Minimum own share, steps 3+ | — | 10.5% |
| Official checks | — | **10/10 PASS** |

ABBA: base_1 82.94 s, solution_1 79.97 s, solution_2 79.96 s, base_2 83.18 s.
Speedup **1.0386×**, below the 1.05 scoring threshold (speed counted as 1.0).
Memory ratio **1.6986×**, reduction **41.13% / 5900 MiB**. Eval delta +0.000%
at the reported precision. First loss gap 0.00%, mean curve gap 0.01%.
79 official regressions and 93 solution tests passed on T4. Public checks do not
establish hidden-seed results or jury acceptance.

## Files and integrity

- `runs_final-result.json`, `runs_final-checks.json`: unaltered official final outputs.
- `runs_chunk256-*`: earlier single-pair pilot in the same saved version; not final ABBA.
- `selection.json`: paired-ratio selection proxy, not a predicted contest score.
- `source-256.json`: source-only archive and individual file hashes.
- `r2-source-manifest.json`: source provenance before phases.
- `raw-evidence.zip`: original downloaded archive, SHA-256
  `3237baa0efd547181cbef3e0ac09d81bc5e214434b7d237681d52e3f023fa430`.
- `verified-summary.json`: derived index, not a replacement for raw outputs.
- Unit/hardware/check/score logs are copied without editing. The raw archive includes
  Triton JSON metadata for attribution; no compiled binaries, tensor weights, model,
  datasets, credentials or optimizer state are included.

Both phases' `checks.result_sha256` were verified against raw result bytes.
Both `checks.solution_sha256` were recomputed from sorted `solution/` git blobs
at the tested SHA (relative path bytes followed by content bytes, as official harness).
All ten final checks and two base/two solution runs were asserted programmatically.
The source notebook embeds only tracked runtime files plus Apache LICENSE/NOTICE;
no GitHub or Kaggle token is needed to extract it.

The optional incompatible preinstalled torchao was removed before base and solution;
no package changes occurred in measured processes. Versions are in `result.json`.
Rerunning is a new measurement; dependency versions/hardware can affect its outcome.
