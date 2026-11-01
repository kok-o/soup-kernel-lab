# Original Brev proof: native casts and exact metadata increment

Same A9 runtime as the separate official Kaggle 57.89 s / 2307 MiB / all10 result.
These Brev measurements are diagnostic and are not Kaggle qualification times.

First full cold train: **45.36 s / 2307 MiB**, loop 29.93 s, eval 2.05204.
Full suite: 348 passed, two optional BLAS tests skipped (backend remains unchanged).

Serial cold train-only A/B/B/A, empty compiler caches each:

| Run | Runtime | Full train seconds | Peak MiB |
|---|---|---:|---:|
| A0 | 35ea24c compact dropout | 47.82 | 2307 |
| B1 | a9ac67e + native casts/metadata | 45.21 | 2307 |
| B2 | a9ac67e | 46.06 | 2307 |
| A3 | 35ea24c | 48.87 | 2307 |

Medians A48.345 s, B45.635 s: 2.71 s improvement; control bookend drift 1.05 s.
The original unmodified run_task.train parent timer was used. Eval is separate.
Every pair run checks its source digest before and after training.

All manifest members and both source snapshots were independently checked against Git;
the archive retains original train logs, metrics, task configs, eval logs/results,
source snapshots and helper scripts. No adapter weights, caches or credentials are included.
The initial full-suite metadata fixture failure is retained. Its fix unwraps the true
stdlib reference, preserving the exact 8-vs-4 stat assertions and runtime semantics.
