# Compact dropout with native casts and exact package metadata

Official Kaggle T4 ABBA: **57.89 s / 2307 MiB, all 10 checks pass**.
Stock median 80.74 s / 14345 MiB; candidate loop 33.22 s.
Eval loss 2.05212 versus stock 2.04967 (+0.120%).
348 tests pass, two optional BLAS-selection tests skip because no backend override is selected.

Runtime `a9ac67ea6c17e61127613e976d39a096a298058d`. Canonical solution SHA256 `53d2057ac60abda2d7ec00a441c91d5b20bb9551973a874582cb1e12e98e338d`.
Archive SHA256 `7a9244ed78c417fffba81286fdba244e67a8210685d7b5c1953d600087658d41`.
Result SHA256 `55a5463566b34b80273ac5bc9102a294295531039e5be540a297c6fbbf1e13b6`.

All 39 manifest entries, absence of unlisted members, the complete source zip
against Git, and source/result/check bindings were independently verified.
The raw archive contains original logs, repeat metrics, task configs and the official checker.
This interactive qualification is separate from a public saved Run All.

The runtime combines the prior compact/grouped/dropout implementation with native
autograd casts and exact installed-package metadata discovery from peer runtime
0d5243c5071ab43b8eee494d533c9323c6ed595a. Floating-point boundaries and FP32
branch-gradient summation are retained; no BLAS backend or precision override is enabled.
Metadata discovery preserves package mapping, CSV validation and fallbacks, and is
guarded to CPython 3.13. Other interpreters retain the standard-library path.

The full suite initially exposed a metadata test fixture that captured an already
patched function as its reference. The fixture now unwraps our own replacement to
the actual standard-library function. Its exact 8-versus-4 filesystem-stat assertions
are unchanged. No runtime or official test was altered to resolve that fixture issue.

The earlier 61.83 s / 2307 MiB result and public reserve-v5 remain preserved.
PR21 review refs remain frozen; this research branch does not move those refs.
