# reserve-v5 research ABBA

Runtime source/tag: `a017d6fea308e7c250323ad70f44af34ef9e3d93` / `reserve-v5`.
Solution SHA-256: `68cab873fca0c754556c68cca72dd80aca67a4b1066619c7c7b5624c0971da32`.
Executed in the Kaggle T4 diagnostic session with the unchanged official harness,
`--repeats 2`, fresh compiler caches and all ten checks (including 79 Soup regressions).

| Official ABBA | Base | Solution |
|---|---:|---:|
| Median full-command time, s | 81.97 | 65.08 |
| Max Nvidia peak, MiB | 14345 | 2447 |
| Eval loss | 2.04963 | 2.05207 |
| Official checks | — | 10/10 PASS |

Solution repeats: 64.98 / 65.18 s, both 2447 MiB. Eval delta +0.119%.
Author contribution minimum 18.8%; starter SwiGLU and its packed rewrite are excluded.
275 combined GPU/regression tests passed before this ABBA. Initial solution-only
screen: 65.49 s / 2447 MiB / eval 2.05218; it is retained separately.

All 56 manifest entries, result/check/provenance hashes and both source ZIPs were
verified against the exact Git source. The later main-branch documentation merge
does not change any `solution/` byte. No compiler cache/model/data rows/adapter weights
or credentials are committed.

[Raw archive](raw-evidence.zip) · [verified binding](verified-summary.json)
· [ABBA result](abba/result.json) · [ten checks](abba/checks.json)
· [source audit](kernel-authorship.json).

This is the research-session ABBA. The public saved Run All is an independent
measurement and is recorded in its own evidence folder after completion.
