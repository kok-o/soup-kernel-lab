# Brev Tesla T4 full cold train + separate eval

These are actual Brev measurements, not Kaggle wall times or official ABBA.
Both variants ran unchanged harness/run_task.py --only solution, with fresh
compiler caches and full parent-timed startup/import/JIT/model/train/save.
The GPU lock prevents concurrent GPU work. Each is one screen, not a paired claim.

| Implementation | Source | Cold train s | NVIDIA MiB | Loop s | Eval loss | Tests |
|---|---|---:|---:|---:|---:|---:|
| Compact dropout, offload10 |35ea24c|47.81|2307|32.41|2.05208|302/302|
| Same runtime, offload14 |c7ccd89|49.31|2207|33.89|2.05207|302/302|

Original full train logs, eval JSON/logs, metrics,302-test logs and complete
solution source zips for both commits are retained in raw-evidence.zip.
Both source snapshots match Git byte for byte. All manifest entries and canonical
solution/result hashes were independently verified. verified-summary.json records
the complete commits, bindings and archive checksum; it labels these as Brev.

Separately,35ea24c passed official Kaggle ABBA61.83s/2307MiB/10of10,
quality+0.118%, source3172cf8b138a331262bb63a471a31f67e2936336bb78ca13fc5192ade6a46ce5.
See ../t4-compact-dropout-20261031/. The47.81s Brev time must not be presented
as Kaggle speed. The14-layer branch has no Kaggle qualification yet.
