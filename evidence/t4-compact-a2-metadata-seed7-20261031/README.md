# A9 independent seed7: official Kaggle T4 ABBA/all10

Same complete byte-identical source as seed1 qualification: a9ac67e,
canonical SHA53d2057ac60abda2d7ec00a441c91d5b20bb9551973a874582cb1e12e98e338d.

Seed7: **57.59 s / 2237 MiB, all10 checks pass**.
Stock80.10s/14351MiB; train loop32.80s versus55.94s.
Eval2.05379 versus2.05117, quality delta+0.128%.
Seed1 remains57.89s/2307MiB/all10 and quality+0.120%.
The lower seed7 peak is a separate batched data order, not a new memory optimization.

The original harness seed parameter was used for both sides with unchanged data/task.
Every run's recorded trainer seed is7; two repeats on both sides, empty compiler caches.
The official checker ran without skips, including its Soup regression gate.
The same-source full348pass/2optionalBLASskip suite was already qualified in seed1;
it was not redundantly rerun before this additional seed. No test threshold was altered.

All38 manifest entries, absence of unlisted entries, complete source against Git,
source/result/check binding and all10 results were independently verified.
Original train/eval logs, metrics, task configs and checker output are in raw-evidence.zip.
This interactive qualification is distinct from the saved public Run All release.
