# Fresh reserve-v2 profile and padding trace

Source `bef6d7a99779593bcd80d43df013ce34561d55b9`, Solution SHA256
`4f1789966593fd5c1e37f36e6fbd0ea7735e34d0a20b8ba811a556699f41f193`.
Все 52 SOURCE_MANIFEST записи сверены с git; все 17 payload files — с ZIP manifest.

Официальный `run_task.py --only solution --profile` сделал один обычный solution run
67.39 с / 3469 MiB и отдельный profiler run. Обычный run не является ABBA.
Profiler active window: 16 шагов; profile wall 129.57 с не является scoring time.
`apply_s` обычного run 8.06 с; `step_s[0]` 6.6339 с. Public ABBA apply 8.30/8.37 с,
step 0 6.7694/6.6530 с. Перемещение импортов само по себе не доказывает ускорение
полной команды: Soup может импортировать те же зависимости позднее.

| Группа CUDA kernels | Доля total kernel time, % |
|---|---:|
| GEMM | 36.781 |
| Activation codec | 11.698 |
| Attention | 9.669 |
| NF4 dequantization | 7.297 |
| Training dtype casts | 5.147 |
| RMSNorm | 2.102 |
| CE | 0.505 |
| Other | 26.801 |

Группировка по именам, это GPU kernel time, не доли wall time и не causal ablation.
[Полный profile](profile.json), [summary с top kernels](verified-summary.json).

Отдельная инструментированная тренировка наблюдала фактические 2D attention masks:
64 батча, 52 890 valid tokens / 91 856 padded slots, **42.4207% padding**.
Батчи/порядок/настройки не менялись. Трассировка добавляет CPU/GPU synchronization;
её время не используется для scoring. [Все формы и длины](padding.json).
MLP и head уже compact; профиль помогает оценить оставшийся attention/QKV.

[Raw ZIP](raw-evidence.zip) SHA256:
`91db62828c7945f0ae4be578807f29745634ca0f2fc8c0f11bf869a0c6cd295b`.
Веса, adapters, task rows, caches и credentials исключены.
