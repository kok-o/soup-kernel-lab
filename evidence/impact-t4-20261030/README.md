# Проверенный T4 ABBA, 30 октября 2026

Code: `fda492acf93f67815a89c46120b362ca9821f56c`. [Kaggle saved development version](https://www.kaggle.com/code/koko667/setww-impact-t4?scriptVersionId=363832117).
Это private Quick Save исследовательской сессии, не финальный public Save & Run All.

**10/10 checks**, memory 14345 → 4203 MiB (3.413×), median time 79.87 → 80.12 s,
eval 2.04964 → 2.04965, own share min 6.6615509%. 183 tests passed на T4.

[summary.json](summary.json), [result.json](result.json), [checks.json](checks.json).
Все исходные попытки, их source snapshots и logs: [raw-evidence.zip](raw-evidence.zip).
Архив содержит 161 проверенный файл; модели, adapters, task rows и compiler caches исключены.

| ABBA run | Command time, s | nvidia-smi peak, MiB |
|---|---:|---:|
| base_1 | 75.59 | 14345 |
| solution_1 | 77.59 | 4203 |
| solution_2 | 82.65 | 4203 |
| base_2 | 84.15 | 14345 |

Время в summary — медиана двух запусков, память — максимальный peak.
Не выдаём разброс отдельных запусков за ускорение. Профиль первого pilot не включён в итог ABBA.

- Archive SHA256: `cf19fec5890d0bc8963bbd2297314bbacf4b62713352bc01399c7c282732ba08`
- Result SHA256: `ea4badfdf647d85fd67157587ed33439aa9d0016152920fb821ec0e8ba643655`
- Solution SHA256: `6f722767b6723facf60b9ac6f4e0139027a2c548a087de702b75525b9940a1a4`

Локально проверены manifest всех 161 файлов, точное совпадение 16 source-файлов финального
GPU run с GitHub commit и result/checks hash binding. Все 15 official blobs не изменены.

В архиве сохранены отрицательные GEMM/FMA, full-activation INT8/attention-layout NaN,
scale-cache screen и NF4 cast diagnostics. Для повтора старой попытки сначала восстановить
её `solution-source.zip`; не запускать старую гипотезу на текущем runtime и не смешивать числа.

Сжатие активаций lossy; frozen embedding storage lossless с проверкой каждого значения.
Starter SwiGLU помогает общему стеку, но исключён из авторских долей.
