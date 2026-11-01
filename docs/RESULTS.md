# Журнал результатов

Текущий квалифицированный резерв — [`reserve-v2`](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/tree/reserve-v2),
source `bef6d7a`: ABBA 79.94 → 68.09 с, 14345 → 3469 MiB, eval +0.123%,
own min 17.8593%, 10/10 checks и 211 T4 tests.
[Проверенные исходные данные](../evidence/final-t4-20261030/README.md).
Отдельная пара seed 7 также прошла 10/10: 74.67 → 65.10 с, 14351 → 3433 MiB,
eval +0.118%; это один repeat, не ABBA.
[Seed 7](../evidence/final-t4-seed7-20261030/README.md).
Числа ниже сохраняют историю более ранних кандидатов.

Первый компонент R1 проверен на Kaggle T4. Подробные raw данные и исходники:
[evidence/r1-t4-20261030](../evidence/r1-t4-20261030/README.md).
Числа организаторов и старые R&D-результаты не являются результатами этой команды.

| Кандидат | Tested commit | Notebook/version | Время базы/наше, с | Peak базы/наш, MiB | Eval delta, % | Own share min, % | Checks | Решение |
|---|---|---|---|---|---|---|---|---|
| Starter / baseline | — | — | — | — | — | не авторский kernel | не запускались | только стартовый пример |
| R1 v1 pilot | 00d00d6 | Kaggle R1 | 85.92 / 89.49 | 14345 / 10005 | −0.001 | 25.7 | 10/10 | только один pilot |
| R1 v1 ABBA | 00d00d6 | Kaggle R1 | 79.32 / 90.36 | 14345 / 10005 | −0.000 | 25.3 | 9/10 | FAIL time +13.9% |
| R1 v2 runtime shapes pilot | 2e2a4b5 | Kaggle R1 | 78.05 / 75.44 | 14345 / 10005 | +0.000 | 0.9 | 9/10 | FAIL own share |
| R1+R2, chunk 128, ABBA | 3527e32 | Kaggle R2 v2 | 85.82 / 81.75 | 14345 / 8305 | +0.001 | 0.8 | 9/10 | FAIL own share |
| R1+R2+casts, chunk 256, ABBA | b0dcbfe | Kaggle R2 v5 | 83.06 / 79.97 | 14345 / 8445 | +0.000 | 10.5 | 10/10 | R2 завершён; сохранён резерв |
| R1+R2+INT8, pilot | source snapshot в evidence | Kaggle impact | 75.83 / 78.84 | 14345 / 5063 | +0.002 | 6.7 | 10/10 | допущенный pilot |
| R3 RMSNorm | — | — | — | — | — | — | не запускались | исследование |
| Комбинация + lossless storage, ABBA | fda492a | Kaggle 363832117 | 79.87 / 80.12 | 14345 / 4203 | +0.000 | 6.6616 | 10/10 | допущенный резерв; speed tie |

Notebook: https://www.kaggle.com/code/koko667/setww-r1-ce-metrics-t4
Все три R1 фазы в одном архиве evidence с manifest и ссылками на tested commits.
Изменения памяти включают starter SwiGLU; это не CE-only ablation.

[Исходные результаты R2](../evidence/b0dcbfe10d7d8fbd47aa0d8fe7d69f978e26d62c/r2-t4-final-v5/README.md)
и [его отчёт](R2_CHUNKED_HEAD.md) сохранены отдельно от итогового стека #9.
R2 и #9 измерены в разных сессиях: абсолютные секунды между ними не доказывают
вклад отдельного компонента. #9 с peak 4203 MiB сохранён как предыдущий резерв.

Новые строки после замера ссылаются на `evidence/<tested-sha>/<experiment>/`.
Для отрицательного результата сохраняем причину и исходные данные.
Не объединяем замеры с разных GPU/конфигураций или несопоставимых baseline.

Допуски: eval ≤+0.5%, first loss gap ≤1%, mean curve gap ≤3%,
metrics/rewind по правилам, время/peak ≤1.10 базы, own share ≥5%, checks 10/10.
Публичные проверки не гарантируют результат скрытого набора.

[Итоговые ABBA evidence](../evidence/impact-t4-20261030/README.md). Кэш nested scales: solution-only 82.59 s / 5141 MiB, не включён.
