# Доказательства измерений

## Сохранённые результаты

- [Выбранный стек #9, полный T4 ABBA](impact-t4-20261030/README.md):
  14345 → 4203 MiB, 10/10 checks; текущий итоговый кандидат.
- [R1 CE: pilot, отрицательный ABBA и runtime-shape версия](r1-t4-20261030/README.md).
- [Завершённый R2, T4 ABBA v5](b0dcbfe10d7d8fbd47aa0d8fe7d69f978e26d62c/r2-t4-final-v5/README.md):
  14345 → 8445 MiB, 10/10 checks; исторический резерв, основа head включена в #9.
- [R2 без casts, FAIL own share](3527e32abe8bdd92c447232250bb1a622fc7541e/r2-t4-abba-v2/README.md),
  [pilot 128 с casts](80fb6e53b2c2d3dcc68c51ca43109e5d3e0816f1/r2-t4-pilot-v4/README.md),
  [отрицательный NF4 fixture](0cdf2c10c7e86b6eda08208f0e68a4fe848115ef/r2-t4-v3-negative-test/README.md),
  [environment preflight](3527e32abe8bdd92c447232250bb1a622fc7541e/r2-kaggle-preflight-v1/README.md)
  и [CPU prototype/interop](e03855af986bded5be3fc5bd85c584abe296c141/r2-cpu/README.md).

Материалы R2 перенесены из `51455595a1d1d621bb0804eca923b0fdaafb038a`
без изменения исходных blobs. Его [pinned notebook](../templates/r2_verified_notebook.ipynb)
воспроизводит `b0dcbfe`; для выбранного стека используется
[impact_t4_repro.ipynb](../templates/impact_t4_repro.ipynb).

## Формат новых измерений

После запуска сохранять небольшие исходные файлы под:

```text
evidence/
  <tested-commit>/
    <experiment>/
      README.md
      result.json
      checks.json
      train.log
      eval.log
```

README эксперимента содержит commit, notebook/version URL, GPU, версии,
команду, вариант R1/R2/R3, failed checks и границы выводов.
Допускается несколько исходных train/eval logs с оригинальными именами.

Не загружать caches, adapters, модели, данные и секреты.
Пояснения писать отдельно, значения исходных JSON не менять.
