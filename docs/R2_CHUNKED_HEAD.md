# R2 · @Nari-Ab · исследование завершено

Задача [#2](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/issues/2) выполнена:
проверена гипотеза уменьшения GPU peak через chunked frozen `lm_head`, реализован
scaled backward, сохранены метрики и rewind, проведён официальный T4 ABBA.
Исходная работа — [PR #8](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/pull/8).

Основа R2 уже включена в итоговый [PR #9](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/pull/9)
с сохранённым ancestry `3527e32abe8bdd92c447232250bb1a622fc7541e` и авторством
@Nari-Ab. В `main` выбран стек R1 + R2 + saved activation/frozen storage:
**14345 → 4203 MiB, 10/10 checks**. [Результат и ограничения](IMPACT_T4.md).

Старый entry point PR #8 не переносится поверх принятого решения. Из PR #8
сохранены исходные результаты, положительные и отрицательные опыты, provenance
и самодостаточный notebook. Это завершённое исследование и воспроизводимый
резервный кандидат; отдельного ускорения или повторной реализации от R2 не требуется.
Новые исследования скорости требуют отдельной задачи.

## Что проверил R2

- Frozen bias-free head вычисляется группами плоских позиций `[batch*seq]`.
  Полные `[B,T,V]` logits не сохраняются между forward и backward.
- На forward/backward удерживается одна FP16 compute-копия frozen head.
- Backward пересчитывает только supervised позиции; incoming AMP scale и общий
  denominator действуют до низкоточного округления logits gradient.
- Shifted targets, `-100`, entropy prompt/последнего токена, accuracy, token counts,
  LoRA updates/RNG и per-row rewind сохранены. Встроенный `stream_loss_chunk` не включён.
- Исследованы chunks 128/256; финальный R2-кандидат использует 256, `compact=False`.
- Triton dtype casts заменяют реальные NF4 input/output casts. Этот дополнительный
  вариант измерен в R2; в текущий runtime `main` он не подключается.

Текущий интегрированный `main` использует aligned CE, chunk 512 и `compact=True`.
Его результаты относятся ко всему стеку и не являются изолированной ablation R2.
Исходный causal CE adapter R2 сохраняется в истории и notebook; он считает
последний реальный target каждого chunk reference CE, не теряя границу.

## Результат исходного R2-кандидата

Измеренный source: `b0dcbfe10d7d8fbd47aa0d8fe7d69f978e26d62c`.
Отчётный head PR #8: `51455595a1d1d621bb0804eca923b0fdaafb038a`;
все файлы `solution/` совпадают с измеренным source.

| Показатель | Base | R2-кандидат |
|---|---:|---:|
| Полная команда train, медиана ABBA, с | 83.06 | 79.97 |
| Максимальный nvidia-smi peak, MiB | 14345 | 8445 |
| Public eval loss | 2.04968 | 2.04968 |
| Собственные kernels, минимум двух запусков | — | 10.5% |
| Официальные проверки | — | 10/10 PASS |

Снижение peak — 41.13%, memory ratio 1.6986×. Speedup 1.0386× ниже порога
1.05, поэтому баллы за скорость не заявляются. Время ABBA: 82.94 / 79.97 /
79.96 / 83.18 с. Это совместный кандидат R1 + R2 + casts + starter SwiGLU;
SwiGLU исключён из авторской доли. Скрытый seed не проверялся.

[Raw JSON, логи, source manifest и исходный архив](../evidence/b0dcbfe10d7d8fbd47aa0d8fe7d69f978e26d62c/r2-t4-final-v5/README.md).
[Kaggle notebook, сохранённая версия 5](https://www.kaggle.com/code/narimanabdikarim/setww-r2-abba-3527e32)
остаётся private; исходные outputs доступны в репозитории.

## Проверки и воспроизведение

На T4 сохранены **93 solution tests** и **79 официальных регрессий** с патчами.
При review повторены CPU-проверки: **147 passed / 25 CUDA skips**, Soup 0.77.0,
Transformers 5.16.1, TRL 0.29.1, PEFT 0.20.0, Accelerate 1.14.0,
Python 3.13.3 / Torch 2.14.0+cpu. Это отдельная проверка корректности;
GPU performance берётся только из исходного Kaggle T4 / Torch 2.11.0+cu128 ABBA.

Сверены хеши всех 27 файлов manifest с git blobs, `checks.result_sha256`,
`checks.solution_sha256`, raw archive и все 29 встроенных файлов notebook.
`harness/`, `task/`, `tests/` исходного R2 не изменены относительно starter команды.
По raw данным повторно проверены loss curve, метрики, rewind и own share
10.479% / 10.482% без starter SwiGLU.

Для повторения R2 импортировать [исторический pinned notebook](../templates/r2_verified_notebook.ipynb)
в Kaggle, выбрать T4 x2, включить Internet и выполнить Run All. Исходники
измеренного SHA, LICENSE/NOTICE встроены; GitHub token не нужен.
Он воспроизводит R2-кандидат, а для итогового решения команды используется
[impact_t4_repro.ipynb](../templates/impact_t4_repro.ipynb).

История экспериментов сохранена, включая:

- [CPU prototype и R1 interop](../evidence/e03855af986bded5be3fc5bd85c584abe296c141/r2-cpu/README.md).
- [Неудачный environment preflight](../evidence/3527e32abe8bdd92c447232250bb1a622fc7541e/r2-kaggle-preflight-v1/README.md).
- [ABBA без casts: FAIL own share 0.8%](../evidence/3527e32abe8bdd92c447232250bb1a622fc7541e/r2-t4-abba-v2/README.md).
- [Отрицательный odd-width NF4 fixture](../evidence/0cdf2c10c7e86b6eda08208f0e68a4fe848115ef/r2-t4-v3-negative-test/README.md).
- [Pilot chunk 128 с casts: 10/10 PASS](../evidence/80fb6e53b2c2d3dcc68c51ca43109e5d3e0816f1/r2-t4-pilot-v4/README.md).

Источники идей: [Liger FLCE](https://arxiv.org/abs/2410.10989),
[Soup chunked head](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/utils/stream_loss_chunk.py),
[Torch autocast cache](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp).
Код R2/casts написан @Nari-Ab, R1 kernels сохранены с авторством @kok-o.
