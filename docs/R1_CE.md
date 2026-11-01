# R1: CE и метрики на Triton

Ветка: `codex/research-ce`, задача [#1](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/issues/1).
Это первая реализация R1; она сохраняет штатный `lm_head` и полные logits.
R2 сможет использовать этот API для chunked head отдельно.

## Публичный API

```python
from solution.ce import causal_ce

loss, stats = causal_ce(logits, labels, num_items_in_batch=None)
row_loss, row_tokens = stats.row_losses()
```

- `logits`: `(B, T, V)`, FP16/FP32, конечные значения. Strides могут быть произвольными.
- `labels`: `(B, T)`, int64, на том же устройстве; `[0, V)` или `-100`.
- `loss`: scalar FP32. Без явного count — среднее по supervised shifted labels;
  с count — сумма, делённая на переданный `num_items_in_batch`.
- `stats.token_loss`: detached `(B, T)`, loss позиции `t` предсказывает `labels[t+1]`.
  Последняя позиция и ignored targets имеют ровно нулевой loss.
- `stats.entropy`: detached `(B, T)` по **всем** позициям, включая prompt и последнюю.
  Потребитель применяет `attention_mask`, не assistant mask.
- `stats.correct` и `stats.valid`: boolean `(B, T)`, последняя позиция False.
- `stats.row_losses()`: Python means/counts для rewind; пустая supervised строка даёт 0/0.

All-ignored mean возвращает NaN, как PyTorch, с нулевыми dLogits на ignored positions.
CPU использует обычный PyTorch; CUDA — собственный Triton. BF16 не входит в R1 API.

Для R2: этот API принимает **полную причинно выровненную последовательность**.
Нельзя независимо вызвать `causal_ce` на обрезанных chunks и сложить их: потеряется
цель на каждой границе. Перед подключением R2 нужно согласовать aligned-target API
с look-ahead labels и общей нормировкой. Первую R2 реализацию можно проверять
с reference CE независимо от этого trainer patch.

## Устройство CUDA-пути

`solution/_ce_kernels.py` содержит три оригинальных kernel:

1. `ce_partition_kernel`: для каждой token/vocabulary tile считает maximum, sum(exp),
   weighted centered sum и first-index argmax. Tile 4096 не требует помещать весь словарь
   151936 в один Triton program.
2. `ce_merge_kernel`: объединяет tile statistics через общий maximum, получает log-sum-exp,
   CE, entropy и accuracy. Entropy использует центрированные значения для устойчивости.
3. `ce_backward_kernel`: `(softmax - one_hot) * incoming_gradient` в FP32, затем store
   в dtype logits. GradScaler scale учитывается **до** FP16-приведения.

Большой intermediate log-softmax/softmax не сохраняется. Для backward нужны исходные logits,
labels и маленький LSE tensor. Полный dLogits всё ещё создаётся; это не chunked head R2.

## Подключение

`solution.ce_patch.patch_sft_ce()` патчит только подходящий training-путь Qwen2 с CUDA,
NLL, обычной attention mask, без custom loss, virtual tokens, auxiliary loss или parallelism.
Неподдержанные случаи и eval остаются на исходном trainer path.

HF parent `Trainer.compute_loss` продолжает запускать модель и применять нормировку.
На реальном Qwen instance устанавливается loss callback, активный в контексте этого SFT шага.
После возможного FP32-преобразования Accelerate detached stats прикрепляются к output logits.
TRL metrics используют их; `rewind_hf.row_losses` переиспользует те же per-token losses.
Sampler, FIFO, форматы журнала, trainer settings и адаптер не меняются.

`solution.apply()` идемпотентен. Официальный SwiGLU остаётся отдельным helper;
его kernel time не считается нашим вкладом.

## Проверка

```bash
python -m pytest solution -q --tb=short
python harness/run_task.py --repeats 2
python harness/check_rules.py
```

Проверки R1 сравнивают CE с conventional `S-1` PyTorch reference, entropy с FP64 reference,
маски, ignored rows, ties между tiles, большой словарь, strides, scale 32768 и три реальных
LoRA updates с dropout 0.05. CUDA integration test использует маленький настоящий Qwen
и pinned TRL без загрузки pretrained weights; сверяет loss, metrics, rewind и gradients.

Локально: **14 passed, 16 skipped** в полном `solution/` suite; CUDA недоступна.
Ruff нового кода чистый. На настоящей **Tesla T4, torch 2.11.0+cu128 / Triton 3.6.0**:
**30 passed**, 14 dependency deprecation warnings, 44.54 с.

Первый официальный pilot, один base/solution: **10/10 checks PASS**, включая 79 Soup tests.
Base: 85.92 с / 14345 MiB / eval 2.04967. Solution: 89.49 с / 10005 MiB / eval 2.04965.
Пик уменьшился примерно на 30.3%; полное время выросло на 4.2%.
Средняя кривая loss отличается на 0.01%; first-step gap 0.00%.
Официальный harness сообщает own share 25.7% для шагов 3+.

`train_loop_s` в pilot: 53.92 → 66.01 с. Это не ускорение обучения;
полный запуск и внутренний loop имеют разные границы измерения.

Последовавший **ABBA v1 не прошёл check 8**: полное время x1.139 к базе
(лимит x1.10). Остальные 9/10 checks PASS; память осталась 10005 MiB,
own share min 25.3%. Положительный одиночный pilot не заменяет этот результат.

Во **v2** длина sequence и зависящие от неё strides передаются как runtime scalars
с `do_not_specialize`. В v1 их constexpr создавали новые компиляции при смене длины
batch, в том числе после первых двух шагов. Алгоритм и objective не менялись.
CUDA suite v2: **30 passed, 14 warnings in 21.40s**.

**Pilot v2: 9/10 checks PASS**, единственный FAIL — собственные kernels дают 0.9%
вместо требуемых 5%. Base: 78.05 с / 14345 MiB / eval 2.04966.
Solution: 75.44 с / 10005 MiB / eval 2.04967. Speedup 1.0346, memory ratio 1.4338,
eval delta округляется официальным harness до +0.000%. Кривая и выходы Soup проходят;
79 regression tests PASS. Время CE kernels на шагах 3+ — 411.872 мс из 46.022 с.

В v1 тот же счётчик показывал около 15.3 с: повторные компиляции попадали в окно
CUDA events. Высокие 25% v1 не следует трактовать как долю чистой GPU-арифметики.
Для финального решения выбираем нормальный runtime-shape вариант и расширяем
полезный охват kernels вместе с R2/R3; возвращать лишние компиляции ради квоты нельзя.

V2 отдельно пока не допускается. Повторять ABBA V2 без расширения полезного вклада
не стали: уже известен FAIL check 4. Исходные результаты и оба исходника сохранены в
[evidence/r1-t4-20261030](../evidence/r1-t4-20261030/README.md).

## Ограничения и источники

Ни одна версия не прошла все условия устойчивой сдачи: v1 ABBA нарушает лимит времени,
v2 pilot нарушает own-time quota. R1 подтверждает корректность CE/metrics и уменьшение
памяти в комбинации со starter SwiGLU, не готовность конкурсного решения.
Выигрыш времени v2 <5%, ниже порога баллов; это один pilot, не статистический speedup.
Вклад starter SwiGLU в память не отделён отдельным ablation.
Не переносим результаты на другие модели, GPU, seeds или данные без проверки.
Second-order gradients не поддерживаются этим экспериментом.

Интеграционная семантика сверена с:

- [Transformers 5.16.1 causal loss](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py).
- [TRL 0.29.1 compute_loss](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py).
- [Soup 0.77.0 rewind](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/rewind_hf.py).
- [H2 evidence](https://github.com/SoupTeam/soup-rnd/tree/research/h2-kernel-comparison) — мотивация
  проверить CE отдельно; ни Liger, ни другие внешние kernel implementations не копировались.
