# T4: объединённый CE, чанковая голова и хранение активаций

Статус: **финальный для этой версии кода ABBA прошёл 10/10 checks**.
14345 → 4203 MiB (3.413×), 79.87 → 80.12 s, own share min 6.6616%,
eval +0.000% по harness. 183 tests на T4. Ускорение пока не получено.
[Проверенные raw evidence](../evidence/impact-t4-20261030/README.md).

Таблица ниже — **предыдущий pilot без frozen storage**, не итоговый ABBA.

| Показатель, один base + один solution | Base | Solution |
|---|---:|---:|
| Полная команда train, s | 75.83 | 78.84 |
| nvidia-smi peak, MiB | 14345 | 5063 |
| Eval loss, без наших патчей | 2.04967 | 2.04971 |
| Собственные kernels / wall шагов 3+ | — | 6.7% |

Память base/solution = **2.8333×**, уменьшение peak на **64.7%**.
Время solution/base = 1.040; speedup = 0.9618×.
Eval loss отличается на +0.002%; первый train loss — на 0.00%, средний gap кривой — 0.01%.
Все метрики, порядок строк, token counts, rewind и форма адаптера прошли официальный checker.
Эти числа относятся к объединённому решению, включая разрешённый starter SwiGLU.
Это не изолированная оценка вклада CE или сжатия.

## Что изменено

1. R1 вычисляет CE, entropy и первый argmax через оригинальные Triton reductions
   по блокам словаря 4096. Backward умножает на входящий GradScaler scale в FP32
   **до** записи FP16 gradient. Размер последовательности и меняющиеся strides — runtime scalars.
2. R2 вычисляет frozen output head группами по 512 **плоских позиций**.
   Forward сохраняет компактные статистики; backward пересчитывает logits только
   для supervised позиций с исходным autocast и глобальным знаменателем loss.
   Удаляется только padding внутри того же batch; строки между batch не смешиваются.
3. Новый `aligned_ce` принимает уже сдвинутые targets. Последняя реальная позиция
   каждого чанка участвует в loss; отдельный reference CE на хвосте больше не нужен.
   Entropy включает prompt и последнюю attention-позицию исходной строки.
4. Оригинальные INT8 kernels хранят крупные сохранённые для backward активации
   в блоках 256, с FP32 scale = absmax/127. Live forward tensors и параметры не меняются.
   Ошибка восстановления одного значения до итогового dtype cast ограничена absmax/254.
   Это **lossy storage**, его влияние на обучение проверяется полным harness.
5. Hooks ограничены training decoder, CUDA FP16/FP32, non-leaf tensors с requires_grad,
   размерностью 2/3 и минимум 262144 элементами. Параметры и 4-D attention tensors
   сохраняются исходными. Метаданные и кэш не удерживают исходные активации;
   weak references и version counter проверяют lifetime и in-place изменения.

Ни один встроенный режим Soup не включён. NF4/double quant, LoRA r16/alpha32/dropout0.05,
optimizer, RNG, batch, sampler, число шагов и trainer settings остаются заданными задачей.
Генерация/eval и неподдерживаемые варианты головы используют исходный model forward.
`harness/`, `task/`, `tests/` сохранены без изменений.

## Воспроизведение

Зависимости и неизменённые официальные команды — в [RUNBOOK](RUNBOOK.md).
Kaggle T4 x2, используется GPU 0, FP16; Internet включается до загрузки модели и данных.
Несовместимый предустановленный `torchao==0.10.0` удаляется до измерений для обеих сторон.

```bash
SOUP_APPLY_SOLUTION=1 python -m pytest solution tests -q --tb=short
python task/prepare_data.py
python harness/run_task.py --profile --out runs_nonattention_pilot
python harness/check_rules.py --result runs_nonattention_pilot/result.json
```

На настоящей T4: **179 passed**, включая все 79 Soup tests и CUDA проверки.
Профиль — отдельный диагностический запуск на 24 шага; его 128.2 s и peak 5075 MiB
не включены в таблицу выше. Каждый официальный timed run начинает с пустыми compiler caches.

Полный `--repeats 2` и checker без skip-tests выполнены позднее: [итоговый ABBA](../evidence/impact-t4-20261030/README.md). Для окончательной сдачи остаётся публичный Kaggle Save & Run All и проверка final version URL.

## Отрицательные результаты

- Обычный `tl.dot` для головы на Triton 3.6/T4 оказался непригоден:
  лучший forward для M512/K1536/N151936 — 211.78 ms против 12.04 ms у PyTorch.
  В этой версии компилятора SM75 идёт по FMA fallback, а не Tensor Cores.
  Реализация и bounded microbench сохранены в `research/rejected_head_gemm/`.
  В runtime `solution/` этих kernels нет.
- Первый вариант с INT8 для всех крупных saved tensors получил NaN Q/K gradients
  и отказ Soup сохранять финальный адаптер. Source и восстановленные values были конечными.
  Двухшаговая изоляция: без сжатия, только MLP и с исключением 4-D attention — конечные
  градиенты; широкий вариант — NaN. Поэтому финальный на этом этапе охват исключает 4-D.
  Причина сбоя attention ещё исследуется; не выдаём предположение за установленный факт.
- R1 CE отдельно имел лишь 0.9% own share и 9/10 checks; это не допущенное решение.
  Новый полезный охват — хранение реально потребляемых backward активаций.
  Никаких задержек, лишних повторов или повторной JIT-компиляции ради квоты не добавлено.

## Происхождение и границы

- Official starter: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
- R1: [PR7](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/pull/7),
  исходная проверенная ветка `2426ab72daa407f41a4b46ff8aa35bf30d33ddac`.
- R2: [PR8](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/pull/8),
  chunked-head source `3527e32abe8bdd92c447232250bb1a622fc7541e`;
  здесь добавлены aligned CE, backend capture и lazy imports. Чужая ветка не перезаписывается.
- Идея saved-tensor storage: официальный
  [PyTorch API](https://docs.pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.saved_tensors_hooks).
  Block quantization и kernels написаны для этого эксперимента; готовые kernel libraries не используются.
  Сжатие активаций/int8 прямо предложено в разделе 9 исходного DOCX ТЗ.
- Причина отказа от обычного GEMM:
  [Triton 3.6 AccelerateMatmul](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L346).
- Starter SwiGLU сохранён и отдельно отмечен как пример; он не участвует в own-time quota.

Проверен только этот fixed Qwen/NF4 task на T4. Скрытые строки и другой seed не проверялись;
допуск на них и итоговое решение жюри не следуют из одного публичного pilot.

## Lossless frozen embedding/head storage

`solution/frozen_storage.py` проверяет exact FP16 round-trip всей frozen tied матрицы небольшими блоками. Shared Parameter остаётся тем же объектом; embedding hook возвращает FP32 до остальных input hooks. Head вне CUDA FP16 autocast восстанавливает исходный FP32 compute. Trainable, untied или непредставимые веса не меняются.

На T4 проверены точные logits и input gradients с/без AMP, отрицательный случай непредставимого значения и полный ABBA. Проверенный source commit: `fda492acf93f67815a89c46120b362ca9821f56c`.
