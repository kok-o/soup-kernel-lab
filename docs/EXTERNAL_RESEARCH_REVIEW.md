# Проверка предложенного расширенного стека

30 октября 2026. Предложенный план — набор гипотез, не воспроизведённый T4 результат.
Оценки 44–49 s, 2.15–2.45 GiB, 18–28% own share и 98–100 баллов не подтверждены логами.
Результаты соперников в его таблице также не измерены.

## Что подтверждается и что требует исправления

| Предложение | Проверка | Решение |
|---|---|---|
| Убрать якобы сохранённый A из MatMul4Bit | В bnb 0.50.2 обычный forward уже хранит `(None, B)`, вход A не сохраняет | Обещание освободить этим ~1.8 GiB неверно для заданной версии |
| Своя NF4 распаковка и fusion Gate+Up/QKV | Возможное уменьшение dispatch/GEMM overhead; готового замера нет | Полезное следующее исследование, отдельно от ошибочного аргумента про A |
| Frozen embedding/head в FP16 | PEFT действительно поднимает non-4bit параметры в FP32 | Проверять lossless storage с FP32 embedding output и прежней политикой head |
| Async CPU offload | Свой offload разрешён, но обмен PCIe и pinned buffers стоят времени | Не обещать нулевой overhead или 2.2 GiB без полного замера |
| Полное unpadding + Varlen GQA | Убирать padding внутри batch разрешено | Сначала измерить реальные padded shapes; нужны RoPE, masks, dropout/RNG и backward tests |
| Varlen attention на Tensor Cores через обычный tl.dot | Triton 3.6 на SM75 делает FMA fallback | В таком виде не подходит нашему стеку; нужен иной проверенный backend/собственный MMA путь |
| Forward-Grad вместо повторного расчёта logits | Можно убрать повторный head projection; dHidden GEMM остаётся | Учесть реальный AMP scale до FP16 gradient cast и проверить LoRA updates |
| Runtime sizes, фиксированные configs, cold JIT | Правильное направление; в R1 v2 это уже сделано | 1.0–1.3 s компиляции не гарантированы; проверять каждый cold run |

## Существенные численные границы

Просто сделать embedding FP16 и пустить дальше FP16 residual не означает сохранить
прежний расчёт. Qwen RMSNorm приводит нормированные значения к input dtype **до**
умножения на weight; при FP32 norm weight результат снова может стать FP32.
Меняются промежуточные округления, а обещание перевести весь residual stream в FP16 не следует
из одной замены embedding. Наш проверяемый вариант оставляет embedding output FP32,
сохраняет shared Parameter identity и проверяет каждый вес exact FP16 round-trip.

Forward-Grad нельзя делать как unscaled FP16 `dLogits`, затем умножать dHidden на
grad_output. При большом словаре/знаменателе часть `p / N` округлится в ноль раньше AMP scaling.
Умножение после GEMM этого не восстановит. Нужно перенести настоящий scale до FP16 store,
сохранить глобальный знаменатель, учесть gradient accumulation и проверить смену loss scale.
Текущий R2 специально пересчитывает backward с реальным incoming scale.

QKV/Gate+Up fusion меняет формы GEMM и порядок FP16 округления dHidden относительно
суммы отдельных FP16 продуктов, поднятых в FP32. Это проверяемая оптимизация, не автоматически
битовое совпадение. Отдельно нужны bias, q/v LoRA, dropout и версия NF4 quant-state.

## Числа, которые нельзя принимать как доказательство

- 52890 logged tokens — число реальных токенов. Оно не задаёт сумму `batch_size * padded_length`:
  длина batch меняется. Утверждение о >50% padding требует самих shapes.
- Если считать 23.72 s неизменяемым overhead, арифметический предел при нулевом train loop:
  `78.05 / 23.72 ≈ 3.29×`, а не 1.7–1.9×. Сам overhead ещё не доказан неизменяемым.
- Число запусков kernels не определяет own share. Checker делит CUDA-event time собственных
  kernels на wall time шагов 3+ и берёт минимум по повторам.
- Прогнозы 2.2 GiB и zero-overhead offload требуют измерить PCIe, overlap, число
  одновременно живых буферов и полный cold command. Одних размеров входов слоя недостаточно.

## Подтверждённый результат этой команды на момент разбора

R1 + R2 + INT8 saved activations без 4-D attention:
**10/10 pilot checks**, 14345 → 5063 MiB (2.8333×),
75.83 → 78.84 s, eval +0.002%, own share 6.7%, 179 тестов на T4.
Это один pilot, не финальный ABBA. Сжатие 4-D attention отвергнуто из-за NaN;
даже layout-only contiguous copy воспроизводит сбой. Причину не объявляем установленной.

Кэш распакованных nested scales прошёл 182 core tests, но solution-only screen дал
82.59 s / 5141 MiB. Это не парное сравнение и не доказанное ускорение; основной кандидат сохранён.
Экспериментальный cast NF4 на искусственной ширине 257 выявил misaligned address
уже в **исходном** bitsandbytes reference при CUDA_LAUNCH_BLOCKING=1. Это не доказанный баг
Triton cast; нужны проверки на реальных выровненных размерах Qwen.

## Проверенные первичные источники

- [bnb 0.50.2 MatMul4Bit, сохранение и backward](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py#L273)
- [bnb 0.50.2 dequantize_4bit и nested scales](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py#L915)
- [PEFT 0.20.0 prepare_model_for_kbit_training](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py#L142)
- [Qwen2 RMSNorm и residual, Transformers 5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L215)
- [Triton 3.6: FMA fallback для compute capability <80](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L346)
- [Официальные правила](STARTER_GUIDE.md), [наш T4 pilot](IMPACT_T4.md)

## Уже проверенная доработка по мотивам плана

Lossless frozen storage с FP32 embedding output реализован и прошёл полный ABBA: **14345 → 4203 MiB, 3.413×, 10/10 checks**, 79.87 → 80.12 s, own min 6.6616%, 183 tests. Это фактический выигрыш, а не прогноз 2.2 GiB.

Отдельный `research/probe_cast_aligned.py` проверил три выровненных NF4 формы (128→256, 1536→1536, 1536→8960): **3 passed на T4**, точные outputs/gradients. Это уточняет прежний сбой stock reference на ширине 257. Casts остаются вне runtime до отдельного полного сравнения времени.
