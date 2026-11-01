# Исследовательский план для T4

Обновление 30 октября: интегрированы frozen FP32 RMSNorm, compaction MLP,
FP16 MLP intermediates, hybrid activation storage и forward-gradient head.
Квалифицированный [`reserve-v2`](FINAL_T4_RESEARCH.md): ABBA 68.09 с / 3469 MiB,
eval +0.123%, own min 17.8593%, 10/10 checks. Seed 7 отдельно прошёл 10/10.
Ниже — история первоначальных гипотез; вклад каждого компонента не изолирован ABBA.

Цель: допущенный конкурсом выигрыш по полному времени или peak GPU memory.
R1 реализован и проверен на T4: [результаты и ограничения](R1_CE.md).
V2 даёт 9/10 checks; собственный вклад 0.9% при требуемых ≥5%.
Объединённый R1+R2+activation storage прошёл полный T4 ABBA: 10/10 checks,
14345 → 4203 MiB, own min 6.6616%. [Результат](IMPACT_T4.md).
Этот ранний допущенный вариант сохраняем как предыдущий резерв.

## R1 — CE и метрики

**Вопрос:** можно ли убрать повторные чтения logits и временные tensors при вычислении
cross-entropy, entropy и accuracy? Сначала штатный `lm_head` и полные logits сохраняются.

Forward/backward — собственный Triton. Loss и accuracy используют shifted labels
и маску `-100`; entropy учитывает все непустые attention positions, включая prompt
и последнюю позицию. Нужны правильный знаменатель loss и per-row данные rewind.
Проверяем FP16/FP32, реальные strides, крайние logits, ignored labels и incoming grad scale.

[Задача #1](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/issues/1).

## R2 — chunked head

**Вопрос:** можно ли уменьшить peak, вычисляя frozen `lm_head` по небольшим группам
токенов и сохраняя только нужные статистики?

Начать с reference CE независимо от R1. Размеры 128/256 — начальные гипотезы,
не предписанные настройки задачи. В backward допустим собственный пересчёт,
его время включается в замер. Incoming loss scale следует учитывать до
низкоточного приведения. Модель, trainable parameters, RNG и порядок batch сохраняются.

Согласовать с R1: scalar loss; per-token loss/correctness для shifted labels;
entropy по attention mask; данные для per-row rewind; нужные градиенты.
Формы, dtype и reduction denominator фиксируются по реальному baseline.

[Задача #2](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/issues/2).

**Исследование завершено.** R2 проверил reference, AMP backward, метрики/rewind,
chunks 128/256 и официальный T4 ABBA: 10/10 checks, 14345 → 8445 MiB.
Его chunked head уже используется в принятом #9 с дополнительными оптимизациями
хранения (4203 MiB для всего стека). [Отчёт и воспроизведение](R2_CHUNKED_HEAD.md).
Исходные данные R2 сохраняются как отдельный измеренный кандидат; новые задачи
ускорения не требуются для завершения этой гипотезы.

## R3 — RMSNorm

**Вопрос:** можно ли ускорить Qwen RMSNorm и уменьшить затраты на приведения типов?
Начальная форма — hidden width 1536, но runtime dtype/eps сверяются с baseline.

Сохраняем FP32 reduction и фактическое dtype promotion от weight.
Проверяем backward и влияние на полный запуск, затем полезность вместе с R1/R2.
Само наличие RMSNorm не гарантирует нужные ≥5% времени.

[Задача #3](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/issues/3).

## Почему выбраны эти направления

| Исследование Soup R&D | Что используем | Что не переносится как доказательство |
|---|---|---|
| [H2](https://github.com/SoupTeam/soup-rnd/tree/research/h2-kernel-comparison) | Узкая граница CE и проверки градиентов | Liger-код, exact одной микропорции и результат на RTX4090/BF16 не доказывают T4 скорость/качество |
| [D2 / PR5](https://github.com/SoupTeam/soup-rnd/pull/5) | Проверки autocast, округления, backward и updates | Llama/dropout-free FastLoRA и MoE/SSD gains не являются ready-made путём для resident Qwen/NF4 |
| [D2 RMS](https://github.com/SoupTeam/soup-rnd/blob/koko/d2-liger-streaming/src/soup_cli/utils/_liger_streaming_kernels.py) | Идеи сохранения арифметики RMSNorm | Liger integration и width4096 результат не квалифицируют Qwen1536 |
| [Chunked Soup 0.77.0](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/utils/stream_loss_chunk.py) | Архитектура передачи метрик без полного logits | Нельзя просто включить встроенный режим |
| [D1](https://github.com/SoupTeam/soup-rnd/blob/main/benchmarks/D1_ACCEPTANCE.md) | Знания о NF4 и численных рисках | Положительные WMMA/BF16 memory results не доказывают FP16/T4 Triton backend |
| [Исторический CCE](https://github.com/SoupTeam/soup-rnd/blob/main/benchmarks/probe-v0.73.0-what-bounds-streaming.md) | Мотивация уменьшить большие tensors | Speedup при batch 1→3 нельзя переносить в фиксированный batch4 конкурса |

## Что не является критерием успеха

Не требуется побитовое совпадение любой промежуточной операции:
решение должно пройти официальные допуски качества, кривой и метрик.
Однако совпадение loss без проверки backward и updates недостаточно.
Ускорение микрокернела или падение `torch.cuda.max_memory_allocated`
не заменяет full-command time и nvidia-smi peak.

## Происхождение кода

Новые kernels пишем самостоятельно. Для каждого финального kernel указываем автора,
источники идей, путь, операцию и измеренную долю времени в SUBMISSION.
Старые собственные материалы R&D описываем как старые; чужой код и официальный
SwiGLU не выдаём за авторский конкурсный вклад. Автоматический check не заменяет review жюри.
