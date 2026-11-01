# R1: первые проверки на Kaggle T4

Дата: 30 октября 2026. Код solution/ соответствует commit
[`00d00d6b678548851b4725c390bbd91152f411a3`](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/commit/00d00d6b678548851b4725c390bbd91152f411a3).
Development notebook: https://www.kaggle.com/code/koko667/setww-r1-ce-metrics-t4

Среда: Tesla T4, одна видимая GPU, torch 2.11.0+cu128, Triton 3.6.0;
зависимости из официального notebook. Код и unit tests загружены source-only архивом
с SHA-256 manifest, без compiler caches, моделей, данных и credentials.

## Unit tests

`python -m pytest solution -q --tb=short`: **30 passed, 14 warnings in 44.54s**.
Это включает starter tests, CUDA CE forward/backward, большой словарь,
strides, GradScaler scale, три реальных LoRA updates и tiny Qwen/TRL/rewind.
Предупреждения — deprecation в dependencies, не failures.

## Pilot: по одному запуску

```bash
python task/prepare_data.py
python harness/run_task.py --out runs_pilot
python harness/check_rules.py --result runs_pilot/result.json
```

| Показатель | Base | Solution |
|---|---:|---:|
| Full-command time, с | 85.92 | 89.49 |
| nvidia-smi peak, MiB | 14345 | 10005 |
| Eval loss | 2.04967 | 2.04965 |
| train_loop_s | 53.92 | 66.01 |

Speedup 0.9601; memory ratio 1.4338; eval delta −0.001%.
Own share 25.7% по официальному harness, шаги 3+.
Все 10 checks PASS; Soup regressions: 79 passed, 14 warnings in 0.50s.
First-step loss gap 0.00%, mean curve gap 0.01%.

Это память за счёт R1 вместе со starter SwiGLU, а не изолированный эффект R1.
Скорость не улучшилась; один pilot не доказывает устойчивую разницу времени.
## ABBA v1: отрицательный результат

Два base и два solution в порядке ABBA: **9/10 checks PASS**.
Check 8 FAIL: время x1.139 к базе, выше допустимых x1.10.
Память solution 10005 MiB в обоих повторах; own share min 25.3%.
Качество, кривая, метрики, rewind и 79 Soup tests прошли проверку.

Эта версия не допускается по результату ABBA, несмотря на успешный одиночный pilot.
Её исходники отдельно сохранены в notebook как `r1_v1_source.zip`.

## V2: runtime sequence и strides

Убрана специализация по меняющимся длинам batch, вызывавшая новые компиляции на шагах 3+.
Алгоритм CE/metrics/backward не менялся. Новые kernels снова проверены:
**30 passed, 14 warnings in 21.40s** на той же T4.
Kernel file SHA-256: `ecc328c127fcd9317c92b79fe95fb4e97274a287457a043019ed7356b7e10b48`.
Код v2: [`2e2a4b5fce8f1c047dcbf95fedbe27db452dbf95`](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/commit/2e2a4b5fce8f1c047dcbf95fedbe27db452dbf95).

| Показатель | Base | V2 |
|---|---:|---:|
| Full-command time, с | 78.05 | 75.44 |
| nvidia-smi peak, MiB | 14345 | 10005 |
| Eval loss | 2.04966 | 2.04967 |
| train_loop_s | 54.33 | 52.05 |

**9/10 checks PASS**. Единственный FAIL: own share **0.9%**, требуется ≥5%.
На шагах 3+: 411.872 мс CE kernels / 46.022 с loop = 0.89495%.
Speedup 1.0346, memory ratio 1.4338, official eval delta +0.000%.
First-step gap 0.00%, mean curve 0.01%; metrics/rewind и 79 Soup tests PASS.

V2 пока не допускается отдельно. Её ABBA не запускался, поскольку quota уже провалена.
Нужен больший полезный авторский охват вычислений; искусственные задержки,
busy work или возврат лишних компиляций не являются решением.

## Приложенные доказательства

[raw-evidence.zip](raw-evidence.zip), 248843 bytes, SHA-256:
`57b5b31c77ad2b12be047a8484766dd2a7177e46bafc9b953eba0f0ad1bb658e`.

В архиве: исходные result/checks всех трёх фаз, train/eval logs, trainer_state,
rewind, unit-test logs, source manifests, source v1 и source v2.
Weights, adapter tensors, данные и compiler caches не включены.

При загрузке проверены SHA-256 архива, `result_sha256` каждого checks.json и точное
совпадение v2 solution/ с локальным кодом. [summary.json](summary.json) — производная
сводка для чтения, а не замена raw JSON. Рядом опубликованы checks и test logs.

Notebook сохранён Quick Save как исследовательская версия; это не финальный публичный
Save & Run All по требованиям сдачи. Решение проверялось на открытых строках и seed задачи.
