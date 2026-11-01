# T4 research update · reserve-v5

Selected public ABBA: **88.65 → 64.57 s**,
**14345 → 2447 MiB**, eval
+0.116%, 10/10, 275 tests, own min 18.6%.
Source `a017d6fea308e7c250323ad70f44af34ef9e3d93` / reserve-v5.
[Public evidence](../evidence/reserve-v5-public-20261031/README.md). Independent research ABBA: 81.97 → 65.08 s,
2447 MiB, eval +0.119%, 10/10.
[Research binding](../evidence/t4-fast-reserve-v5-20261031/README.md).

The selected combination adds compressed offload reads in SwiGLU backward, native varlen
attention, retained native GQA outputs, fresh-buffer MLP scatter, early FP16 tied storage
and complete cold launcher preparation. The real preparation peak falls by 889.97 MiB;
unprofiled first step falls from 4.319 to 1.007 s. Those diagnostic observations have
their own raw records and are not substituted for full-command ABBA measurements.

[Nine isolated screens and negative results](../evidence/speed-experiments-20261031/README.md)
retain exact source/result/archive hashes. Forced bnb SM75 fused GEMM loses on measured
large-M shapes; all-INT4 and direct fallback did not establish a useful speed gain.
The varlen model test caught and fixed FP32-Q gradient scatter typing before qualification.
No public result or earlier reserve has been overwritten.

## Earlier record, preserved

# T4 research update · reserve-v4

Selected memory reserve: public ABBA **87.23 → 69.42 s, 14345 → 2463 MiB**,
eval +0.128%, 10/10, 261 T4 tests. Source `9e6a312`, tag reserve-v4.
[Public evidence](../evidence/reserve-v4-public-20261030/README.md).
Independent ABBA of that exact source: 80.56 → 68.94 s, 2463 MiB, +0.120%, 10/10.
[Research evidence](../evidence/t4-2500-qualified-20261030/README.md).

## New source-bound experiments

| Change | Full command, s | Peak, MiB | Status |
|---|---:|---:|---|
| LoRA scale/add fusion | 69.11 | 3429 | ABBA, 10/10 |
| LoRA + shared KV | 69.46 | 3061 | ABBA, 10/10 |
| + RoPE/cast, reserve-v3 | 68.84 | 3041 | research ABBA, 10/10 |
| Same v3 public Run All | 67.86 | 3041 | public ABBA, 10/10 |
| + native dropout mask bits | 67.89 | 2879 | screen, 239 tests |
| + resident packed SwiGLU | 66.99 | 2799 | screen, 246 tests |
| + PR14 NF4 backward | 69.15 | 2799 | screen, 255 tests |
| + ordered offload first 12 | 70.96 | 2419 | screen, 250 tests |
| + pinned pool/two GPU slots | 71.35 | 2461 | screen, 254 tests |
| First 8 ordered, head256, fork sum | 70.74 | 2469 | screen, 261 tests |
| First 10 ordered, head512, one JIT worker | 67.49 | 2463 | screen, 261 tests |

All rows retain raw logs, source archives, commit/digest/result bindings under evidence/t4-*.
Single screens are not ABBA and do not isolate a causal gain across sessions.
The PR14 implementation is original @Nari-Ab work, retained as an optional separate branch;
its current no-offload screen did not justify enabling it in the selected reserve.
The two-slot prefetch passed parity tests but consumed more memory without a useful speed gain.
The packed SwiGLU rewrite is explicitly starter-derived and excluded from author timing.

## Earlier record, preserved

# T4: память и скорость

Публичный Run All того же source прошёл 10/10:
[80.06 → 69.95 с, 14345 → 3469 MiB, eval +0.118%](../evidence/reserve-v2-public-20261030/README.md).
[Свежий профиль и фактический padding](../evidence/reserve-v2-profile-20261030/README.md):
42.4207% padding по 64 батчам, codec 11.698% GPU kernel time, attention 9.669%.
Профилированное время не используется для scoring.

Квалифицированный source [`bef6d7a99779593bcd80d43df013ce34561d55b9`](https://github.com/BAITC-Hacks/hack-3ef0581f-setww/tree/bef6d7a99779593bcd80d43df013ce34561d55b9/solution).

| Официальный ABBA | Base | Solution |
|---|---:|---:|
| Медиана полной команды, с | 79.94 | 68.09 |
| Максимальный nvidia-smi peak, MiB | 14345 | 3469 |
| Public eval loss | 2.04966 | 2.05219 |
| Минимальная авторская доля, шаги 3+ | — | 17.8593% |
| Official checks | — | **10/10 PASS** |

Первый loss gap 0.02%, mean curve gap 0.17%; допуски 1% и 3%.
Eval +0.123%, 211 T4 tests, все 10 official checks. Source/result binding проверен.

## Screening ledger

Ниже по одному solution run: это не ABBA-медианы и не изолированные причинные эффекты.
Все compiler caches внутри measured commands были пустыми. Profile run не включён во время.

| Кандидат | Полная команда, с | Пик, MiB | Eval loss |
|---|---:|---:|---:|
| `cast256` | 84.15 | 4183 | 2.04971 |
| `cast512` | 77.94 | 4203 | 2.04972 |
| `castnorm256` | 82.34 | 4183 | 2.04970 |
| `forward_unpad512_screen` | 70.31 | 3939 | 2.04974 |
| `fp16_hybrid64` | 65.05 | 3469 | 2.05205 |
| `fp16_offload64` | 71.67 | 2507 | 2.05219 |
| `fp16_offload_attention` | 75.69 | 2223 | 2.05209 |
| `hybrid64_attention` | 77.24 | 2669 | 2.05192 |
| `hybrid64_offload` | 74.53 | 2607 | 2.05195 |
| `hybrid64_owned` | 66.91 | 3489 | 2.05212 |
| `int4_forward_unpad512_screen` | 67.77 | 3309 | 2.05636 |
| `pipelined_offload_screen` | 69.17 | 3123 | 2.05217 |
| `runtime_compile_screen` | 64.78 | 3469 | 2.05210 |
| `unpad256_screen` | 71.82 | 3819 | 2.04971 |

[Машинный ledger](../evidence/final-t4-20261030/EXPERIMENT_LEDGER.json).
Каждый прототип имеет `runs_*/solution-source.zip` в raw archive.

## Выбор и ограничения

- Чанк 256 почти не снизил peak и оказался медленнее 512 в первом screen.
- Unpadding MLP улучшил оба ресурса; LoRA/RNG parity проверена.
- Глобальный INT4/256 дал eval 2.05636; выбран INT4/64 только для MLP.
- FP16 MLP использует исходный starter SwiGLU. Все его произведения внутри kernels
  считаются FP32; fork casts сохраняют FP32 accumulation градиентов.
- Owned CE расходует временный logits buffer один раз. Generic CE не мутирует входы.
- Attention recompute прошёл LoRA/RNG tests и полный screen, но overhead выше прогноза.
  Exact checkpoint inputs и изоляция hooks позволяют использовать non-reentrant checkpoint.
- Offload экономит память с затратой времени. Pipelined вариант использует events,
  record_stream, prefetch и оставляет последние четыре слоя на GPU.
- CPU compile overlap внутри measured PID дал 64.78 против 65.05 с: слишком мало
  для вывода на фоне шума. Оставлен выключенным, prebuilt caches не используются.

Источник формул/API: [Qwen2](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py),
[bnb](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py),
[Triton](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/jit.py).
История Soup R&D и R2: [RESEARCH](RESEARCH.md), [R2_CHUNKED_HEAD](R2_CHUNKED_HEAD.md).
Готовые kernels не присвоены; sample SwiGLU исключён из author quota.
