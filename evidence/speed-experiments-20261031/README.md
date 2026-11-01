# T4 speed experiments, 30–31 October

These are independent, source-bound single-run screens, not ABBA qualification.
Each folder contains its unedited raw evidence ZIP and a verified summary. All archive
manifest entries, result hashes and solution bytes were checked against their Git commits.
Full-command time includes startup/JIT/model loading/training/saving; eval is separate.
Differences between these single screens do not establish a causal speedup.

| Screen | Full command, s | Peak Nvidia, MiB | Eval loss | Source prefix |
|---|---:|---:|---:|---|
| [No offload speed base](speed-base/verified-summary.json) | 66.09 | 2799 | 2.05218 | cb84f59 |
| [Packed offload reader](packed-offload/verified-summary.json) | 66.83 | 2463 | 2.05207 | 4ccf655 |
| [All-INT4 storage policy](speed-all4/verified-summary.json) | 66.74 | 2643 | 2.05312 | bc34f6b |
| [Retain native GQA outputs](gqa-saved/verified-summary.json) | 65.54 | 2827 | 2.05209 | 39f3d1d |
| [In-place MLP scatter](gqa-scatter/verified-summary.json) | 66.44 | 2827 | 2.05214 | 078b357 |
| [Direct installed bnb fallback](direct-bnb/verified-summary.json) | 66.44 | 2827 | 2.05212 | 8cb3876 |
| [Native varlen attention](varlen/verified-summary.json) | 66.09 | 2751 | 2.05207 | 107b8df |
| [Retain FP16 before PEFT prepare](early-half/verified-summary.json) | 66.50 | 2827 | 2.05210 | e5f83f9 |
| [Prepare launchers during imports/loading](launcher-preload/verified-summary.json) | 63.39 | 2827 | 2.05203 | 74685e7 |

All-INT4 above changes the policy only; it does not port the grouped scheduling from PR15.
The direct bnb path and all-INT4 did not improve the full-command screen enough to select.
The packed offload reader keeps gate/up compressed during H2D and decodes directly in
the excluded starter-derived SwiGLU backward. No author quota is claimed for that rewrite.

## Startup memory: real task preparation

The diagnostic invokes the unchanged task, observes the actual PEFT call and exits before
trainer construction. Native and retained-storage variants run in separate subprocesses.
This experiment resets Torch peak tracking at preparation entry and is explicitly separate
from official scored training. It does not report Nvidia training peaks.

| Torch allocated, MiB | Native PEFT upcast | Retained FP16 |
|---|---:|---:|
| Before preparation | 1090.429 | 1090.429 |
| Peak during preparation | 1980.679 | 1090.708 |
| After preparation | 1535.830 | 1090.705 |

The temporary preparation peak decreases by **889.971 MiB**. The real tied matrix has
233373696 elements (445.125 MiB FP16 / 890.25 MiB FP32). The native PEFT call still
freezes all parameters, upcasts the other eligible parameters and installs its normal
input-gradient/checkpointing behavior. The retained variant preserves the original matrix
pointer/tie and installs the existing FP32 embedding-output hook before PEFT's grad hook.
Other parameter dtypes match: FP32 and packed uint8. Full training peak remains 2827 MiB
in this independent screen; the preparation reduction is not a training-peak reduction.

Raw diagnostic records: [native](preparation-native.json),
[retained storage](preparation-retained-half.json).
Tests compare exact parameter values, embedding dtype/values, storage identity,
input-gradient boundaries and head/input gradients, with checkpointing on and off.

Reference: [PEFT v0.20.0 preparation](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py).

## Cold launchers

An unprofiled 64-step lifecycle trace separates startup, loading, the first steps and saving.
The previous worker had finished before forward (join approximately 0.5 ms), yet eleven
first JIT launches took a total of 3.214 s. Its first step took 4.319 s.

`warmup()` compiles the specialization but leaves the host launcher and CUDA module lazy
in Triton 3.6. The new worker starts before the remaining apply imports and reads each
compiled kernel's `run` property, initializing those handles without executing a GPU kernel.
It joins before forward. No cache is shipped or shared between timed commands; no external
background process is introduced. The measured full-command scope includes all this work.

The independent trace has **no JIT launch over 30 ms** and a 1.007 s first step.
This diagnostic explains the screen's roughly three-second difference; it is not an ABBA
estimate. [Before](lifecycle-before.json), [after](lifecycle-preloaded.json),
[computed comparison](diagnostic-summary.json).

Reference: [Triton 3.6 CompiledKernel](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/compiler/compiler.py).

## Varlen correctness

The installed T4 Torch 2.11 wheel accepts native CUTLASS `cu_seqlens` forward/backward.
Two six-head groups share the two unique KV heads. Dense padded LoRA dropout and RoPE
run first; only then are valid Q/K/V rows packed. FP16 per-head KV gradients widen
before the same FP32 six-head sum. No third-party attention implementation is installed.

Primitive tests cover FP16/FP32 boundaries, irregular sequence lengths, gradients at
scales 1 and 65536, zero padded outputs and unchanged CUDA RNG. The combined source
adds a real two-layer Qwen/LoRA test proving the registered path is entered after dense
dropout, comparing loss/trainable gradients and CUDA RNG against dense masked GQA.
That test caught an FP32-Q gradient scatter dtype mismatch; the combined source explicitly
restores the input dtype before scatter. Source `107b8df` is the earlier isolated screen;
the combined candidate uses the corrected implementation.

Reference: [PyTorch 2.11 native API](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml).
