# Fresh profile of the qualified 28-layer offload stack

**Diagnostic evidence, not a new speed or memory qualification.** The original
unprofiled result remains 65.33 s full cold /495 MiB, all ten checks, at
`06dc4b1957fbe1e23b4b32fa420d3205890fa553`.

The [free Kaggle T4 diagnostic notebook](https://www.kaggle.com/code/koko667/soup-lab-profile-offload28)
ran three separate complete 64-step children: phase observation, CPU/CUDA
timeline and allocation history. All exited successfully. The source notebook
is in `kaggle/profile-offload28/`; its observer/driver is pinned at
`49f333512ab4d49abe24023c8230e73d1f41de59`. All 96 runtime source files match
the qualified commit and all 15 official files match the imported baseline.
No official source was edited or measured work moved outside the command.

## Startup, training and completion

The phase observer uses Python 3.13 monitoring events for the actual function
code objects and delegates real import loaders. It does not replace dataset
functions or fingerprints. The following is an additive main-thread partition;
inclusive nested spans must not be added to it again.

| Main-thread interval | Diagnostic seconds |
|---|---:|
| Imports | 7.151 |
| Apply and compilation preparation, excluding nested imports | 5.147 |
| Model loading, quantization and pinning | 4.497 |
| Data preparation and real fingerprints | 2.911 |
| Trainer construction | 0.008 |
| Full training interval | 41.811 |
| Saving | 0.463 |
| Other setup/completion | 3.904 |
| Observed child partition | 65.892 |

The external diagnostic command takes 68.44 s. About 2.53 s elapse from the
child's atexit mark to its parent return. The parent time includes real process
finalization and sampler/polling effects; it is not all removable Python work.
Inclusive `apply()` is 7.06 s under observation versus about 5.18 s in the
qualified measurement. Treat phase times as attribution, not accepted speed.

There are **26** compile jobs, 13.566 s summed worker duration, overlapping
model/data setup. Their 64 join calls sum to only **0.00171 s**. Compilation is
already substantially hidden; the worker sum is not a 13.6 s startup saving.
Model pinning is 1.089 s within loading. Real data fingerprint spans take
2.242 s inside data work. Fingerprint substitution remains prohibited.

## CPU/CUDA timeline and dependencies

Three active steps after two initial steps and one profiler warmup contain:

| GPU interval union | Diagnostic ms |
|---|---:|
| Covered GPU activity span | 4208.5 |
| Any compute/copy activity | 1530.8 |
| Compute | 988.2 |
| H2D | 613.5 |
| H2D overlapping compute | 220.0 |
| D2H | 148.9 |
| D2H overlapping compute | 0.0 |

The trace records 2,973 pinned H2D copies (7.376 GB) and 1,815 pinned D2H
copies (1.923 GB). Activation prefetch contributes 1,677 H2D activities;
weight leases transfer 4.052 GB and the vocabulary head 1.400 GB H2D. Native
cuBLAS kernels already use Turing Tensor Cores. NF4 dequantize sums to about
96.9 ms and head row statistics to 8.1 ms in these three observed steps.

CUDA API observations include 169 `cudaHostAlloc` calls (313 ms summed CPU),
4,809 `cudaMemcpyAsync` calls (54.4 ms), 11,376 kernel launches (139.1 ms),
9,453 event records (31.2 ms) and 3,855 stream event waits (8.1 ms). These are
observations under the profiler; not every API call or wait is removable.

**Do not interpret the 2,677.7 ms with no GPU activity as recoverable time.**
The same steps 4-6 take 0.921/0.599/0.562 s under phase observation; with heavy
shape/stack profiling they take 1.607/1.281/67.268 s, with the last including a
56.18 s export. CPU profiling changes launch timing and can hold references.
The raw timeline and analyzer retain copy/compute unions, overlaps, correlated
launches and >50 us gaps. They establish where to test a change, not its speedup.

The next narrow experiment groups adjacent saved host views by layer and
coalesces their H2D copies into existing GPU staging. It targets call count and
repeated pinned allocation. Its performance and memory must be measured cold.
It does not retry the rejected gate-to-up predictor or D2H-overlap schedule.

## Memory peak

The allocator peak is first observed by the end-of-step-40 mark:
**275.130 MiB max allocated /326 MiB max reserved**. Backward reconstruction
from the history ring reproduces the allocated peak, retaining persistent
allocations that predate the ring. Largest groups live at that peak:

| Allocation origin | Live MiB |
|---|---:|
| Activation prefetch staging, two slots | 65.375 |
| Host-head GPU weight buffers, two slots | 48.000 |
| Streamed NF4 backward | 38.556 |
| Library/unknown stack attribution | 32.998 |
| Packed SwiGLU gate gradient | 24.387 |
| Packed SwiGLU up gradient | 24.387 |
| NF4 GPU weight slots | 13.558 |

Smaller inputs/norm/linear allocations fill the balance. The external meter is
495 MiB, including context/library allocations absent from the PyTorch allocator
history. **275 MiB allocated is not the 495 MiB headline.** Only 5 MiB external
margin remains at the requested threshold.

Allocation-history mode takes 417 s because it serializes large snapshots at
new peaks. Timeline mode takes 143.99 s /507 MiB. Both include diagnostic work
and are explicitly excluded from accepted performance and memory claims.

## Reproduce and inspect

`stack-profile-raw.zip` is a 42.5 MiB compressed archive of the untouched
runtime snapshot, observer, raw Chrome timeline, allocation histories, phase
spans, logs and metrics. It contains no weights, adapters or compiler caches.
`MANIFEST.json` binds every entry; `VERIFIED.json` checks source provenance.
`ANALYSIS.json` preserves the full machine-readable interval/peak analysis.

```bash
python research/analyze_stack_profile.py \
  --raw evidence/kaggle-offload28-profile-20261101/stack-profile-raw.zip \
  --out analyzed-profile.json
```

The analyzer verifies all manifest entries before analysis. Raw SHA256:
`f7e1b94502e471f7dafe3f34f8ed1f7613dba7d19177275f871ff29b9530a6cc`.
Sources: [PyTorch profiler overhead and reference retention](https://docs.pytorch.org/docs/2.11/profiler.html),
[allocator history coverage and ring limits](https://docs.pytorch.org/docs/2.11/torch_cuda_memory.html).
