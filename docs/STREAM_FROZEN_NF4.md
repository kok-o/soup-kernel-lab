# Streaming frozen NF4: working primitive and first full screen

Started2026-10-31 17:12:04UTC. The requested15-minute initial code/proof window
ends17:27:04UTC. GPU assigned lab3 only; stop new tests17:50UTC, archive locally
by17:55UTC, VM cutoff18:00UTC. No Kaggle push. The final competition deadline
is18:59UTC. These are independent research artifacts; reserves are unchanged.

## Implemented

`solution/stream_nf4.py` stores immutable pinned CPU packed weights plus the
complete nested QuantState tree. Exactly two GPU slots copy packed bytes and
metadata. Copy-stream ready events precede compute; last-reader events precede
slot overwrite. Slot views are never kept in the long-lived autograd context.
Forward delegates the installed native bnb MatMul4Bit operation; backward
reacquires the host weight and repeats original dequantize+matmul. A mutation
version check protects the host source. This is real weight streaming, separate
from activation offload.

`solution/stream_loader.py` scopes HF get_device interception to the original
weight conversion call. Frozen projection tensors materialize on CPU; one
matrix at a time goes through original CUDA NF4 quantization and immediately
returns to pinned host RAM. Its original GPU Params4bit/view is discarded.
The integrated loader also routes embedding/head materialization to CPU.
Selective `_apply` handling protects host frozen weights from HF/Trainer device
movement; PEFT trainable adapters move to CUDA explicitly. LoRA initialization
still occurs before the installed PEFT layer moves adapters to the base device.

Verified HostHead file is copied exactly from runner1 source
`a25a65d8df02e1c7889d817faded362122266bf6`. This replaces the old AG head with
incorrect supervised-only entropy. Only its file/calls are used: no custom
triton_key, allocator tuning, fused optimizer, GC or other AG changes are copied.
The underlying runtime base is a7; official harness/task/tests are unchanged.

## Evidence obtained

1. Source `b70ea122dd4a1cd88e0038a49826082c02eb5d2f`: three actual layer0 NF4
   gate/up/down weights, all forward/backward byte-exact at scales1/1024/65536,
   contexts outlive slot reuse, three LoRA/dropout/AdamW updates and RNG exact,
   host mutation detected. Two slots occupy14,201,544bytes. Diagnostic external
   peak279MiB;64.24s includes proof/import/compilation. This is not fulltrain time.
   [Layer raw/evidence](../evidence/stream-layer-20261031).
2. Source `c7bada3`: initial whole-model short synthetic FW/BW/update succeeds.
   All196projection weights remain on CPU after load/PEFT/backward. That early
   diagnostic retained/upcast the GPU embedding:1761MiB external peak. It is
   preserved, not claimed as a memory improvement.
3. Source `6fb075e`: integrated CPU-first weights plus corrected HostHead passes
   original64step solution-only cold screen on lab3: **60.19s /1207MiB**, loop
   **41.53s**. This has no base/eval/check_rules admission. It misses both the
   requested400MiB and<=53s goals. [Raw screen](../evidence/stream-full-initial-20261031).
4. Source `afc358a`: one bounded follow-up uses activation offload28 instead of
   a7's12, with a CPU-NF4 parameter/allocation census at Trainer init and
   steps1/2/64. Running; no result assigned in this report yet.

## Budget and next gate

Exact shape calculations are in `research/STREAM_MEMORY_BUDGET.json`: roughly
644.6MiB host NF4 weights/statistics,445.125MiB host embedding, maximum26.25MiB
temporary dense quantization/dequantization matrix,13.54MiB two packed slots,
33.25MiB LoRA params/grads/two Adam states, and48MiB HostHead weight buffers.
CUDA context/libraries/workspace, saved activations, head temporaries and
allocator reservation are additional and not assumed zero. These components
are not necessarily all simultaneously live.400MiB external peak is unproven.

The concrete next test is the all28offload census screen already launched.
If it establishes materially lower peak, compare complete task/output settings,
then original tests/ABBA/check_rules without skips and separate evaluation on
the exact resulting source. No partial/regression result becomes a reserve.
Do not transfer the279MiB layer diagnostic peak to a fulltrain claim. Further
changes need evidence: scalar census must identify whether startup placement,
saved activations or HostHead temporaries dominate. No tile or codec sweep.

The existing two-slot primitive enforces lifetime safety but does not prefetch a
future layer's weight early. Repeated FW/BW H2D traffic and event/metadata cost
can reduce speed; the first full screen confirms no<=53s gain yet. Any later
overlap must retain the same ready/read-done ordering and native precision.
