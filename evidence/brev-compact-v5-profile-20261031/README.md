# Compact native decoder, Brev diagnostic

Not official Kaggle qualification. Runtime 85fd6faee9a44990b6639098dc2fc2c5cd805a70,
with test-only update 83f5e8a. Both source archives match Git byte for byte.
The downloaded archive SHA256 and all 38 manifest members were independently verified.

| One cold screen on the same Brev T4 | Train wall s | Loop s | NVIDIA MiB | Eval loss |
|---|---:|---:|---:|---:|
| reserve-v5 a017d6f | 51.67 | 36.45 | 2447 | 2.05208 |
| compact decoder | 51.21 | 35.68 | 2409 | 2.05210 |

This is a single comparison; no established sub-second speedup is claimed.
The unchanged official train/task/eval path was used, with fresh compiler caches.
Timing includes launch/import/JIT/model/data/train/save and excludes separate eval.

All 28 decoder layers used compact real-token outputs, verified by the profile.
Original dense native dropout masks/RNG, original RoPE positions, four causal
sequence boundaries, native q/k/v/o modules, norms, residuals and MLP math remain.
Only layout and GEMM M change. Root/head still receive the original dense shape.

The first strict synthetic comparison had 6 passes / 2 failures; the raw failure
is retained. Stage diagnostics found exact input norms and dropout/RNG, followed
by changed FP16 GEMM rounding. Two-layer hidden relative L2 was 0.126%, gradients
with the lossy codec differed 1.18-2.16%. With only codec storage made lossless,
gradient relative L2 was 0.119-0.229%. Adam's first update is sensitive to near-zero
gradient sign, so update-vector equality is not an appropriate geometry test.

The final unit test isolates layout/autograd with lossless saved activations:
hidden L2 <=0.2%, gradients <=0.5%, same loss/RNG, finite nonzero actual LoRA
updates, embedding gradient on/off. Native dropout values/dX/RNG are bit exact;
varlen no-gather vs gather is separately tested at scales 1 and 65536.
8 targeted and 283 full tests passed. Actual compressed training accuracy still
requires the official 10 checks before adoption.

Separate detailed profiles cover 16 active steps. Total GPU kernel time was
6.3449 s for v5 and 5.4408 s compact. These profiled totals are not full train wall
results. Pack calls increased 3104 ->3520 and unpack 2208 ->2624; CPU codec ranges
also grew, explaining why removing GPU padding does not give the same wall gain.
Head forward/dHidden is much smaller than the NF4 projection work in this profile.
Raw CPU shapes/calls, GPU tables, lifecycle, train logs, quality and test evidence
are in raw-evidence.zip. Range totals overlap and must not be added together.
