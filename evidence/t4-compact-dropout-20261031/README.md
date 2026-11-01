# Compact dropout: official T4 ABBA, all10 checks

Runtime35ea24c796a35a93e8beca5943675fc4fae70d16.
Canonical solution SHA2563172cf8b138a331262bb63a471a31f67e2936336bb78ca13fc5192ade6a46ce5.

| Official cold command | Stock Soup | Candidate |
|---|---:|---:|
| Median train wall s | 80.58 | 61.83 |
| Maximum NVIDIA MiB | 14345 | 2307 |
| Train loop s | 56.68 | 35.80 |
| Eval loss | 2.04965 | 2.05207 |

Two repeats, fresh compiler caches, one clean T4. Quality+0.118%, first loss+0.02%,
mean curve+0.16/0.17%. Own kernel minimum14.2% from steps3+, excluding all three
starter SwiGLU derivatives.302 tests pass; all10 official checks pass,
including79 Soup regressions. All15 official starter files remain unchanged.

The source zip matches Git byte for byte. All39 manifest entries and the
solution/result/check binding were independently verified. Archive SHA256:
346a3a56ce76453217c9b9431baa614e98d2f7acbd6f61b9c2ce55598a250827.
Result SHA256:59551a35b3405c822828a9084897a3ac7f87f073e88601c67065782091f75a0d.

In the same Kaggle draft session, the prior compact+grouped candidate measured
64.16s/2409MiB under its own ABBA. Stock medians were80.54s and80.58s.
This dropout change removes redundant autograd saves around padded index_copy_.
The identical native padded dropout runs inside one custom Function; only valid
mask bits survive. Its backward uses the existing exact mask-gradient kernel.
Values, input gradients and Philox state match the padded reference bitwise,
including FP16/FP32 and no-input-gradient cases. No floating scatter input is saved.

A full64-step predecessor census confirmed1728 exact-view duplicate packets,
with38.21MiB maximum live duplicated GPU packets (not NVIDIA peak savings).
They came from q/v LoRA dropout reconstruction; actual varlen Q/K/V/output
saved tensors remained4D and uncompressed. The code removes both unnecessary
scatter saves instead of introducing a general storage-alias cache.

Separate Brev diagnostics:47.81s/2307MiB at10 offloaded layers;
49.31s/2207MiB at14 layers, both302 tests and eval2.05208/2.05207.
These Brev times are not Kaggle qualification numbers. The14-layer variant
requires its own Kaggle checks before selection.

This interactive qualification is source-bound. Public saved RunAll remains
a separate release step. Public reserve-v5 a017d6f is preserved.
