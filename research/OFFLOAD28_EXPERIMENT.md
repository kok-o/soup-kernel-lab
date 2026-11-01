# Slot-view cache with all 28 activation layers offloaded

This is a personal post-hackathon experiment, not a change to the historical submission.

Control: `60e973d310ad4b25ecf2eb3d18df284f7a563fcc`, whose runtime files exactly match historical `652334f3`: 20 activation layers offloaded, 8 resident, bounded slot-view metadata cache.

Candidate: the same runtime with the single change `patch_selective_offload(layers=28)` in `solution/__init__.py`. Native NF4 arithmetic, the two weight slots, head chunk 8192, codecs, data, RNG, LoRA and original official files stay fixed.

Hypothesis: offloading the final eight layers restores approximately 500 MiB external peak while retaining the final metadata-cache speed improvement. The time and peak are unknown until measured. Torch allocated memory does not replace external peak.

Protocol: one fresh Kaggle T4 session, same pinned dependencies, separate immutable checkouts. Run the complete patched test suite and original 2+2 ABBA for the control and candidate, followed by all 10 original checks for each. Every timed command starts in its own process and the official harness empties compiler caches. Save both original source-bound evidence archives. Eight full train commands, four per variant. Compare each variant to its own same-session base and report base drift; sequential suites do not prove a per-component causal speedup.

The comparison driver and notebook only orchestrate unmodified official commands. They are outside `solution/` and outside each measured process. No Brev machine is started.

Previously rejected ideas must not be presented as untested:

- Gate-to-up prefetch `c3a4a618`: Brev C/B/B/C 63.64/62.84/64.35/61.19 s at 495 MiB, candidate median slower by 1.18 s. No speed win.
- Broader predictor `47637690`: native parity and tests passed, single Brev 62.59 s /495 MiB did not establish a gain.
- Head chunk 16384 `b365144f`: original Kaggle ABBA 69.89 s /829 MiB, all 10 checks; slower than final 8192 stack.
- INT8 frozen-weight branch `5760de2`: only 8/10 checks; not a qualified reserve.

Use fresh transfer and phase profiles before revisiting these ideas. No prediction is a measured result.
