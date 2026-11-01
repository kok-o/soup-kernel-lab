# Bulk activation transfer experiment

Decision: **not selected**. Original ABBA is 62.55 s versus 62.81 s control,
both 495 MiB /10 checks. The 0.26 s delta is not a convincing speedup. The
selected `solution/` has been restored byte-for-byte to `06dc4b1`; candidate
`77cfc4d` and its complete raw source remain available for reproduction.
[Evidence](../evidence/kaggle-bulk-h2d-20261101/README.md).
The following describes the experiment before measurement.

Control runtime: `06dc4b1957fbe1e23b4b32fa420d3205890fa553`, all 28 layers
offloaded, qualified at 65.33 s /495 MiB on Kaggle T4. Target remains <=46 s
full cold /<=500 MiB external peak with all ten original checks.

Fresh diagnostic profile: 2,973 H2D and 1,815 D2H activities in three steps.
Activation prefetch accounts for 1,677 H2D activities. The CUDA API trace also
contains 169 cudaHostAlloc calls (313 ms summed CPU duration). Heavy profiling
inflates host overhead, so these durations are not projected speed savings.

Hypothesis: reuse fixed pinned host slabs by layer and coalesce adjacent saved
views into a single byte-range H2D per slab. Keep the existing GPU staging,
target views, D2H ordering, events, native arithmetic and all output semantics.
No additional GPU allocation or model/activation compression is introduced.

Each forward hooks closure owns its slab allocation cursors. Slabs with any live
view cannot be leased to another closure. A layer exceeding the default 48 MiB
host capacity uses additional slabs. This increases pinned host RAM; the first
allocation cost stays inside the measured cold command.

Tests cover 4/8-bit byte parity, mutation detection, another-stream consumption,
one H2D copy for adjacent packets, independent live generations and overflow.
The full existing suite and untouched official 2+2 ABBA follow a matched
solution-only screen. Screens and profiled numbers are not qualified results.
The orchestration saves failure evidence and never changes official files.

No speed or memory result is claimed before Kaggle measurement.
