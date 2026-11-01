# Time budget of the retained original baseline

Selected runtime: **`06dc4b1957fbe1e23b4b32fa420d3205890fa553`**. All96 default
source files match it byte-for-byte. No new performance candidate was launched
after the user asked to finish NF4 early release and head4096.

The fresh accepted control is **62.81 s full cold /495 MiB external,10/10**:
**39.65 s training +23.16 s outside training**. Official source-bound evidence
is in `../kaggle-bulk-h2d-20261101/control-raw-evidence.zip`. Bulk H2D, early
release and head4096 did not establish speed improvements and are not selected.

## Why one diagnostic repeat

The original trace has no CUDA sync records and1.37 million Python events. The
same three steps take2.082 s under phase observation versus4.223 s with heavy
profiling. Its2.678 s inactive GPU time is not ordinary Python overhead.

The repeat asks only: **late host submission or already queued CUDA waits?**
It uses the original baseline, disables shapes/stacks/profiler memory, enables
sync/event metadata and captures steps4-11. A separate64-step phase child gives
a matched reference. Both complete all64steps without runtime changes.

[Free T4 notebook](https://www.kaggle.com/code/koko667/soup-lab-light-profile),
version1, is private to the owner. Source: `kaggle/light-profile/`; observer/
driver: `bdec46ad36dae645151a1eea84f0f282ad04cf89`. All manifest entries and96
source files were verified against Git. Raw SHA256:
`0ece550f00812b82eae6f480ea1384c2a5d0c5cbb6fc3dcbc4e536797ca9da67`.

## Measured eight-step coverage

Rows partition first-to-last GPU activity. Concurrent DMA is assigned to the
compute row, so the coverage does not double count overlap.

| Exclusive diagnostic coverage | Seconds |
|---|---:|
| Compute active | 2.4988 |
| Copies active without compute | 1.3100 |
| Other GPU activity | 0.0006 |
| No GPU activity | 4.0868 |
| Total GPU window | 7.8962 |

The inactive portion splits by the first following activity's API:3.6824 s
before the call,0.3221 s during it,0.0822 s after its return. Before-call delay
includes required CPU work and profiler overhead; after-call delay can include
dependencies or scheduling. Neither is automatically removable work.

Sync/event records are now present. **Stream Wait Event record duration is not
the full consumer stall until readiness.** The14 ms overlap of recorded sync
activities with idle GPU does not establish that all actual device waits total
14ms. No such claim is used. Exact causal waits need stream/event reconstruction.

Matched reference steps take **5.3201 s**; profiled CPU step windows take
**7.9009 s**, excluding export. Even the lighter trace adds **2.5808 s** (~49%).
No further profiling run was launched. `TIME_BUDGET.json` preserves raw coverage,
host gaps, macro attribution and limits; `ANALYSIS.json` has detailed operations.

## Rough full-cycle estimate, not a measured64-step partition

Assume profiled hardware durations approximate the matched phase window and
that these eight variable-length steps are representative. Scale hardware time
by `39.65 /5.3201`; assign the remainder to host/API/unresolved waits. This
removes profiler inflation from the residual model, not from measured results.

| Approximate accepted-command portion | Seconds |
|---|---:|
| GPU computation | ~18.6 |
| Copies without compute | ~9.8 |
| Host/API/unresolved waits, residual | ~11.3 |
| Outside training, directly accepted | **23.16** |

The first three rows are **estimates**.11.3 s is not a direct measurement of
removable Python overhead. Shapes, clocks and profiler-induced overlap limit
precision. `MODEL_ESTIMATE.json` records formulas and assumptions; these are
order-of-seconds estimates for prioritization, not a statistical bound.

## Outside training

The accepted23.16 s includes **5.125 s median apply()** plus18.035 s other work.
A detailed phase child reports imports7.66 s, exclusive apply/compile prep5.42 s,
model load/quantize/pin4.66 s, data/fingerprints3.01 s, save0.50 s, trainer0.01 s
and other setup/completion3.96 s. They are diagnostic attribution, not exact
subparts of the accepted23.16 s: that child's outside-loop time is28.04 s.
JIT joins sum to1.10 ms; compilation is already hidden and offers no credible
13-second saving. Do not silently scale those startup spans into accepted time.

## Next direction and potential

The remaining gap is **16.81 s**. Even perfect removal of all estimated
uncovered copying gives only **~53.0 s**. Dequantize alone corresponds to~2.0 s;
head row statistics~0.15 s. Neither can bridge the gap.

Choose the **whole saved-activation storage/transfer lifecycle** as the next
large object, rather than another event-position tweak. In eight steps it has
0.767 s DMA without compute and1.137 s host-before-submit delay inside its
regions. The DMA coverage corresponds to~5.7 s in the model. Its host sum
includes unknown profiler overhead and cannot all be promised as savings.
Per-region DMA may overlap other DMA, so those upper budgets are not additive.

A concrete future design is bounded per-layer packing/transfer scheduling,
sharing existing GPU staging for forward D2H and backward H2D, with explicit
last-reader events. This targets many per-tensor D2H copies/events and overlap
with the next layer. Bulk H2D tested only host slabs and return coalescing; it
left per-tensor D2H on the compute stream. Simply retrying it is not justified.

**Priority estimate:3-6 s potential**, conditional on actually reducing host
work and hiding a substantial DMA share without raising memory. It is a
hypothesis, not a result. Success would still leave~11-14 s elsewhere. An old
unbounded D2H-overlap trial did not win and had an unexplained memory spike;
future work needs an explicit shared-arena/in-flight bound and a new proof.
**No implementation or new candidate run is started now.**

```bash
python research/analyze_time_budget.py \
  --raw evidence/kaggle-baseline-light-profile-20261101/stack-profile-raw.zip \
  --out time-budget.json
```

The raw archive includes original source, observer, phases and CUDA sync/event
data. No weights, adapters, credentials or compiler caches are bundled. The
timeline child's155.64 s /507MiB includes profiler/export work and is not a
qualified performance or memory result.
