# Layer host slabs and bulk activation H2D

**Both variants qualify: 10/10 checks, 495 MiB external peak. No convincing speed
gain from this change is established.** Target <=46 s remains unmet.

| Same-session original 2+2 ABBA | Control | Bulk H2D |
|---|---:|---:|
| Commit | `06dc4b1` | `77cfc4d` |
| Full cold median | 62.81 s | 62.55 s |
| Runs | 62.51 /63.12 s | 62.52 /62.57 s |
| External GPU peak, both repeats | 495 MiB | 495 MiB |
| Training-loop median | 39.65 s | 39.40 s |
| Outside training | 23.16 s | 23.15 s |
| Stock Soup full cold median | 79.56 s | 79.52 s |
| Eval loss | 2.05210 | 2.05204 |
| Relative eval delta vs own stock base | +0.118% | +0.116% |
| Minimum counted own-kernel share | 8.0% | 8.1% |
| Patched suite | 370 passed, 2 skips | 373 passed, 2 skips |
| Original checks | 10/10 | 10/10 |

The observed median delta is 0.26 s (0.4%), less than the control's 0.61 s
run-to-run spread. Four stock Soup times range from 79.41 to 79.68 s. Do not
claim a reliable acceleration from two repeats or compare directly to 65.33 s
from another session. In particular the earlier single screens, 73.15 versus
63.02 s, substantially overstated the difference because the session drifted.

## Implementation and hypothesis

The [fresh profile](../kaggle-offload28-profile-20261101/README.md) identified
1,677 activation-prefetch H2D activities over three steps and repeated pinned
host allocation. The candidate uses reusable aligned host slabs per layer and
coalesces adjacent views into byte-range H2D copies. Existing GPU staging,
stream dependencies, codecs and native weight arithmetic are retained.
Host RAM use increases; the first slab allocations remain inside cold timing.

Tests confirm bit-exact 4/8-bit transfer, mutation detection, another-stream
consumption, one H2D for adjacent packets, simultaneous live generations and
small-slab overflow. No official files, data, allocator or outputs changed.

## Source, notebook and verification

- Control: `06dc4b1957fbe1e23b4b32fa420d3205890fa553`.
- Candidate: `77cfc4d7ef293b1d45e2dab06b5428d178f1faa1`.
- Runtime diff: `solution/activation_offload.py`, plus one new transfer test.
- [Saved free T4 notebook, version 364515851](https://www.kaggle.com/code/koko667/soup-lab-bulk-h2d?scriptVersionId=364515851)
  is private to its owner; source is in `kaggle/bulk-h2d/` and raw evidence here
  is public. The original hackathon notebook was not modified.
- Two sequential original 2+2 ABBA suites, eight full 64-step cold commands,
  unchanged 100 ms external meter, allocator and TritonWatch; no skip-tests.
- `VERIFIED.json` checks archive manifests, all 96/97 source files against the
  pinned commits, 15 official files, source/result/checker bindings, all ten
  checks, every run's 64 steps/rewind/T4/GPU0 and matching data hashes.
- Allocator diagnostics remain 275.1 MiB allocated /326 MiB reserved. These
  are not substituted for the external 495 MiB result.

Raw SHA256:

- Control: `59f10cdb80065ced53e18aaed63e356acd7d6b6e14cd74eb3630697d4e2d1aaf`.
- Candidate: `ec3002817083c3abc0bd7982e10dc3a90a2adf57b3c8b9f7dd046a779ddf678b`.

```bash
kaggle kernels push -p kaggle/bulk-h2d --accelerator NvidiaTeslaT4 --timeout 1800
python research/verify_offload_comparison.py \
  --downloads evidence/kaggle-bulk-h2d-20261101 --out verified-bulk.json
```

The current measured barrier is still about 39.4 s training plus 23.15 s other
work. The profile's summed host-allocation time did not translate into a large
full-command gain. Separate follow-ups test a shorter NF4 backward lease and
head chunk 4096; their results must be verified independently.
