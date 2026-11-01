# Release packed NF4 slots after the last actual reader

Runtime control: `77cfc4d7ef293b1d45e2dab06b5428d178f1faa1` (bulk H2D),
whose unit suite and preliminary Kaggle screen passed. Its full qualification
is still running; the screen alone is not qualification.

The fresh 28-layer trace attributes 4.052 GB H2D over three steps to NF4 weight
leases, with compute/copy overlap incomplete. The existing backward records
the slot's last-reader event after both decode and dense matmul. Source data
flow shows that only decode reads packed bytes; the decoded tensor owns
separate storage. This is a narrower dependency to test, without predicting
future weights or adding transfers, slots or GPU storage.

Change only the scope of the backward lease. Keep the exact existing
`dequantize_4bit(...).to(dy.dtype)` and `torch.matmul(dy, weight)`, their order,
stream, shapes and dtype. Forward remains native MatMul4Bit. The lease's event
is now recorded between decode and matmul, allowing another H2D into that
packed slot while the dense matrix is consumed. This can only help if that
event was on a critical path; no speed gain is assumed from the trace alone.

The new test verifies the release point and bitwise native forward/dX parity.
Existing slot reuse, mutation and scaled-gradient tests remain enabled.
Matched single-run screens precede full original 2+2 ABBA on each immutable
snapshot, all ten original checks and complete raw evidence. No paid GPUs.

This is not a retry of the rejected gate-to-up or broad predictor: it changes
an overlong ownership dependency, without preloading or duplicating a weight.
Do not promote it unless full cold speed, external peak and quality qualify.
