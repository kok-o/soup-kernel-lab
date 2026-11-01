# Diagnostic experiments

These probes are outside the shipped solution and the supported regression suite.
Run `python -m pytest solution tests -q` for the qualified code.

- `rejected_head_gemm/`: numerically tested dense GEMM; rejected for T4 FMA performance.
- `nested_scale_cache/`: exact scale-cache experiment; slower solution-only screen, inactive.
- `staged_cast/probe_dtype_cast_legacy.py`: retained negative NF4 width-257 reference case; stock bitsandbytes can raise a CUDA misaligned-address error. Run in a separate process.
- `probe_cast_aligned.py`: real aligned Qwen shapes, 3 passed on T4; not activated in the qualified solution.
- `diagnose_training.py`: two-step isolation only, never scored benchmark evidence.

Original source snapshots and logs are in evidence/impact-t4-20261030/raw-evidence.zip.
Restore the source snapshot for an old experiment before reproducing it; do not combine results from different variants.
