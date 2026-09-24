# Typed-op migration — P0 baseline

Recorded before any typed-op change (`TYPED_OPS_PLAN.md`, phase P0), with the
Rust sources of commit `91d51de`. Later phases compare against it; the plan
gates P2 and the final PR on "no regression beyond noise".

Machine: 4-core cloud container (the web-session sandbox). Rust stable, polars
1.42.0.

## Execution throughput — `baseline.json`

Release build (`maturin develop --release`), regression harness defaults
(`pipelines` scenario, 300 × 256² images, 1 pinned thread, best of 3 suite
repeats). Compare a candidate with:

```bash
python -m benchmarks.regression.run_suite --out candidate.json
python -m benchmarks.regression.compare \
    benchmarks/reports/2026-09-24-typed-ops-baseline/baseline.json candidate.json
```

`compare.py` applies the harness's calibrated thresholds and exits non-zero on
a regression.

## Plan build — `python -m benchmarks.plan_build`

Microseconds per builder append (best of 7, 100 appends), release build:

| scenario | µs / append |
|---|---|
| chain (`scale`/`clamp`) | 49.8 |
| mixed (`resize` with an expression, `blur`, `crop`, `grayscale`) | 82.8 |
| lazy continuation (`.pipe()` chains) | 83.5 |

## Build cost

| measure | value |
|---|---|
| full `maturin develop --release` (cold `target/release`) | 819 s |
| incremental `maturin develop` (debug) after a one-file change in `polars-cv/src` | 15 s |
| release `_lib.abi3.so` size | 32,134,072 bytes |
