# Full benchmark run + regression check — 2026-07-23

**Question asked:** run the full performance benchmark, and investigate why it
felt far slower than past runs / whether a regression was introduced.

**Environment:** ephemeral shared cloud VM, 4-core Intel Xeon @ 2.80 GHz, Linux
6.18, Python 3.11, release build. Baseline for comparison is the
[`2026-06-12-streaming-analysis`](../2026-06-12-streaming-analysis/) run
(4-core Xeon @ 2.10 GHz) — comparable core count, slightly higher clock here.

**Config run:** `--scenario all --counts 10,100 --sizes 256,512` across all five
adapters (opencv, pillow, torchvision-cpu, polars-cv-eager, polars-cv-streaming),
covering single ops + pipelines + e2e. The default also includes `count=1000`,
which was dropped here — see below.

- `results.json` — raw results (536 rows).
- `tables.md` — throughput tables at count=100.

## Verdict: no polars-cv regression

Comparing polars-cv throughput to the June baseline at count=100 across all 27
operations and both image sizes (54 cells per framework):

| framework | geomean now/baseline | min | max |
|---|---|---|---|
| polars-cv-eager | **1.88×** | 0.78× | 7.74× |
| polars-cv-streaming | **1.53×** | 0.67× | 7.13× |

polars-cv is at parity or **faster** than the baseline on nearly every op — often
dramatically (sharpen ~7×, sobel ~5×, dilate/erode ~4× eager). Only 3 of 108
cells fall below 0.75× (grayscale-512 stream 0.67×, crop_center-512 stream 0.70×,
heavy_pipeline-512 stream 0.72×), all marginal and all within the run-to-run
variance this shared VM exhibits. There is no systematic slowdown.

**Control:** OpenCV (whose code is unchanged) benchmarked at ~0.4–1.2× of its own
baseline with high cell-to-cell variance (e.g. adjust_contrast OpenCV measured
0.96× in one pass and 0.39× in another). That variance — on unchanged code —
confirms the noise is environmental (noisy-neighbor on a shared VM), not a code
change in polars-cv.

**Transient-noise example:** an initial full run showed `flip_vertical`
collapsing to 0.11×/0.21×/0.33× across *all* frameworks including OpenCV. A clean
re-measure put it back to normal (eager 1.34×, stream 0.85×) — a momentary load
spike on that one cell, not a defect.

## Why the wall clock felt "far slower than in the past"

Not a regression — scope and environment:

1. **Scope.** The default `--scenario all` is a full cartesian grid: counts
   {10, 100, **1000**} × sizes {256, **512**} × 3 scenarios × (3 warmup + 10
   iters) = 13 passes per cell. The historical reports were run predominantly at
   **count=100** (the June README reports geomean "over the count=100 cells").
   The 1000-image cells are ~10× the work and 512px ~4× on top, so the default
   full run is far more total work than the count=100 runs remembered. In the
   aborted default run, ~50–60 min elapsed just to reach the count=1000 cells of
   the *first* of two sizes of the *first* of three scenarios.
2. **Environment.** Shared cloud VM; the single-threaded Python OpenCV/Pillow
   loops in particular show large run-to-run variance here.

## Reproduce

```bash
cd polars-cv
uv sync --group dev --group bench     # bench is a dependency-group, not a [bench] extra
uv run --no-sync python -m benchmarks.run_benchmarks \
    --scenario all --counts 10,100 --sizes 256,512 --output json > results.json
```

Note: run as a module (`python -m benchmarks.run_benchmarks`), not as a script
path, or the `benchmarks` package won't import. Use `--no-sync` on `uv run` so
the bench group (torch/torchvision) isn't dropped by an implicit re-sync.
