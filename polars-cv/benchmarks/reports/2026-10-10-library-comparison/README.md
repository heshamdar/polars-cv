# Library comparison: polars-cv vs OpenCV, Pillow, pyvips and torchvision

The first full run of the comparison suite with pyvips, the nine newer single
ops, the four new pipelines and the multi-branch `workflows` scenario. Open
`report.html` in a browser for the interactive report. It has a summary, the
per-case matrix, a dot plot per scenario and speedup by image size.

- **Code**: `58fa704` plus the change that added the above (the run's metadata
  says "+ local changes" because the run predates its commit).
- **Build**: `maturin develop --profile benchmark`.
- **Host**: 4-core Intel Xeon @ 2.80GHz container, Python 3.13.
- **Libraries**: polars 2.0.0, OpenCV 4.11.0, Pillow 12.0.0, pyvips 3.2.0
  (libvips 8.18.7), torch 2.9.1 (CPU), torchvision 0.24.1.
- **Runs**: two, merged by `python -m benchmarks.report`:
  - `run-256-512.json`: 256² and 512² × 200 images;
  - `run-1024.json`: 1024² × 50 images.

  Each case was timed as 2 warmup passes and the mean of 3 timed passes.

```bash
F=polars-cv-eager,polars-cv-streaming,opencv,pillow,pyvips,torchvision-cpu
python -m benchmarks.run_benchmarks --sizes 256,512 --counts 200 --warmup 2 \
    --iterations 3 --frameworks $F --save-json run-256-512.json
python -m benchmarks.run_benchmarks --sizes 1024 --counts 50 --warmup 2 \
    --iterations 3 --frameworks $F --save-json run-1024.json
python -m benchmarks.report run-256-512.json run-1024.json -o report.html
```

## Headline: geometric-mean speedup of polars-cv (eager)

The speedup is computed over every case both libraries ran. Above 1× means
polars-cv is faster.

| Workload | OpenCV | Pillow | pyvips | torchvision (CPU) | polars-cv streaming |
|---|---|---|---|---|---|
| 256² × 200 | 1.67× (36/45) | 6.6× (36/36) | 20× (42/42) | 4.8× (23/24) | 1.08× |
| 512² × 200 | 1.61× (33/45) | 8.5× (36/36) | 11× (42/42) | 5.2× (22/24) | 1.03× |
| 1024² × 50 | 1.12× (24/45) | 8.4× (36/36) | 5.3× (39/42) | 5.0× (22/24) | 1.04× |

Each bracket gives the cases polars-cv won out of the cases both libraries
ran.

## What the run shows

- **Multi-branch workflows are where polars-cv leads most.** Against OpenCV
  it is 3.6–5.0× faster on `multi_output_etl` (one decode feeding a tensor, a
  thumbnail and a mean), 5.3–6.7× on `masked_stats` and 5.4–6.2× on
  `mask_to_contours`, at every size. These graphs share nodes and run across
  rows in parallel. The libraries run them as a Python loop over images.
- **The single-op lead over OpenCV shrinks as images grow, and reverses at
  1024².** At 1024² × 50, OpenCV is faster on 17 of the 29 single ops. The
  biggest gaps are `to_hsv` (polars-cv at 0.10×), `canny` (0.18×), `invert`
  (0.25×), `sobel_x` (0.26×), `convolve2d_5x5` (0.28×), `histogram_equalize`
  (0.36×) and `warp_affine` (0.39×). Every single-op case in polars-cv also
  reads a VIEW blob column and writes one back, a copy in and a copy out that
  OpenCV's in-memory arrays do not pay. At large images on 4 cores that
  bandwidth is a large part of a cheap op's cost. `to_hsv` and `canny` stay
  slower at every size, so they are kernel targets, not overhead.
- **pyvips is slowest here, and that is libvips's design rather than the
  adapter.** A trivial 128² crop costs libvips 0.45–0.7 ms per image at any
  concurrency setting, against 0.17 ms for OpenCV's whole-image invert.
  libvips sets up a pipeline per evaluation and is built to stream very large
  images through small tiles. A per-image loop over small images is its least
  favourable case, and its gap narrows from 20× to 5.3× as images grow from
  256² to 1024².
- **Streaming and eager are within a few percent overall.** Streaming loses
  most on `adjust_brightness` (0.62× of eager at 512²) and `normalize`
  (0.82×).

## Caveats

- One run on a shared 4-core container. Same-binary repeats on this host
  have moved cases by double-digit percentages before (see
  `2026-10-08-row-split-budget`), so read single cells with that error bar.
  The geometric means over 24–45 cases are steadier.
- OpenCV, Pillow and pyvips run as a Python loop over images. That is the
  common way to write them, but it leaves cross-image parallelism on the
  table, which polars-cv gets from Polars.
- Parity: `tests/test_benchmark_adapters.py` and
  `tests/test_benchmark_workflows.py` hold every library to polars-cv's output
  within a stated tolerance. An op with no equivalent call is not run (n/a)
  rather than approximated.
