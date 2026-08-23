# AGENTS.md — Benchmarks (`polars-cv/benchmarks/`)

> Read the [root AGENTS.md](../../AGENTS.md) first for project-wide context.
> Update this file when you add new scenarios, frameworks, or change the benchmark harness.

## Purpose

This directory contains a comprehensive benchmarking suite for comparing polars-cv against other vision processing frameworks. The focus is on **batch preprocessing for inference** and **ETL workloads** — not training-time random augmentation.

## Scope

### What the benchmarks test:
- Batch image decoding and preprocessing
- Single operations (20 benchmarks: resize, grayscale, normalize, flip_horizontal, flip_vertical, crop_center, blur, threshold, rotate_90, rotate_45, invert, adjust_contrast, adjust_brightness, sharpen, pad, erode, dilate, histogram_equalize, canny, sobel_x). The authority is `get_single_op_benchmarks()` in `scenarios/single_ops.py`; `test_benchmark_list_is_current` pins this sentence to it.
- Multi-operation pipelines (light, medium, heavy)
- End-to-end file-to-memory workflows
- Zero-copy ingestion performance
- The remote (`file_path`) fetch stage, against a loopback HTTP server —
  `scenarios/remote_source.py`. Every other scenario is handed bytes that are
  already in memory, so without this the `fetch.rs` / `cloud.rs` path that every
  `s3://`, `gs://`, `az://` and `http://` source goes through is unmeasured.

### What they do NOT test:
- Random augmentation (not supported by polars-cv)
- Training data loading with per-epoch variation
- GPU-based augmentation pipelines
- Real S3/GCS/Azure endpoints. Those need credentials and a bucket, so they
  cannot be a committed benchmark. `remote_source.py` measures the structure
  they share — one client built and one GET issued per file, batched by
  `fetch::prefetch` — with the wide-area latency removed, and can inject a
  synthetic latency (`--latency-ms`) when the point is to model the link.

## Directory Structure

```
benchmarks/
├── __init__.py
├── conftest.py                     # BenchmarkConfig, pytest CLI options
├── run_benchmarks.py               # CLI entry point
├── inference_pipeline_comparison.py # Inference-focused comparisons
├── batch_throughput.py             # Batch decode/preprocess throughput benchmarks
├── plugin_overhead.py              # Per-call plugin/dispatch overhead measurement
├── frameworks/                     # Framework adapters
│   ├── base.py                     # AbstractFrameworkAdapter
│   ├── polars_cv_adapter.py        # polars-cv (eager + streaming)
│   ├── daft_adapter.py             # Daft (native-only + batch-UDF variants)
│   ├── opencv_adapter.py           # OpenCV adapter
│   ├── pillow_adapter.py           # PIL/Pillow adapter
│   └── torchvision_adapter.py      # torchvision (CPU + MPS)
├── scenarios/                      # Benchmark scenarios
│   ├── single_ops.py               # Individual operation benchmarks
│   ├── pipelines.py                # Multi-op pipeline benchmarks
│   ├── e2e_workflow.py             # End-to-end file-to-memory
│   ├── zero_copy_ingestion.py      # Zero-copy path benchmarks
│   └── remote_source.py            # Remote/cloud fetch path (loopback HTTP)
├── utils/                          # Shared utilities
│   ├── data_gen.py                 # Synthetic test data generation
│   ├── memory.py                   # Memory measurement
│   ├── results.py                  # Result formatting and comparison
│   └── validation.py               # Cross-framework result validation
├── regression/                     # Performance-regression harness (commit-to-commit)
│   ├── run_suite.py                # Run the regression suite
│   ├── compare.py                  # Compare runs / flag regressions
│   ├── config.py                   # Regression thresholds + config
│   └── README.md                   # Regression framework docs
└── reports/                        # Dated benchmark runs + analysis writeups
    ├── 2026-06-12-streaming-analysis/  # main vs OpenCV/Pillow/torchvision,
    │                                   # streaming-engine deep dive, raw JSON
    └── 2026-08-23-daft-comparison/     # Daft vs polars-cv: throughput, op
                                        # coverage, setup, flexibility; plus
                                        # capability_probe.py (what each engine
                                        # can express) and parallelism_probe.py
                                        # (core utilization / partition sweep)
```

## Frameworks Compared

| Framework | Description |
|-----------|-------------|
| `polars-cv-eager` | polars-cv with standard `.collect()` |
| `polars-cv-streaming` | polars-cv with `.collect(engine="streaming")` |
| `daft` | Daft using **only** its own image expressions |
| `daft-udf` | Daft with `@daft.func.batch` UDFs filling the gaps |
| `opencv` | NumPy + OpenCV (industry standard baseline) |
| `pillow` | PIL/Pillow |
| `torchvision-cpu` | torchvision on CPU |
| `torchvision-mps` | torchvision on Apple Metal GPU |

### Why Daft has two adapters

Daft's native vision surface covers three of the twenty single-op benchmarks
(resize, grayscale, crop) and none of the five pipelines, so a single adapter
would have to either report gaps everywhere or quietly substitute Python and
call the result "Daft".

`daft` does the first: any op with no native Daft expression raises
`NotImplementedError`, so a missing cell in the results table is a missing
capability rather than a failed run. That is the engine-vs-engine measurement.

`daft-udf` does what a Daft user actually has to do — native expressions where
they exist, a batch UDF where they do not. Its UDF bodies call `OpenCVAdapter`
rather than reimplementing blur/canny/sobel a second time, so `daft-udf` vs
`opencv` isolates Daft's UDF overhead over byte-identical kernels. **Keep it
that way**: a hand-written kernel inside a Daft UDF would be a second
implementation of an op this repo already has, and the ratio would stop meaning
anything.

`NATIVE_OPS` in `daft_adapter.py` is the list of ops Daft can express itself.
Widening it without a real native expression behind it is the one way these
benchmarks can lie.

## Running Benchmarks

```bash
cd polars-cv

# Install benchmark dependencies (bench is a [dependency-groups] group, NOT a
# [project.optional-dependencies] extra — `pip install ".[bench]"` installs
# nothing). Because bench is not a default group, run with `uv run --no-sync`
# below so uv does not re-sync and drop torch/torchvision.
uv sync --group bench

# Run all benchmarks (must be `-m benchmarks.run_benchmarks`, not the script
# path, or the `benchmarks` package will not import)
uv run --no-sync python -m benchmarks.run_benchmarks

# Run specific scenario
uv run --no-sync python -m benchmarks.run_benchmarks --scenario single_ops

# Run specific frameworks (comma-separated)
uv run --no-sync python -m benchmarks.run_benchmarks --frameworks polars-cv-eager,opencv

# With validation (compare outputs against OpenCV reference)
uv run --no-sync python -m benchmarks.run_benchmarks --validate

# Custom sizes and counts (comma-separated)
uv run --no-sync python -m benchmarks.run_benchmarks --counts 100,500 --sizes 256,512,1024
```

## Architecture

### Adapter Pattern

Each framework implements `BaseFrameworkAdapter` (in `frameworks/base.py`) which provides a consistent interface for:
- Single operations (20 benchmarks covering spatial, intensity, morphological, and edge detection operations)
- Pipeline execution (chained operations)
- End-to-end workflows (decode → process → encode)

New operations (rotate, erode, dilate, invert, contrast, brightness, sharpen, pad, histogram_equalize, canny, sobel) have default implementations that raise `NotImplementedError`. Adapters implement what their underlying framework supports; the benchmark runner catches errors for unsupported operations.

### BenchmarkConfig

Defined in `conftest.py`. Controls image counts, sizes, warmup/iterations, and output format. Can be configured via pytest CLI options (`--benchmark-counts`, `--benchmark-sizes`, etc.).

### Validation

`utils/validation.py` uses OpenCV as the reference implementation. Results are compared with configurable tolerance to verify correctness across frameworks.

## Adding a New Framework

1. Create `frameworks/my_framework_adapter.py`
2. Implement `AbstractFrameworkAdapter`
3. Register in `run_benchmarks.py`

## Adding a New Scenario

1. Create `scenarios/my_scenario.py`
2. Define benchmark functions that accept adapters and config
3. Register in `run_benchmarks.py`
4. To make it selectable from the regression suite, add its name to
   `regression/config.py`'s `ALL_SCENARIOS` **and** a branch in
   `regression/run_suite.py`'s `_run_once`. Return
   `benchmarks.frameworks.BenchmarkResult` — a scenario with its own result type
   must convert at that branch (see `zero_copy_ingestion.to_suite_results`).
   `_run_once` rejects anything else, because appending a foreign record used to
   fail several frames later in `_aggregate_best`, naming a missing field rather
   than the scenario that produced it.
