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

### What they do NOT test:
- Random augmentation (not supported by polars-cv)
- Training data loading with per-epoch variation
- GPU-based augmentation pipelines

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
│   ├── opencv_adapter.py           # OpenCV adapter
│   ├── pillow_adapter.py           # PIL/Pillow adapter
│   └── torchvision_adapter.py      # torchvision (CPU + MPS)
├── scenarios/                      # Benchmark scenarios
│   ├── single_ops.py               # Individual operation benchmarks
│   ├── pipelines.py                # Multi-op pipeline benchmarks
│   ├── e2e_workflow.py             # End-to-end file-to-memory
│   └── zero_copy_ingestion.py      # Zero-copy path benchmarks
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
    └── 2026-06-12-streaming-analysis/  # main vs OpenCV/Pillow/torchvision,
                                        # streaming-engine deep dive, raw JSON
```

## Frameworks Compared

| Framework | Description |
|-----------|-------------|
| `polars-cv-eager` | polars-cv with standard `.collect()` |
| `polars-cv-streaming` | polars-cv with `.collect(engine="streaming")` |
| `opencv` | NumPy + OpenCV (industry standard baseline) |
| `pillow` | PIL/Pillow |
| `torchvision-cpu` | torchvision on CPU |
| `torchvision-mps` | torchvision on Apple Metal GPU |

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
