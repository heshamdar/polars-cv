# polars-cv Benchmarks

Comprehensive benchmarking suite for comparing polars-cv against other vision processing frameworks.

## Benchmark Scope

These benchmarks focus on **batch preprocessing for inference** and **ETL workloads**.
polars-cv is designed for deterministic batch processing, not per-sample random
augmentation used in training. For training workloads, polars-cv handles heavy
preprocessing while PyTorch handles random augmentation.

### What These Benchmarks Test

- ✅ Batch image decoding and preprocessing
- ✅ Single operations (resize, crop, normalize, etc.)
- ✅ Multi-operation pipelines
- ✅ End-to-end file-to-memory workflows
- ✅ Inference serving throughput

### What These Benchmarks Do NOT Test

- ❌ Random augmentation (not supported by polars-cv)
- ❌ Training data loading with per-epoch variation
- ❌ GPU-based augmentation pipelines

## Frameworks Compared

| Framework | Description |
|-----------|-------------|
| `polars-cv-eager` | polars-cv with standard `.collect()` |
| `polars-cv-streaming` | polars-cv with `.collect(engine="streaming")` |
| `opencv` | NumPy + OpenCV (industry standard) |
| `pillow` | PIL/Pillow (pure Python baseline) |
| `torchvision-cpu` | torchvision on CPU |
| `torchvision-mps` | torchvision on Apple Metal GPU |

## Installation

Install benchmark dependencies:

```bash
cd polars-cv
uv sync --group bench

# Or with pip (>= 25.1, which understands dependency groups)
pip install -e . --group bench
```

> **Note:** the benchmark dependencies (torch, torchvision, …) live in a
> `[dependency-groups]` group named `bench`, **not** a
> `[project.optional-dependencies]` extra — so `pip install ".[bench]"` matches
> no extra and installs nothing. Because `bench` is not a default group, prefix
> the commands below with `uv run --no-sync` (or activate `.venv` and call
> `python` directly), otherwise `uv run` re-syncs and drops the group.

## Quick Start

```bash
# Run all benchmarks with defaults
uv run --no-sync python -m benchmarks.run_benchmarks

# List available frameworks
uv run --no-sync python -m benchmarks.run_benchmarks --list-frameworks
```

## Usage

### Run Specific Scenarios

```bash
# Single operation benchmarks only
uv run python -m benchmarks.run_benchmarks --scenario single_ops

# Pipeline benchmarks only
uv run python -m benchmarks.run_benchmarks --scenario pipelines

# End-to-end workflow benchmarks only
uv run python -m benchmarks.run_benchmarks --scenario e2e
```

### Select Frameworks

```bash
# Compare only specific frameworks
uv run python -m benchmarks.run_benchmarks --frameworks opencv,pillow

# Compare polars-cv modes
uv run python -m benchmarks.run_benchmarks --frameworks polars-cv-eager,polars-cv-streaming
```

### Configure Image Sizes and Counts

```bash
# Custom image counts
uv run python -m benchmarks.run_benchmarks --counts 10,50,100

# Custom image sizes
uv run python -m benchmarks.run_benchmarks --sizes 224,512,1024

# Both
uv run python -m benchmarks.run_benchmarks --counts 100,500 --sizes 256,512
```

### Output Formats

```bash
# Table output (default)
uv run python -m benchmarks.run_benchmarks --output table

# JSON output (for further analysis)
uv run python -m benchmarks.run_benchmarks --output json > results.json

# CSV output
uv run python -m benchmarks.run_benchmarks --output csv > results.csv
```

### Validation

Verify all frameworks produce equivalent results:

```bash
# Recommended tolerance (accounts for implementation differences)
uv run python -m benchmarks.run_benchmarks --validate --tolerance 0.15

# Strict tolerance (will show minor differences)
uv run python -m benchmarks.run_benchmarks --validate --tolerance 1e-5
```

**Validation Notes:**
- **Reference framework**: OpenCV is used as the baseline for all comparisons
- All frameworks now use aligned configurations for fair comparison:
  - **Grayscale**: BT.601 coefficients (0.299R + 0.587G + 0.114B) - matches OpenCV/Pillow/Torchvision
  - **Resize**: Bilinear interpolation for all frameworks
  - **Threshold**: Uses 127 instead of 128 to avoid boundary edge cases
- Binary image comparisons (like threshold) use a percentage-based metric (fraction of differing pixels) instead of max absolute error
- Grayscale outputs are squeezed from (H, W, 1) to (H, W) for consistent shape comparison

**Expected errors by operation** (these are inherent algorithmic differences):

| Operation | Max Error | Reason |
|-----------|-----------|--------|
| resize | ~10-27% | Different bilinear implementations (sub-pixel sampling, boundary handling) |
| grayscale | ~0.4% | Float precision differences in coefficient calculations |
| blur | ~5-8% | Different Gaussian blur kernel implementations |
| threshold | ~0-1% | Integer vs float precision at boundaries |
| flip/crop/normalize | ~0% | Identical implementations across libraries |

**Recommended tolerances:**
- `--tolerance 0.05` passes 6/8 operations (excludes resize, blur)
- `--tolerance 0.15` passes all 8 operations

### Pipeline Complexity Filter

```bash
# Only light pipelines (2 ops)
uv run python -m benchmarks.run_benchmarks --scenario pipelines --complexity light

# Only heavy pipelines (6 ops)
uv run python -m benchmarks.run_benchmarks --scenario pipelines --complexity heavy
```

### Benchmark Iterations

```bash
# More iterations for stable results
uv run python -m benchmarks.run_benchmarks --warmup 5 --iterations 20
```

## Benchmark Scenarios

### Single Operations

Individual operation benchmarks:

| Benchmark | Description | Framework Support |
|-----------|-------------|-------------------|
| `resize` | Resize to 224x224 | All |
| `grayscale` | RGB to grayscale | All |
| `normalize` | Min-max to [0,1] | All |
| `flip_horizontal` | Horizontal flip | All |
| `flip_vertical` | Vertical flip | All |
| `crop_center` | Center crop to 128x128 | All |
| `blur` | Gaussian blur (sigma=2.0) | All |
| `threshold` | Binary threshold at 127 | All |
| `rotate_90` | 90-degree rotation (zero-copy in polars-cv) | All |
| `rotate_45` | 45-degree rotation with expansion | All |
| `invert` | Pixel inversion (255 - pixel) | OpenCV, Pillow, Torchvision |
| `adjust_contrast` | Contrast adjustment (factor=1.5) | OpenCV, Pillow, Torchvision |
| `adjust_brightness` | Brightness adjustment (factor=1.3) | OpenCV, Pillow, Torchvision |
| `sharpen` | Unsharp-mask sharpening (strength=1.5) | OpenCV, Pillow, Torchvision |
| `pad` | 20px constant padding on all edges | All |
| `erode` | Morphological erosion (3x3) | OpenCV, Pillow |
| `dilate` | Morphological dilation (3x3) | OpenCV, Pillow |
| `histogram_equalize` | Histogram equalization | OpenCV, Pillow, Torchvision |
| `canny` | Canny edge detection (50/150) | OpenCV |
| `sobel_x` | Sobel gradient (X axis) | OpenCV |

**Note:** All polars-cv operations are supported. Operations listed with limited framework support will be skipped for unsupported frameworks.

### Pipelines

Chained operation benchmarks:

| Pipeline | Operations | Complexity |
|----------|------------|------------|
| `light_pipeline` | resize + normalize | light |
| `medium_pipeline` | resize + crop + normalize + flip | medium |
| `heavy_pipeline` | resize + flip + grayscale + blur + normalize + threshold | heavy |
| `imagenet_preprocess` | resize + center crop + normalize | medium |
| `medical_pipeline` | grayscale + normalize + resize | medium |

### End-to-End Workflows

Complete file-to-memory workflows:
- Load images from temporary PNG files
- Apply processing pipeline
- Return results in memory

## Output Example

```
=== Benchmark Results: resize | 100 images | 256x256 ===

| Framework                   | Throughput (img/s) |  Latency (ms/img) |  Memory (MB) |
|-----------------------------|--------------------|-------------------|--------------|
| polars-cv (streaming)   |            15,230  |              0.07 |        128.5 |
| polars-cv (eager)       |            12,450  |              0.08 |        245.3 |
| torchvision (mps) [warm]    |            28,900  |              0.03 |        198.7 |
| torchvision (mps) [cold]    |             9,450  |              0.11 |        412.1 |
| opencv                      |             8,340  |              0.12 |        312.8 |
| torchvision (cpu)           |             6,890  |              0.15 |        356.2 |
| pillow                      |             3,210  |              0.31 |        289.4 |

Output Validation: PASS (all outputs within tolerance)
```

## GPU Benchmarks

For GPU-capable frameworks (torchvision-mps, torchvision-cuda), two measurements are provided:

- **cold**: Includes data transfer time from CPU to GPU
- **warm**: Data already on GPU (pre-transferred)

This provides fair comparison for both real-world (cold) and optimized (warm) scenarios.

## Extending the Benchmarks

### Adding a New Framework

1. Create a new adapter in `benchmarks/frameworks/`:

```python
from benchmarks.frameworks.base import BaseFrameworkAdapter

class MyFrameworkAdapter(BaseFrameworkAdapter):
    name = "my-framework"
    supports_gpu = False

    def is_available(self) -> bool:
        try:
            import my_framework
            return True
        except ImportError:
            return False

    # Implement all abstract methods...
```

2. Register in `benchmarks/frameworks/__init__.py`

### Adding a New Operation

1. Add to `OperationType` enum in `base.py`
2. Implement in all adapters
3. Add benchmark config in `scenarios/single_ops.py`

### Adding a New Pipeline

Add to `get_pipeline_benchmarks()` in `scenarios/pipelines.py`:

```python
PipelineBenchmarkConfig(
    name="my_pipeline",
    operations=[...],
    description="...",
    complexity="medium",
)
```

## Inference Pipeline Comparison

Compare polars-cv against HuggingFace/torchvision for batch inference preprocessing:

```bash
# Run the inference comparison benchmark
uv run python -m benchmarks.inference_pipeline_comparison --num-images 1000

# With custom settings
uv run python -m benchmarks.inference_pipeline_comparison \
    --num-images 5000 \
    --batch-size 64 \
    --skip-serving  # Skip model inference portion
```

This benchmark:

- Compares **batch preprocessing time** (both frameworks do upfront preprocessing)
- Uses HuggingFace `.map(batched=True)` for fair comparison
- Measures DataLoader throughput from preprocessed data
- Optionally runs inference serving throughput comparison

**Note**: This is a fair comparison because both pipelines preprocess all data upfront
before measuring DataLoader/serving throughput. It does not compare training with
random augmentation, which polars-cv intentionally does not support.

## Architecture

```
benchmarks/
├── __init__.py
├── conftest.py                        # Pytest fixtures
├── run_benchmarks.py                  # Main benchmark CLI
├── inference_pipeline_comparison.py   # HuggingFace vs polars-cv
├── frameworks/                        # Framework adapters
│   ├── base.py                        # Abstract base class
│   ├── polars_cv_adapter.py
│   ├── opencv_adapter.py
│   ├── pillow_adapter.py
│   └── torchvision_adapter.py
├── scenarios/                         # Benchmark scenarios
│   ├── single_ops.py                  # Individual operations
│   ├── pipelines.py                   # Chained operations
│   └── e2e_workflow.py                # File-to-memory workflows
└── utils/
    ├── data_gen.py                    # Test image generation
    ├── memory.py                      # Memory profiling
    ├── results.py                     # Results formatting
    └── validation.py                  # Output validation
```

## Tips

1. **Stable Results**: Use more iterations (`--iterations 20`) for stable measurements
2. **Memory Accuracy**: Memory measurements are more accurate with fewer concurrent processes
3. **GPU Warmup**: GPU benchmarks automatically include warmup iterations
4. **Fair Comparison**: Use `--validate` to ensure all frameworks produce equivalent results
