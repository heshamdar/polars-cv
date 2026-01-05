# polars-cv

A Polars plugin for vision/array operations, powered by [view-buffer](../view-buffer).

## Features

- **Lazy Pipeline Definition**: Define image processing pipelines outside DataFrame context
- **Expression Arguments**: Use Polars expressions for dynamic, per-row parameters
- **Zero-Copy Where Possible**: Leverages view-buffer's stride-aware operations
- **Multiple Source/Sink Formats**: PNG, JPEG, NumPy, PyTorch, and more

## Installation

```bash
# From source (requires Rust toolchain)
cd polars-cv
maturin develop --release

# Or with pip (once published)
pip install polars-cv
```

## Quick Start

```python
import polars as pl
from polars_cv import Pipeline

# Define a static pipeline
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
    .normalize(method="minmax")
    .sink("numpy")
)

# Apply to DataFrame
df = pl.DataFrame({"images": [img1_bytes, img2_bytes]})
result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
```

## Dynamic Pipelines

Use Polars expressions for per-row parameter values:

```python
# Dynamic pipeline with expression arguments
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=pl.col("target_h"), width=pl.col("target_w"))
    .crop(top=pl.col("crop_y"), left=pl.col("crop_x"), height=100, width=100)
    .sink("numpy")
)

df = pl.DataFrame({
    "images": [img1_bytes, img2_bytes],
    "target_h": [224, 256],
    "target_w": [224, 256],
    "crop_x": [10, 20],
    "crop_y": [5, 15],
})

result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
```

## Pipeline Operations

### Source Formats

| Format | Description |
|--------|-------------|
| `image_bytes` | Decode PNG/JPEG (auto-detect) |
| `blob` | VIEW protocol binary |
| `raw` | Raw bytes (requires dtype) |
| `file_path` | Read from file path |

### Operations

**View Operations (Zero-Copy)**
- `transpose(axes)` - Permute dimensions
- `reshape(shape)` - Reshape array
- `flip(axes)` / `flip_h()` / `flip_v()` - Flip along axes
- `crop(top, left, height, width)` - Crop region

**Compute Operations**
- `cast(dtype)` - Change data type
- `scale(factor)` - Multiply by factor
- `normalize(method)` - MinMax or ZScore normalization
- `clamp(min, max)` - Clamp to range

**Image Operations**
- `resize(height, width, filter)` - Resize image
- `grayscale()` - Convert to grayscale
- `threshold(value)` - Binary threshold
- `blur(sigma)` - Gaussian blur

### Sink Formats

| Format | Description |
|--------|-------------|
| `numpy` | NumPy-compatible bytes |
| `torch` | PyTorch-compatible bytes |
| `png` | Re-encode as PNG |
| `jpeg` | Re-encode as JPEG (with quality) |
| `blob` | VIEW protocol (for chaining) |
| `array` | Polars Array type (fixed shape) |
| `list` | Polars nested List (variable shape) |

## Shape Hints

Provide shape information to help pipeline planning:

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .assert_shape(height=256, width=256, channels=3)
    .resize(height=224, width=224)
    .sink("numpy")
)
```

## Working with List Columns

For batch processing (list of images per row):

```python
batch_df = pl.DataFrame({
    "image_batches": [[img1, img2], [img3, img4, img5]],
})

result = batch_df.with_columns(
    pl.col("image_batches").list.eval(
        pl.element().cv.pipeline(pipe)
    )
)
```

## Development

```bash
# Run Python tests
pytest tests/

# Build for development
maturin develop

# Build release
maturin build --release
```

## CI/CD and Publishing

This project uses GitHub Actions for continuous integration and publishing to PyPI.

### Workflows

- **CI** (`ci.yml`): Runs on push/PR to main
  - Linting (ruff, cargo clippy, cargo fmt)
  - Tests across Python 3.9-3.12 on Linux, macOS, Windows
  - Build verification

- **Publish** (`publish.yml`): Runs on release creation
  - Builds wheels for all platforms (Linux, macOS universal2, Windows)
  - Publishes to TestPyPI first for validation
  - Publishes to PyPI after TestPyPI succeeds

### Required GitHub Secrets

To enable publishing, configure these secrets in your GitHub repository settings
(Settings → Secrets and variables → Actions → New repository secret):

| Secret | Description | Source |
|--------|-------------|--------|
| (none required) | Uses trusted publishing with OIDC | Configure on PyPI |

### PyPI Trusted Publisher Setup

This project uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) with OIDC,
which is more secure than API tokens. To set it up:

1. **PyPI** (https://pypi.org):
   - Go to your account → Publishing → Add a new pending publisher
   - Owner: `<your-github-username>`
   - Repository name: `polars_plugin_dev`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

2. **TestPyPI** (https://test.pypi.org):
   - Same steps as above
   - Environment name: `testpypi`

### GitHub Environments

Create two environments in your repository (Settings → Environments):

1. **testpypi** - For TestPyPI publishing
2. **pypi** - For production PyPI publishing (consider adding required reviewers)

### Release Process

1. Update version in both `Cargo.toml` and `pyproject.toml`
2. Commit and push to main
3. Create a GitHub release with a version tag (e.g., `v0.1.0`)
4. GitHub Actions automatically:
   - Builds wheels for all platforms
   - Publishes to TestPyPI
   - Tests installation from TestPyPI
   - Publishes to PyPI

### Manual Publishing (Alternative)

If you prefer using API tokens instead of trusted publishing:

```bash
# Build wheels
maturin build --release

# Publish to TestPyPI
maturin publish --repository testpypi

# Publish to PyPI
maturin publish
```

## License

MIT OR Apache-2.0

