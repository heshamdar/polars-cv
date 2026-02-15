# polars-cv
**ℹ️ Note:**
This is a largely AI developed project and still in its early stages. Use at your own discretion.

A Polars plugin for high-performance vision and array operations.

## Features

- **Modular Pipelines**: Define image processing pipelines and apply them to DataFrame columns.
- **Expression Arguments**: Use Polars expressions for dynamic, per-row parameters.
- **Zero-Copy Performance**: Efficient memory management with stride-aware operations.
- **Multi-Domain**: Seamlessly move between images, geometry (contours), and numeric results.

## Installation

```bash
pip install polars-cv
```

## Quick Start

```python
import polars as pl
from polars_cv import Pipeline

# Define a pipeline and apply it to a column
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()

df = pl.DataFrame({"image": [img1_bytes, img2_bytes]})
result = df.with_columns(
    processed=pl.col("image").cv.pipe(pipe).sink("numpy")
)
```

## Source Behavior (Auto DType)

`image_bytes` and `file_path` sources decode image format and dtype at runtime:

- PNG/JPEG usually decode as `u8`
- 16-bit PNG decodes as `u16`
- TIFF may decode as `u8`, `u16`, `f32`, or `f64`

This means the pipeline dtype starts as `auto` for these sources unless you pin it with `dtype=...` or an operation that determines dtype (such as `normalize`, `threshold`, or `cast`).

```python
# Runtime decode with automatic dtype
auto_pipe = Pipeline().source("image_bytes").resize(224, 224)

# Pin expected dtype at source (runtime cast when needed)
typed_pipe = Pipeline().source("image_bytes", dtype="f32").resize(224, 224)
```

When using `sink("list")` or `sink("array")`, dtype must be known at planning time. For `image_bytes` / `file_path`, choose one of:

- set `dtype` in `source(...)`
- add `.cast("...")`
- use a dtype-fixing operation before the sink

```python
safe_for_list = (
    Pipeline()
    .source("file_path", dtype="f32")
    .resize(224, 224)
    .sink("list")
)
```

## Dynamic Pipelines

Use Polars expressions for per-row parameter values:

```python
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=pl.col("target_h"), width=pl.col("target_w"))
    .crop(top=pl.col("crop_y"), left=pl.col("crop_x"), height=100, width=100)
)

df = pl.DataFrame({
    "image": [img1_bytes, img2_bytes],
    "target_h": [224, 256],
    "target_w": [224, 256],
    "crop_x": [10, 20],
    "crop_y": [5, 15],
})

result = df.with_columns(
    processed=pl.col("image").cv.pipe(pipe).sink("numpy")
)
```

## Operations

- **Image**: `resize`, `grayscale`, `blur`, `threshold`, `crop`, `rotate`, `pad`, `flip`.
- **Compute**: `normalize`, `scale`, `clamp`, `relu`, `cast`.
- **Geometry**: `extract_contours`, `rasterize`, `area`, `perimeter`, `centroid`, `bounding_box`.
- **Points**: `normalize`, `translate`, `scale`, `rotate`, `distance`, `manhattan_distance`, `distance_to_contour`, `signed_distance_to_contour`, `nearest_point_on_contour`, `angle_to`, `midpoint`, `interpolate`, `within_bbox`.
- **Analysis**: `histogram`, `perceptual_hash`, `extract_shape`.
- **Reductions**: `reduce_sum`, `reduce_mean`, `reduce_std`, `reduce_max`, `reduce_min`, `reduce_percentile`.

For full details, see the [Documentation](https://heshamdar.github.io/polars-cv/)
