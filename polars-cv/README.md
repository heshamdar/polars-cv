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

## Scaling to Large Datasets (Streaming)

The `.cv.pipe(...)` expression is a normal elementwise Polars plugin, so the
**parallelism comes from Polars' engine, not from inside the plugin**. On a
plain `DataFrame.with_columns(...)`/`.select(...)` call (eager), the whole
column is processed on a single thread. For larger-than-a-handful image
workloads, run through the lazy **streaming** engine instead — Polars splits the
column into morsels and processes them across its worker pool, and can spill
intermediate state to disk when memory is tight:

```python
result = (
    df.lazy()
    .with_columns(processed=pl.col("image").cv.pipe(pipe).sink("blob"))
    .collect(engine="streaming")
)
```

This is the recommended path for anything beyond small/interactive use; the
plugin's per-morsel graph is cached, so per-morsel overhead is just a hash
lookup. (The detection-metrics APIs already collect with
`engine="streaming"` internally.)

## Source Behavior (Auto DType)

`image_bytes` and `file_path` sources decode image format and dtype at runtime:

- PNG/JPEG usually decode as `u8`
- 16-bit PNG decodes as `u16`
- TIFF may decode as `u8`, `u16`, `f32`, or `f64`

This means the pipeline dtype starts as `auto` for these sources unless you pin it with `dtype=...` or an operation that determines dtype (such as `normalize`, `threshold`, or `cast`).

```python
# Runtime decode with automatic dtype
auto_pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

# Pin expected dtype at source (runtime cast when needed)
typed_pipe = Pipeline().source("image_bytes", dtype="f32").resize(height=224, width=224)
```

When using `sink("list")` or `sink("array")`, dtype must be known at planning time. For `image_bytes` / `file_path`, choose one of:

- set `dtype` in `source(...)`
- add `.cast("...")`
- use a dtype-fixing operation before the sink

```python
pipe = Pipeline().source("file_path", dtype="f32").resize(height=224, width=224)

result = df.with_columns(values=pl.col("path").cv.pipe(pipe).sink("list"))
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

- **Image**: `resize`, `resize_scale`, `resize_to_height`, `resize_to_width`, `resize_max`, `resize_min`, `thumbnail`, `grayscale`, `blur`, `threshold`, `crop`, `rotate`, `pad`, `letterbox`, `flip_h`, `flip_v`.
- **Color**: `cvt_color`, `to_hsv`, `to_lab`, `to_bgr`, `to_ycbcr`.
- **Channels**: `channel_select`, `channel_swap`.
- **Intensity**: `adjust_contrast`, `adjust_gamma`, `adjust_brightness`, `invert`.
- **Convolution & Edge Detection**: `convolve2d`, `sobel`, `laplacian`, `sharpen`, `canny`.
- **Morphology**: `erode`, `dilate`, `morphology_open`, `morphology_close`, `morphology_gradient`.
- **Affine Transforms**: `warp_affine`, `shear`, `rotate_and_scale` (with automatic pipeline fusion).
- **Enhancement**: `equalize_histogram`.
- **Compute**: `normalize`, `scale`, `clamp`, `relu`, `cast`.
- **Layout**: `transpose`, `reshape`.
- **Geometry**: `extract_contours`, `rasterize`, `area`, `perimeter`, `centroid`, `bounding_box`.
- **Points**: `normalize`, `translate`, `scale`, `rotate`, `distance`, `manhattan_distance`, `distance_to_contour`, `signed_distance_to_contour`, `nearest_point_on_contour`, `angle_to`, `midpoint`, `interpolate`, `within_bbox`.
- **Bounding Boxes**: `pairwise_iou`, `match_detections` (via `.bbox` namespace).
- **Analysis**: `histogram`, `perceptual_hash`, `extract_shape`, `label_reduce`.
- **Reductions**: `reduce_sum`, `reduce_mean`, `reduce_std`, `reduce_max`, `reduce_min`, `reduce_argmax`, `reduce_argmin`, `reduce_percentile`, `reduce_popcount`.
- **Metadata**: `.cv.width()`, `.cv.height()`, `.cv.channels()`, `.cv.image_dtype()`.
- **Display**: `show_images()` for Jupyter notebook visualization.
- **Detection Metrics**: Precision-Recall, AP, mAP, FROC, LROC, F1, confusion matrix, bootstrap confidence intervals.

## Detection Metrics

Evaluate object detectors with industry-standard metrics:

```python
from polars_cv.metrics import PreMatchedAdapter, precision_recall_curve, average_precision

# Wrap pre-matched detection data
adapter = PreMatchedAdapter()
table = adapter.match(df, pred_col="confidence", gt_col="is_tp", image_id_col="slide_id")

# Compute metrics
pr = precision_recall_curve(table)
ap = average_precision(table)

print(f"AP: {ap:.3f}")
print(pr.summary_table())
```

Available matchers: `ContourMatcher` (heatmap/mask), `BBoxMatcher` (bounding boxes),
`PreMatchedAdapter` (pre-computed TP/FP).

Available metrics: `precision_recall_curve`, `average_precision`,
`mean_average_precision`, `froc_curve`, `lroc_curve`, `confusion_at_threshold`,
`precision_at_threshold`, `recall_at_threshold`, `f1_at_threshold`.

For full details, see the [Documentation](https://heshamdar.github.io/polars-cv/)
