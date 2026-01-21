# Geometry Operations

polars-cv provides comprehensive geometry operations through the `.contour` namespace on Polars expressions.

## Contour Schema

Contours are stored as Polars Struct columns:

```python
from polars_cv import CONTOUR_SCHEMA, POINT_SCHEMA, BBOX_SCHEMA

# POINT_SCHEMA: Struct({x: f64, y: f64})
# CONTOUR_SCHEMA: Struct({exterior: List(POINT), holes: List(List(POINT)), is_closed: bool})
# BBOX_SCHEMA: Struct({x: f64, y: f64, width: f64, height: f64})
```

## Measurements

Compute geometric properties directly on columns:

```python
df.with_columns(
    area=pl.col("contour").contour.area(),
    perimeter=pl.col("contour").contour.perimeter(),
    centroid=pl.col("contour").contour.centroid(),
    bbox=pl.col("contour").contour.bounding_box(),
)
```

## Transforms

```python
df.with_columns(
    moved=pl.col("contour").contour.translate(dx=10, dy=20),
    scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0),
    simple=pl.col("contour").contour.simplify(tolerance=1.0),
    hull=pl.col("contour").contour.convex_hull(),
)
```

## Rasterization

Convert contours to binary masks using pipelines:

```python
# Rasterize to 200x200 mask
pipe = Pipeline().source("contour", width=200, height=200)

result = df.with_columns(
    mask=pl.col("contour").cv.pipe(pipe).sink("numpy")
)
```

### Shape Inference

Infer dimensions from an existing image pipeline:

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").resize(200, 200))

# Rasterize contour to match image dimensions
mask = pl.col("contour").cv.pipe(Pipeline().source("contour", shape=img))
```

## Native Mask Metrics

Pixel-based metrics for binary masks:

```python
from polars_cv import mask_iou, mask_dice

result = df.with_columns(
    iou=mask_iou(pred_expr, gt_expr),
    dice=mask_dice(pred_expr, gt_expr),
)
```

