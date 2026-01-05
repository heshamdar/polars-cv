# Geometry Operations

polars-cv provides comprehensive geometry operations through the `.contour` namespace.

## Contour Schema

Contours are stored as Polars Struct columns:

```python
from polars_cv import CONTOUR_SCHEMA, POINT_SCHEMA, BBOX_SCHEMA

# POINT_SCHEMA: Struct({x: Float64, y: Float64})
# CONTOUR_SCHEMA: Struct({
#     exterior: List(POINT_SCHEMA),
#     holes: List(List(POINT_SCHEMA)),
#     is_closed: Boolean
# })
# BBOX_SCHEMA: Struct({x: Float64, y: Float64, width: Float64, height: Float64})
```

## Creating Contours

```python
from polars_cv.geometry.schemas import contour_from_points
import polars as pl

# From a list of (x, y) tuples
contour = contour_from_points([
    (10, 10), (10, 90), (90, 90), (90, 10)
])

# Create DataFrame with contour column
df = pl.DataFrame({"contour": [contour]}).cast({"contour": CONTOUR_SCHEMA})
```

## Measurements

### Area

Compute the area enclosed by the contour.

```python
df.with_columns(area=pl.col("contour").contour.area())

# Signed area (positive for CCW, negative for CW)
df.with_columns(signed_area=pl.col("contour").contour.area(signed=True))
```

### Perimeter

Compute the perimeter (total edge length).

```python
df.with_columns(perimeter=pl.col("contour").contour.perimeter())
```

### Winding Direction

Get the winding direction of the contour.

```python
df.with_columns(winding=pl.col("contour").contour.winding())
# Returns "cw" (clockwise) or "ccw" (counter-clockwise)
```

### Centroid

Compute the geometric center.

```python
df.with_columns(centroid=pl.col("contour").contour.centroid())
# Returns Struct({x: Float64, y: Float64})
```

### Bounding Box

Compute the axis-aligned bounding box.

```python
df.with_columns(bbox=pl.col("contour").contour.bounding_box())
# Returns Struct({x: Float64, y: Float64, width: Float64, height: Float64})
```

## Predicates

### Is Convex

Check if the contour is convex.

```python
df.with_columns(is_convex=pl.col("contour").contour.is_convex())
```

### Contains Point

Check if a point is inside the contour.

```python
df.with_columns(
    contains=pl.col("contour").contour.contains_point(pl.col("point"))
)
```

## Pairwise Metrics

### IoU (Intersection over Union)

Compute IoU between two contours.

```python
df.with_columns(
    iou=pl.col("contour_a").contour.iou(pl.col("contour_b"))
)
```

### Dice Coefficient

Compute Dice coefficient (F1 score for overlap).

```python
df.with_columns(
    dice=pl.col("contour_a").contour.dice(pl.col("contour_b"))
)
```

### Hausdorff Distance

Compute the maximum distance between contour points.

```python
df.with_columns(
    distance=pl.col("contour_a").contour.hausdorff(pl.col("contour_b"))
)
```

## Transforms

### Translate

Move the contour by an offset.

```python
df.with_columns(
    moved=pl.col("contour").contour.translate(dx=10, dy=20)
)
```

### Scale

Scale the contour around its centroid.

```python
df.with_columns(
    scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0)
)
```

### Simplify

Simplify using Douglas-Peucker algorithm.

```python
df.with_columns(
    simple=pl.col("contour").contour.simplify(tolerance=1.0)
)
```

### Flip

Reverse the winding direction.

```python
df.with_columns(
    flipped=pl.col("contour").contour.flip()
)
```

### Convex Hull

Compute the convex hull.

```python
df.with_columns(
    hull=pl.col("contour").contour.convex_hull()
)
```

### Ensure Winding

Ensure a specific winding direction.

```python
df.with_columns(
    ccw=pl.col("contour").contour.ensure_winding(direction="ccw")
)
```

### Normalize Coordinates

Normalize coordinates to [0, 1] range.

```python
df.with_columns(
    normalized=pl.col("contour").contour.normalize(ref_width=200, ref_height=200)
)
```

### To Absolute

Convert normalized coordinates to absolute pixels.

```python
df.with_columns(
    absolute=pl.col("contour").contour.to_absolute(ref_width=200, ref_height=200)
)
```

## Rasterization

Convert contours to binary masks using pipelines:

```python
from polars_cv import Pipeline

# Rasterize to 200x200 mask
pipe = Pipeline().source("contour", width=200, height=200).sink("numpy")

result = df.with_columns(mask=pl.col("contour").cv.pipeline(pipe))
```

### Shape Inference

Infer dimensions from another pipeline:

```python
# Image pipeline
img = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(200, 200)
)

# Contour source with shape inference
mask = pl.col("contour").cv.pipe(
    Pipeline().source("contour", shape=img)
)
```

## Native Mask Metrics

For pixel-based metrics on rasterized masks:

```python
from polars_cv import mask_iou, mask_dice, Pipeline

# Define pipelines
pred_pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
gt_pipe = Pipeline().source("contour", width=200, height=200)

# Compute metrics
result = df.with_columns(
    iou=mask_iou(
        pl.col("prediction").cv.pipe(pred_pipe),
        pl.col("ground_truth").cv.pipe(gt_pipe),
    ),
    dice=mask_dice(
        pl.col("prediction").cv.pipe(pred_pipe),
        pl.col("ground_truth").cv.pipe(gt_pipe),
    ),
)
```

## Next Steps

- [Domains](../concepts/domains.md) - Multi-domain pipelines
- [ML Integration](../ml-integration.md) - Using geometry in ML

