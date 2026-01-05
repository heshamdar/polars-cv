# Geometry

Geometry operations for contours and points.

## Schemas

### CONTOUR_SCHEMA

```python
from polars_cv import CONTOUR_SCHEMA

# Struct({
#     exterior: List({x: Float64, y: Float64}),
#     holes: List(List({x: Float64, y: Float64})),
#     is_closed: Boolean
# })
```

### POINT_SCHEMA

```python
from polars_cv import POINT_SCHEMA

# Struct({x: Float64, y: Float64})
```

### BBOX_SCHEMA

```python
from polars_cv import BBOX_SCHEMA

# Struct({x: Float64, y: Float64, width: Float64, height: Float64})
```

## Creating Contours

```python
from polars_cv.geometry.schemas import contour_from_points

contour = contour_from_points([
    (10, 10), (10, 90), (90, 90), (90, 10)
])
```

## ContourNamespace

The `.contour` namespace provides contour operations on Polars expressions.

### Measurements

```python
df.with_columns(
    area=pl.col("contour").contour.area(),
    perimeter=pl.col("contour").contour.perimeter(),
    winding=pl.col("contour").contour.winding(),
    centroid=pl.col("contour").contour.centroid(),
    bbox=pl.col("contour").contour.bounding_box(),
)
```

### Predicates

```python
df.with_columns(
    is_convex=pl.col("contour").contour.is_convex(),
    contains=pl.col("contour").contour.contains_point(pl.col("point")),
)
```

### Pairwise

```python
df.with_columns(
    iou=pl.col("contour_a").contour.iou(pl.col("contour_b")),
    dice=pl.col("contour_a").contour.dice(pl.col("contour_b")),
    hausdorff=pl.col("contour_a").contour.hausdorff(pl.col("contour_b")),
)
```

### Transforms

```python
df.with_columns(
    translated=pl.col("contour").contour.translate(dx=10, dy=20),
    scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0),
    simplified=pl.col("contour").contour.simplify(tolerance=1.0),
    flipped=pl.col("contour").contour.flip(),
    hull=pl.col("contour").contour.convex_hull(),
    normalized=pl.col("contour").contour.normalize(ref_width=200, ref_height=200),
    absolute=pl.col("contour").contour.to_absolute(ref_width=200, ref_height=200),
    ensured=pl.col("contour").contour.ensure_winding(direction="ccw"),
)
```

## API Reference

::: polars_cv.geometry.contours.ContourNamespace
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

