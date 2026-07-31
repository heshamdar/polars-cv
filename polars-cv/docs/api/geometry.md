# Geometry

API reference for geometry operations on contours and points.

## Schemas

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

### CONTOUR_SCHEMA

```python
from polars_cv import CONTOUR_SCHEMA

# Struct({
#     exterior: List({x: Float64, y: Float64}),
#     holes: List(List({x: Float64, y: Float64})),
#     is_closed: Boolean
# })
```

## Helper Functions

```python
from polars_cv.geometry.schemas import contour_from_points

contour = contour_from_points([
    (10, 10), (10, 90), (90, 90), (90, 10)
])
```

## Per-Row Parameters and Nulls

Numeric parameters on all three namespaces accept a `pl.Expr` as well as a
literal, resolved per row. A null in such a column raises by default;
`on_null("null")` — shared by `.contour`, `.point` and `.bbox` — yields null for
the affected rows instead:

```python
df.with_columns(
    norm=pl.col("contour").contour.on_null("null").normalize(pl.col("w"), 100)
)
```

`on_null()` returns a copy of the accessor with the policy applied and chains
ahead of the call, so it never appears in a method signature. The `.cv`
namespace has no equivalent: its parameters belong to a `Pipeline`, so the
control there is [`Pipeline.on_null_param()`](pipeline.md). See
[Geometry Operations](../user-guide/operations/geometry.md#expression-parameters)
for which parameters are per-row.

## ContourNamespace

::: polars_cv.geometry.contours.ContourNamespace
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## PointNamespace

::: polars_cv.geometry.points.PointNamespace
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## BBoxNamespace

::: polars_cv.geometry.bbox.BBoxNamespace
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
