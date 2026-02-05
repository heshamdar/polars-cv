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
