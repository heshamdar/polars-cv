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

The `holes` field is the only thing that makes a ring a hole. Point order is never
read as a hole signal, and no operation requires a particular winding — `area`,
`centroid`, `contains_point`, `iou`, `dice` and rasterization all give the same
answer whichever way each ring is wound.

The region a contour describes is its exterior minus the **union** of its hole
rings. Overlapping rings are not subtracted twice, and a ring nested inside
another hole remains a hole. Winding is reported by
[`.contour.winding()`][polars_cv.geometry.contours.ContourNamespace.winding] and set
by [`.contour.ensure_winding()`][polars_cv.geometry.contours.ContourNamespace.ensure_winding],
and is consulted nowhere else.

`is_closed=False` makes a contour an open polyline, measured as a line by the
boundary functions and refused by the region ones (see
[Open polylines](../user-guide/operations/geometry.md#open-polylines)); an
unspecified `is_closed` reads as closed. Rings are implicitly closed, so do not
repeat the first point at the end.

### Other schemas

| Name | Type | What it holds |
|------|------|---------------|
| `RING_SCHEMA` | `List(POINT_SCHEMA)` | One ring: the `exterior` field, or one entry of `holes` |
| `CONTOUR_SET_SCHEMA` | `List(CONTOUR_SCHEMA)` | Several contours per row, as `extract_contours` returns |
| `POINT_SET_SCHEMA` | `List(POINT_SCHEMA)` | Several points per row (keypoints, landmarks) |
| `ANNOTATED_POINT_SCHEMA` | `Struct({x, y, label: String, confidence: Float64})` | A point with a label and score |
| `CORRESPONDENCE_SCHEMA` | `Struct({right_idx: List(UInt32), overlap: List(Float64), duplicate: List(Boolean)})` | What `.contour.correspond()` / `.bbox.correspond()` return, each list aligned with the left set |

All are importable from `polars_cv` and `polars_cv.geometry`.

## Helper Functions

```python
from polars_cv.geometry.schemas import contour_from_points

contour = contour_from_points([
    (10, 10), (10, 90), (90, 90), (90, 10)
])
```

## Per-Row Parameters and Nulls

Parameters on all three namespaces accept a `pl.Expr` as well as a literal,
resolved per row — numbers, and the enums and flags too, since eligibility is
decided by whether a value changes the output shape, rank or dtype rather than
by its type. A null in such a column raises by default;
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

`on_error("null")`, chained the same way, nulls a row whose **data** the
function refuses (e.g. `close_along_border` on a line that does not reach the
frame) instead of failing the query; an error about the column itself (its
arity or dtype) still raises. See
[Invalid rows](../user-guide/operations/geometry.md#invalid-rows).

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

## Constructors from coordinate lists

Build `POINT_SCHEMA` / `CONTOUR_SCHEMA` / `CONTOUR_SET_SCHEMA` columns from
plain `[x, y]` or `[y, x]` pairs; `.point.to_coords()` and
`.contour.to_coords()` are the inverse. `bbox_from_coords` builds
`BBOX_SCHEMA` boxes from four numbers in a named layout, one of
`BoxFormat` (`"xyxy"`, `"xywh"`, `"cxcywh"`; `from polars_cv.geometry import BoxFormat`).

::: polars_cv.geometry.coords.point_from_coords
    options:
      show_root_heading: true
      heading_level: 3

::: polars_cv.geometry.coords.contour_from_coords
    options:
      show_root_heading: true
      heading_level: 3

::: polars_cv.geometry.coords.contour_set_from_coords
    options:
      show_root_heading: true
      heading_level: 3

::: polars_cv.geometry.coords.bbox_from_coords
    options:
      show_root_heading: true
      heading_level: 3
