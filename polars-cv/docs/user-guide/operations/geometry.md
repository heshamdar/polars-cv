# Geometry Operations

polars-cv provides three expression namespaces for geometry: `.contour` for polygon operations, `.point` for point operations, and `.bbox` for bounding-box operations.

## Expression parameters

Numeric parameters in these namespaces accept either a literal or a **Polars
expression**, resolved per row at execution time — the same rule as the image
operations:

```python
# Normalize each contour against its own image's dimensions
df.with_columns(
    norm=pl.col("contour").contour.normalize(pl.col("img_w"), pl.col("img_h"))
)
```

This covers `normalize`, `to_absolute`, `translate`, `scale`, `simplify`,
`area(signed=)` and `match_detections(threshold=)` on `.contour`; `normalize`,
`to_absolute`, `translate`, `scale`, `rotate(angle=)` and `interpolate(t=)` on
`.point`; and `match_detections(threshold=)` on `.bbox`.

Structural parameters stay literal-only, as they do elsewhere: `scale`'s
`origin`, `ensure_winding`'s `direction` and `match_detections`' `strategy`
select behaviour rather than carrying a value.

An aggregation broadcasts, matching Polars' own semantics — `pl.col("w").max()`
produces one value applied to every row.

A null parameter raises by default. `on_null(...)` on the accessor opts into a
null result for the affected rows instead, mirroring `Pipeline.on_null_param`
(these namespaces have no `Pipeline` object, so the policy chains ahead of the
call):

```python
df.with_columns(
    norm=pl.col("contour").contour.on_null("null").normalize(pl.col("w"), 100)
)
```

For a fallback value instead, fill the null in the expression:
`pl.col("w").fill_null(1.0)`.

## Schemas

Geometry data uses Polars Struct columns:

```python
from polars_cv import CONTOUR_SCHEMA, POINT_SCHEMA, BBOX_SCHEMA

# POINT_SCHEMA: Struct({x: f64, y: f64})
# BBOX_SCHEMA: Struct({x: f64, y: f64, width: f64, height: f64})
# CONTOUR_SCHEMA: Struct({exterior: List(POINT), holes: List(List(POINT)), is_closed: bool})
```

---

## Contours

The `.contour` namespace operates on polygon columns.

### Measurements

```python
df.with_columns(
    area=pl.col("contour").contour.area(),
    perimeter=pl.col("contour").contour.perimeter(),
    centroid=pl.col("contour").contour.centroid(),
    bbox=pl.col("contour").contour.bounding_box(),
)
```

### Transforms

```python
df.with_columns(
    moved=pl.col("contour").contour.translate(dx=10, dy=20),
    scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0),
    simplified=pl.col("contour").contour.simplify(tolerance=1.0),
    hull=pl.col("contour").contour.convex_hull(),
)
```

### Rasterization

Convert contours to binary masks:

```python
pipe = Pipeline().source("contour", width=200, height=200)

result = df.with_columns(
    mask=pl.col("contour").cv.pipe(pipe).sink("numpy")
)
```

Infer dimensions from an existing image:

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").resize(height=200, width=200))
mask = pl.col("contour").cv.pipe(Pipeline().source("contour", shape=img))
```

`shape=` takes a **reference pipeline** (a `LazyPipelineExpr`) instead of literal
dimensions: the referenced pipeline's output `[H, W]` is resolved and used to size
the raster canvas, so the mask matches the source image without hard-coding its
size. The same `shape=` reference is accepted by `rasterize(...)` directly when
you already have a contour-domain pipeline:

```python
mask = (
    pl.col("contour").cv.pipe(Pipeline().source("contour"))
    .rasterize(shape=img)
)
```

Provide either explicit `width`/`height` **or** `shape=` — not both.

---

## Points

The `.point` namespace operates on point columns.

### Transforms

```python
df.with_columns(
    normalized=pl.col("point").point.normalize(width=100, height=100),
    absolute=pl.col("point").point.to_absolute(width=100, height=100),
    moved=pl.col("point").point.translate(dx=10, dy=20),
    scaled=pl.col("point").point.scale(sx=2.0, sy=2.0),
    rotated=pl.col("point").point.rotate(math.pi / 2),
)
```

### Distances

```python
df.with_columns(
    euclidean=pl.col("p1").point.distance(pl.col("p2")),
    manhattan=pl.col("p1").point.manhattan_distance(pl.col("p2")),
    to_boundary=pl.col("point").point.distance_to_contour(pl.col("contour")),
    signed=pl.col("point").point.signed_distance_to_contour(pl.col("contour")),
)
```

### Geometric Operations

```python
df.with_columns(
    angle=pl.col("p1").point.angle_to(pl.col("p2")),
    mid=pl.col("p1").point.midpoint(pl.col("p2")),
    interp=pl.col("p1").point.interpolate(pl.col("p2"), t=0.25),
    nearest=pl.col("point").point.nearest_point_on_contour(pl.col("contour")),
    inside=pl.col("point").point.within_bbox(pl.col("bbox")),
)
```

---

## Bounding Boxes

The `.bbox` namespace operates on `List[BBOX_SCHEMA]` columns for detection tasks.

### Pairwise IoU

Compute IoU between two sets of bounding boxes:

```python
df.with_columns(
    iou_matrix=pl.col("pred_bboxes").bbox.pairwise_iou(pl.col("gt_bboxes")),
)
```

### Detection Matching

Greedy one-to-one matching between predicted and ground-truth bounding boxes:

```python
df.with_columns(
    match_result=pl.col("pred_bboxes").bbox.match_detections(
        pl.col("gt_bboxes"),
        threshold=0.5,
        scores=pl.col("pred_scores"),
    ),
)
```

---

## Mask Metrics

Pixel-based metrics for binary masks:

```python
from polars_cv import mask_iou, mask_dice

result = df.with_columns(
    iou=mask_iou(pred_expr, gt_expr),
    dice=mask_dice(pred_expr, gt_expr),
)
```
