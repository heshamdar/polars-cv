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

Eligibility is decided by **effect, not type**: a parameter may be per-row when
its value changes no output shape, rank or dtype. That admits the enums and
flags as well as the numbers — `ensure_winding(direction=)` and
`scale(origin=)` reorder or move a ring's vertices and leave
`List(Struct(CONTOUR_SCHEMA))` exactly as it was, so both take an expression
too.

On `.contour`: `normalize`, `to_absolute`, `translate`, `scale` (`sx`, `sy` and
`origin`), `simplify`, `ensure_winding(direction=)`, `area(signed=)`,
`label_reduce(reduction=, region_mode=)` and `match_detections(threshold=)`.
On `.point`: `normalize`, `to_absolute`, `translate`, `scale`, `rotate(angle=)`
and `interpolate(t=)`. On `.bbox`: `match_detections(threshold=)`.

`match_detections`' `strategy` is literal-only — it has one accepted value,
`"greedy"`, so there is nothing to vary.

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

Rings are implicitly closed — do not repeat the first point. A ring is a hole
because it sits in `holes`, not because of how it is wound: every operation is
winding-independent, so `flip()` and `ensure_winding()` change what
`winding()` reports without changing the region the contour describes.
`is_closed` is reserved and never read.

---

## Contours

The `.contour` namespace operates on polygon columns.

### One contour or a whole set

Every `.contour` accessor reads **either arity**: a `CONTOUR_SCHEMA` struct per
row, or the `CONTOUR_SET_SCHEMA` list of them that `extract_contours()`
produces. The result is wrapped to match the input, so the same call answers for
both:

```python
one = pl.col("contour").contour.area()   # Struct column  -> Float64
many = pl.col("contours").contour.area() # List(Struct)    -> List(Float64), one per contour
```

That is what lets the namespace read the column its own pipeline produced —
`extract_contours()` sinks a set, and every accessor takes one.

Two-operand accessors (`iou`, `dice`, `hausdorff_distance`) **broadcast**: a set
on one side and a single contour on the other gives one result per contour,
whichever side the set is on. A set on *both* sides raises rather than guessing,
because it could mean the N×M matrix (`pairwise_iou`) or an index-wise pairing
(`.explode()` one side), and those are different answers.

The set-level accessors (`pairwise_iou`, `match_detections`, `label_reduce`) run
the rule backwards: a lone contour is read as a set of one.

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

The column may hold **one contour per row** (`CONTOUR_SCHEMA`) or a **whole set**
(`List(CONTOUR_SCHEMA)`) — the source reads both, and a set paints the *union* of
its members: each member's exterior minus its own holes. One contour's hole never
erases another's fill, and the mask does not depend on the order of the set.

That is what closes the loop, because `extract_contours()` sinks a contour set:

```python
contours = (
    pl.col("image")
    .cv.pipe(Pipeline().source("image_bytes").grayscale().threshold(128).extract_contours())
    .sink("native")                      # List(CONTOUR_SCHEMA), one set per row
)

mask = pl.col("contours").cv.pipe(Pipeline().source("contour", width=200, height=200))
```

The trip back is lossy in one known direction: `extract_contours()` traces the
*centres* of the boundary pixels, so a region filling `w x h` pixels returns
bounding `(w-1) x (h-1)` and re-rasterizing erodes it by a pixel per round trip.

Staying inside one pipeline — `extract_contours().rasterize(...)` — produces the
same mask as sinking the set and reading it back through `source("contour")`.

`fill_value` and `background` may be inverted (`fill_value=0, background=255`);
the same region is painted either way.

Infer dimensions from an existing image:

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").resize(height=200, width=200))
mask = pl.col("contour").cv.pipe(Pipeline().source("contour", shape=img))
```

`shape=` takes a **reference pipeline** (a `LazyPipelineExpr`) instead of literal
dimensions: the referenced pipeline's output `[H, W]` is resolved and used to size
the raster canvas, so the mask matches the source image without hard-coding its
size. The same `shape=` reference is accepted by `rasterize(...)` directly when
you already have a contour-domain pipeline — which is what `extract_contours()`
leaves you with:

```python
mask = (
    pl.col("image")
    .cv.pipe(Pipeline().source("image_bytes").grayscale().threshold(128).extract_contours())
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
