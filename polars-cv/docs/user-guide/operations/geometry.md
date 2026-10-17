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
`largest(k=)`, `close_along_border` (every argument), `to_coords(order=)`,
`boundary_distances(sample_step=)`, `label_reduce(reduction=, region_mode=)`,
`correspond(threshold=)` and `correspond_by_coverage(tolerance=, threshold=,
sample_step=)`. On `.point`: `normalize`, `to_absolute`, `translate`, `scale`,
`rotate(angle=)`, `interpolate(t=)` and `to_coords(order=)`. On `.bbox`:
`correspond(threshold=)`.

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

### Invalid rows

A row whose data a function refuses — a line too far from the frame for
`close_along_border`, an open contour given to `area`, a zero `normalize`
size — fails the query by default, naming the row. `on_error("null")` nulls
just those rows, so they can be counted and inspected, mirroring
`source(on_error="null")`:

```python
closed = pl.col("line").contour.on_error("null").close_along_border(pl.col("w"), pl.col("h"))
```

An error about the **column** rather than a row — a contour set where one
contour per row is expected, a dtype no reader understands — still raises
under `"null"`: no row could succeed. `on_error` and `on_null` compose.

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

### Open polylines

`is_closed=False` makes a contour an **open polyline** — an annotated skin
line or muscle edge — whose edges join consecutive points only. It has no
region, so the functions split by what they read:

- **Boundary** functions measure it as a line: `perimeter` (its arc length),
  `hausdorff_distance`, `boundary_distances`, `bounding_box`, `simplify`,
  `smooth`, `densify`, `convex_hull`, the point-wise transforms (`translate`, `scale`, `flip`,
  `normalize`, `to_absolute`), and `.point.distance_to_contour` /
  `nearest_point_on_contour`.
- **Region** functions refuse it, naming the row: `area`, `centroid`,
  `largest`, `is_convex`, `winding`, `ensure_winding`, `contains_point`,
  `iou`, `dice`, `pairwise_iou`, `correspond`, `label_reduce`, `boundary`,
  `offset`, `union_all`, the set operations (`union`, `intersection`,
  `difference`, `symmetric_difference`) and `.point.signed_distance_to_contour`.
  `scale(origin="centroid")` refuses one too.
- `rasterize` refuses it too; `stroke()` it first (see [Drawing lines](#drawing-lines)).

Pipelines carry lines the same way: `source("contour")` reads open rows, and
each contour op behaves as its `.contour` accessor does (one definition serves
both), so a pipeline region op refuses a line with the same message.

`boundary()` turns a region's rings into lines (the exterior, then each hole,
each returning to its first point), so the result is a set; `close()` joins a
line's ends into the region it bounds, undoing `boundary()` for a region
without holes:

```python
# A region's outline as lines, and their lengths
lines = Pipeline().source("contour").boundary()
lengths = Pipeline().source("contour").boundary().perimeter()

# An annotated polyline as the region it encloses
area = pl.col("polyline").contour.close().contour.area()
```

An open contour may not have holes. An unspecified `is_closed` (a null, or a
struct without the field) reads as closed.

A line whose ends lie on the image frame closes into the region it bounds
along the frame, through the corners it passes:

```python
# A pectoral-muscle edge from the top edge to the left edge -> its corner region
region = pl.col("edge").contour.close_along_border(pl.col("w"), pl.col("h"))
```

`arc=` picks the way round (`"shortest"`, `"clockwise"`,
`"counterclockwise"`, as displayed); an end farther than `max_snap` (default
2 px) from the frame is refused rather than joined.

### From coordinate lists, and back

```python
from polars_cv.geometry import point_from_coords, contour_from_coords

df.with_columns(
    pt=point_from_coords(pl.col("yx"), order="yx"),        # [row, col] pairs
    line=contour_from_coords(pl.col("pts"), closed=False),  # an open polyline
)
df.select(pl.col("line").contour.to_coords(order="xy"))    # List(Array(f64, 2))
```

Integers and `Array(_, 2)` pairs are accepted; a pair that is not two
non-null, finite numbers is an error naming its row. `contour_set_from_coords`
builds a set; `.contour.to_coords()` refuses a contour with holes.

Boxes come from four numbers per row, in a layout you name — the three are
indistinguishable from the data, so `format=` is required:

```python
from polars_cv.geometry import bbox_from_coords

df.with_columns(
    box=bbox_from_coords("xyxy_list", format="xyxy"),            # one column of 4
    coco=bbox_from_coords(["x", "y", "w", "h"], format="xywh"),  # four columns
    yolo=bbox_from_coords("cxcywh", format="cxcywh"),
)
```

The result is `BBOX_SCHEMA` (`x, y, width, height`); a null box gives null.

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

The set-level accessors (`pairwise_iou`, `correspond`,
`correspond_by_coverage`, `label_reduce`, `largest`, `set_boundary_distances`,
`single`) run the rule backwards: a lone contour is read as a set of one.

`single(label=)` takes each row's one contour from a set, for a column where
every row should hold exactly one (the outline of an instance mask, say). A set
of any other size fails the query, quoting that row's `label` rather than a
row number, which under the streaming engine is only a position in one batch:

```python
outline = pl.col("contours").contour.single(label=pl.col("image_id"))
# ComputeError: the plugin failed with message:
#   contour_single: 'img7' holds 2 contours, not one
```

### Measurements

```python
df.with_columns(
    area=pl.col("contour").contour.area(),
    perimeter=pl.col("contour").contour.perimeter(),
    centroid=pl.col("contour").contour.centroid(),
    bbox=pl.col("contour").contour.bounding_box(),
)
```

### Boundary distances

`hausdorff_distance` is vertex-to-vertex, so two tracings of one outline at
different vertex densities come out apart. `boundary_distances` measures each
boundary's samples to the other's *edges*, both directions:

```python
d = pl.col("pred").contour.boundary_distances(pl.col("gt"), sample_step=0.5)
# Struct{mean_a_to_b, mean_b_to_a, assd, hd, hd95}
```

`assd` is the average symmetric surface distance, `hd` the Hausdorff
distance and `hd95` the larger directed 95th percentile (MONAI's
convention). For physical units, `.contour.scale(..., origin="origin")` both
sides by the pixel spacing first.

A region cut off by the image edge has a frame segment in its outline that no
annotation traces. Pass the image as `frame=` (a bbox per row) and boundary on
or outside it is not measured, in either direction. Every remaining sample is
still measured to the other side's *whole* outline, so an annotation drawn a
pixel inside the edge is a pixel away rather than measured across the region.
Inset the bbox to also skip boundary near the edge:

```python
frame = pl.struct(  # a BBOX_SCHEMA struct: Float64 fields
    x=pl.lit(0.0), y=pl.lit(0.0), width=pl.col("w").cast(pl.Float64), height=pl.col("h").cast(pl.Float64)
)
d = pl.col("pred").contour.boundary_distances(pl.col("gt"), frame=frame)
```

For masks with several regions, `.contour.set_boundary_distances(other)` reads
each side's contour set as **one** boundary, the union of its outlines (the
surface distance MONAI computes on masks). Every sample is measured to the
nearest edge of any region on the other side. `boundary_distances` instead
gives one result per contour of a set. An empty set (an empty mask) has no
boundary and gives null. `sample_step` and `frame` work as above.

### Keeping the largest

`largest(k=1)` keeps the `k` largest contours of a set by area, largest first
(equal areas in input order) — as `.contour.largest(k)` and as a pipeline op
after `extract_contours()`. The result is always a set. To drop specks
relative to each image's size rather than in pixels, filter at extraction with
`extract_contours(min_area_fraction=...)` (a fraction of height x width, in
(0, 1]; with `min_area` too, a contour must pass both):

```python
pipe = (
    Pipeline().source("image_bytes").grayscale().threshold(128)
    .extract_contours(min_area_fraction=0.001)
    .largest(k=3)
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

`scale` takes `origin=` — `"centroid"` (the default), `"bbox_center"` or
`"origin"`. `.contour.scale` and `Pipeline.scale_contour` are one operation
with one definition, so they always agree.

### Shaping

Five more ops, each both a pipeline op and a `.contour` accessor with one
definition (`stroke`, the fifth, is under [Drawing lines](#drawing-lines)):

- `offset(distance, join="round")` grows a set's regions by `distance` pixels,
  or shrinks them for a negative one: every point within `distance` of their
  union (or inside it and at least `-distance` from its edge). The set is read
  as one union, so growing can merge regions and shrinking can split them or
  leave none — the result is always a set. Convex corners turn by `join`
  (`"round"`, `"miter"` — cut off past 4 times the distance, SVG's default
  limit — or `"bevel"`); concave ones stay sharp. Round arcs are polygons
  within 1/64 pixel of the true arc up to a radius of about 32 pixels, and
  within about `r / 2000` beyond (0.12 pixel at 1000), the finest arc the
  polygon engine draws.
- `union_all()` merges a set's regions into disjoint ones (a hole another
  member covers is filled) — always a set.
- `smooth(iterations=1)` rounds corners by Chaikin's corner cutting: each pass
  replaces every edge by its points a quarter and three quarters along. A line
  keeps its ends (one returning to its first point is smoothed through it).
- `densify(max_segment_length)` splits every edge into
  `ceil(length / max_segment_length)` equal parts, keeping every point.

`offset` and `union_all` take regions; `smooth` and `densify` take lines too.

```python
# A 3 px halo round each blob, as one mask
halo = (
    Pipeline().source("image_bytes").grayscale().threshold(128)
    .extract_contours().offset(3.0).rasterize(width=200, height=200)
)
# Soften a traced outline's pixel staircase
soft = pl.col("contours").contour.smooth(iterations=3)
```

### Combining two sets

`union`, `intersection`, `difference` and `symmetric_difference` combine two
contour sets: each side is read as the union of its regions, and the result is
disjoint regions — always a set, since combining shapes can split or merge
them. As `.contour` accessors the other side is a contour column (a single
contour is a set of one; a null side nulls the row); in a pipeline it is
another contour-domain expression, and the result chains on:

```python
# Prediction regions outside the ground truth, per row
false_positive = pl.col("pred").contour.difference(pl.col("truth"))

# The same in a pipeline, measured
pred = pl.col("pred").cv.pipe(Pipeline().source("contour"))
truth = pl.col("truth").cv.pipe(Pipeline().source("contour"))
missed_area = truth.difference(pred).area().sink("native")
```

Rasterizing a result paints exactly the pixelwise `|`, `&`, `& ~` or `^` of
rasterizing the two sides.

### Rasterization

Convert contours to binary masks:

```python
pipe = Pipeline().source("contour").rasterize(width=200, height=200)

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

mask = pl.col("contours").cv.pipe(Pipeline().source("contour").rasterize(width=200, height=200))
```

The trip back is lossless: `extract_contours()` traces the *edges* of the
boundary pixels, so a region filling `w x h` pixels returns bounding `w x h`,
its area is its pixel count, and re-rasterizing gives back the same mask.

Staying inside one pipeline — `extract_contours().rasterize(...)` — produces the
same mask as sinking the set and reading it back through `source("contour")`.

`fill_value` and `background` may be inverted (`fill_value=0, background=255`);
the same region is painted either way.

By default a pixel is filled when its centre lies inside, so edges are hard.
`supersample=n` anti-aliases them: each pixel samples its `n x n` sub-pixel
centres and is painted `background + (fill_value - background) * k / n**2` for
the `k` that fall inside (rounded half away from zero), from 1 (the default) up
to 16.

```python
# Smooth edges: 4 x 4 samples per pixel
mask = pl.col("contour").cv.pipe(
    Pipeline().source("contour").rasterize(width=200, height=200, supersample=4)
)
```

### Drawing lines

`rasterize` fills regions. A line has no region, so `stroke(width=1.0,
cap="round", join="round")` first turns paths into the regions they cover:
every point within `width / 2` of a path, its ends shaped by `cap`
(`"round"`, the default; `"square"`; `"butt"`, flat at the end point) and its
corners by `join` (`"round"`, `"miter"`, `"bevel"`). Every member of the set
is read as its outline: an open line is capped at its ends, while a region's
rings — and a line that returns exactly to its first point — are closed
paths with no ends. The result is a set of disjoint regions, a pipeline op and
a `.contour` accessor alike, and `width`, `cap` and `join` each accept an
expression. See [Drawing with polars-cv](../drawing.md) for building pictures
from these.

A 1-pixel stroke on integer coordinates puts its edges exactly on pixel
centres, so which side of each edge a pixel lands on is decided by the
scanline's tie rule; use `supersample` (or a non-integer width) for an even
line.

```python
# A region's outline, and its fill
outline = (
    Pipeline().source("contour")
    .stroke(width=2.5, join="miter")
    .rasterize(width=200, height=200, supersample=4)
)
filled = Pipeline().source("contour").rasterize(width=200, height=200)
```

Infer dimensions from an existing image:

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").resize(height=200, width=200))
mask = pl.col("contour").cv.pipe(Pipeline().source("contour").rasterize(shape=img))
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

The `.point` namespace operates on point columns: one point per row
(`POINT_SCHEMA`) or a point set (`POINT_SET_SCHEMA`, `List(point)`). Over a set
every method gives one value per point, in input order, as a `List` — so a set
of points is measured against the row's contour without an explode/group-by:

```python
pl.col("pts").point.distance_to_contour(pl.col("contour"))  # List(Float64)
```

A contour or bbox operand broadcasts against the set; two point columns
broadcast either way (a set against a single point), and a set on both sides is
refused. A null point in a set gives a null in its place.

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

### Correspondence

Greedy one-to-one pairing between two bbox sets, by overlap. Boxes are visited
in `order` and each takes the highest-IoU partner not already taken:

```python
df.with_columns(
    pairs=pl.col("pred_bboxes").bbox.correspond(
        pl.col("gt_bboxes"),
        threshold=0.5,
        # Visit highest-confidence first. `correspond` only sees overlap, so
        # what "confidence" means -- and that it should decide the order -- is
        # yours to say. Omit `order` to visit in natural order.
        order=pl.col("pred_scores").list.eval(
            pl.element().rank(method="ordinal", descending=True).arg_sort()
        ),
    ),
)
```

The result is a struct matching `CORRESPONDENCE_SCHEMA`: `right_idx` (the
partner's index, null where unpaired), `overlap` (its IoU) and `duplicate`
(left unpaired although it cleared the threshold against a partner another
element had already claimed — a repeated hit, which LUNA16/CAMELYON-style
evaluation ignores), all positionally aligned with the left column. Counting
how many pairings your dataset contains is a question about your dataset, so
`correspond` does not answer it.

`.contour.correspond(...)` is the same rule over contour IoU, and
`.contour.correspond_by_coverage(other, tolerance, ...)` pairs by *coverage* —
the fraction of each target's boundary inside the candidate or within
`tolerance` of its edges — which scores line-shaped targets (polylines) that
IoU cannot.

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
