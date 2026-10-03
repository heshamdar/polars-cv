# Migrating from 0.30 to 0.31

0.31 is mostly additions — open polylines, boundary distances, coverage
matching, header metadata from paths. The breaking changes are a removed
spelling, one default that now reports "unknown" instead of a guessed value,
and inputs that used to be silently mis-measured and are now read as written
or refused. This page lists what to check coming from 0.30; the
[changelog](../changelog.md) has the full list.

## Removed

- `extract_contours(mode="tree")`: it ran exactly as `mode="all"` while
  documented as a hierarchy, which nothing builds. Use `mode="all"`; `"tree"`
  is refused like any unknown value.

## Results that change

**`froc_auc`'s default no longer extends the curve.** Extension is still
available, opt-in: `froc_auc(..., extrapolate="flat")` gives the 0.30 value
(the LUNA16 `np.interp` convention). A FROC curve stops at the highest
FP/image any threshold reaches; `froc_auc` always filled it flat past that
point, while `froc_summary_table` reported null there — and for
thresholded-mask detections, which sit at one operating point, that gave an
AUC close to the curve's maximum sensitivity. Now every FROC/LROC curve
reader (`froc_auc`, `froc_sensitivity_at_fp`, `froc_summary_table`,
`lroc_auc`, `lroc_sensitivity_at_fpf` and the bootstrap CIs) takes the same
`extrapolate=` option, `"none"` (null off the curve) by default. Report
`froc_operating_range(table)` alongside to show how far each curve reaches.
LROC values are unchanged (its curve always spans [0, 1]).

**`is_closed=False` is read.** The field was written `True` and ignored, so a
contour with `is_closed=False` was measured as the ring its closing edge would
make. It is now an open polyline: boundary functions (perimeter, distances,
nearest point, Hausdorff, transforms) measure it as a line, and region
functions (`area`, `centroid`, `iou`, `contains_point`, `rasterize`, …)
refuse it, naming the row. Contours with `is_closed` true, null or absent
are unaffected. If you wrote `False` meaning a region, write `True` — or
close the line along the image frame with `.contour.close_along_border`.

**`CORRESPONDENCE_SCHEMA` has a third field**, `duplicate: List(Boolean)`.
Code that compares the struct's dtype to a hand-written two-field struct, or
unnests it into a fixed set of columns, needs the new field.

## Newly refused

These used to produce a silently wrong result:

- **`PreMatchedAdapter`** requires `class_col` when the detections or
  `image_meta` carry a `class_id` column (the class key was dropped and every
  join matched nothing), and raises on a named `iou_col`, `weight_col`,
  `det_idx_col`, `n_gts_col`, `gt_label_col` or `group_col` that is missing
  (it fell back to a default). Without `iou_col` the `iou` column is null and
  `at_iou_threshold` / `mean_average_precision` raise.
- **`ContourMatcher(score_col=...)` with a heatmap prediction** is refused;
  it was accepted and ignored. `score_col` now scores contour predictions.
- **A `file://` URL naming a host** (`file://server/data/a.png`) is refused by
  every path reader. It named no local file, and under `allowed_roots` the
  check and the read resolved it differently, so it could read outside the
  allowed roots. Write `file:///absolute/path`. A percent-encoded `file://`
  URL is now decoded (`%20` reads as a space).
- **`close_along_border`** refuses a NaN or negative `max_snap` and a
  non-finite frame size; **`point_from_coords` / `contour_from_coords`** refuse
  a NaN or infinite coordinate (every geometry reader already did).

## Behaviour worth knowing

- **A sourceless `Pipeline()` continues any domain.** Its first op now
  anchors it, so `contours.pipe(Pipeline().simplify(2.0))` works as
  `contours.simplify(2.0)` does.
- **A one-row geometry operand broadcasts**, as a per-row parameter does: a
  bbox or point built from literals no longer fails with "row 1 out of
  bounds".
- **`.cv.width()`/`height()`/`channels()`/`image_dtype()` take path columns**,
  reading a local file only as far as its header; `.cv.read_bytes()` first is
  no longer needed for metadata.
