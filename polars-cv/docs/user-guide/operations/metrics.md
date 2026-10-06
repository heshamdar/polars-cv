# Evaluation Metrics

polars-cv evaluates detection, instance-segmentation, heatmap (FROC) and
semantic-segmentation models with Polars lazy expressions. There is a one-call
entry point for each setup. Every number it reports can be traced to the
building blocks below it, so you can drop down a layer whenever you need
something the one-call form does not offer.

## Evaluating a model

### Object detection (boxes)

Predictions and ground truth as they usually come: one row per object.

```python
import polars as pl
from polars_cv.metrics import evaluate_detections

preds = pl.read_csv("predictions.csv")  # image_id, class_id, x1, y1, x2, y2, score
gts = pl.read_csv("annotations.csv")    # image_id, class_id, x1, y1, x2, y2

report = evaluate_detections(
    preds, gts, geometry=("x1", "y1", "x2", "y2"), box_format="xyxy"
)
report.summary       # map, map_50, map_75, mar
report.per_class     # ap, ap_50, ap_75, recall, n_gts, n_preds per class
report.ci("map")     # bootstrap interval, images resampled
report.matches(0.5)  # every prediction row with is_tp / iou / matched_gt_row
```

The geometry can also be a single column. `box_format` says what four numbers
mean (`"xyxy"`, `"xywh"` as in COCO, or `"cxcywh"` as in YOLO); it is required,
because the data cannot tell them apart. A `BBOX_SCHEMA` struct column needs no
format.

**The defaults are COCO's.** `iou_thresholds="coco"` means:

- predictions are matched again at each IoU threshold from 0.50 to 0.95 in
  steps of 0.05, so a detection that loses its box at one threshold can win it
  at another;
- AP is 101-point;
- each image keeps its 100 highest-scoring predictions per class;
- a class with no ground truth is left out of the means.

On data without crowd regions, the results match pycocotools' `COCOeval` to
1e-12. Area ranges (small/medium/large) and crowd/ignore regions are not
modelled. For Pascal VOC, pass `iou_thresholds=0.5`, which gives all-points AP
at one threshold.

Images that appear in neither table still count, for example empty images whose
predictions are all false positives. Pass the full image list as `images=`. It
can be a frame carrying per-image `weight=` and `group=` columns.

### Instance segmentation (polygons or masks)

Use the same call, with a `CONTOUR_SCHEMA` polygon column or one binary mask
per object instead of boxes. Matching is then by region IoU:

```python
report = evaluate_detections(preds, gts, geometry="polygon")
report = evaluate_detections(preds, gts, geometry="mask")  # one region per mask
```

A mask holding several separate regions fails the query and names its image.
Silently picking one region would score a different object. Split such masks
into one row per region, or pass polygons.

### Heatmaps and lesion detection (FROC)

For a model that outputs a probability map, pass one row per image:

```python
from polars_cv.metrics import evaluate_heatmaps

report = evaluate_heatmaps(df, heatmap="prob_map", gt="lesion_mask")
report.summary  # map, mar, sensitivity@0.125 ... sensitivity@8, cpm
report.ci("cpm")
```

Each heatmap is thresholded into candidate regions, and each region is scored
from the heatmap and matched to the mask's regions. Any `ContourMatcher` option
can be passed through, for example `extraction_threshold=0.3`,
`match_by="coverage"` for line-shaped targets, or `score_reduction="mean"`.
`extrapolate="flat"` reads the FROC curve the way LUNA16 does.

### Semantic segmentation (Dice, IoU, Hausdorff)

```python
from polars_cv.metrics import evaluate_segmentation, segmentation_measures

report = evaluate_segmentation(df, pred="pred_mask", target="gt_mask")
report.summary     # mean dice, iou, assd, hd, hd95 (+ undefined counts)
report.per_image
report.ci("hd95")

# Or per row, inside your own query:
df.with_columns(m=segmentation_measures("pred_mask", "gt_mask", spacing=(0.8, 0.8)))
```

- **Boundaries.** These are the masks' exact pixel-edge outlines, holes
  included, measured point-to-edge.
- **The image edge.** With `frame="image"` (the default), boundary on the
  image edge is not measured: a structure cut off by the field of view has an
  edge there that is not anatomy.
- **Units.** `spacing=(row, col)` gives distances in physical units.
- **Empty masks.** Two empty masks agree perfectly (Dice 1). If one mask is
  empty, Dice is 0 and the boundary distances are null; the summary counts
  those rows as undefined.

## How it fits together

```
object tables ──► match_detections ──┐
heatmaps/masks ─► ContourMatcher ────┼─► DetectionTable ─► Statistic ─► value / by_group / bootstrap_ci
pre-matched ────► PreMatchedAdapter ─┘                                    ▲
                                                     DetectionReport ─────┘
```

1. **Matching** turns predictions and ground truth into a `DetectionTable`,
   with one row per detection plus per-(image, class) metadata.
   `match_detections` is the general entry point for object tables. The
   matchers below read one row per (image, class) holding lists.
2. **Statistics.** Every metric is a `Statistic`:
   - `AP`, `Recall`, `PrecisionAt`, `FROCSensitivity`, `CPM`, `FROCAUC`,
     `LROCAUC`, …
   - `MeanOver(statistic)` averages one over classes and IoU thresholds;
     `mean_ap()` is mAP.

   A statistic reads a table as a whole (`.value`), per group (`.by_group`) or
   per bootstrap replicate (`bootstrap_ci`). It is the same estimator every
   time.
3. **Reports** pick statistics and show them. `report.metrics` lists which
   statistic each number came from.

So a custom number (say AP@0.6 per scanner, with an interval) is one line on
the report's table:

```python
from polars_cv.metrics import AP, bootstrap_ci

bootstrap_ci(report.table.at_iou_threshold(0.6), AP(), group_by="group_id").collect()
```

The function forms (`average_precision`, `froc_auc`, …) are the ungrouped
readings of the same statistics.

## DetectionTable

The `DetectionTable` is the canonical intermediate representation. It wraps two
aligned lazy frames:

- **detections** — one row per detection with `image_id`, `class_id`, `score`,
  `is_tp`, `gt_idx`, `iou`, `det_idx`.
- **image_metadata** — one row per (image, class) with `n_gts`, `weight`,
  `gt_label`.

`weight` (default `1.0`) weights every metric: PR/AP/mAP, the threshold
metrics, the confusion counts' `weighted_*` fields, FROC/LROC and their
bootstrap intervals. Use it to reweight a study to a target vendor or prevalence
mix; see [Weighted tables](#weighted-tables) for how the intervals treat
weights estimated from the sample.

## Matchers

### PreMatchedAdapter

For data that already has per-detection TP/FP assignments:

```python
from polars_cv.metrics import PreMatchedAdapter, precision_recall_curve

adapter = PreMatchedAdapter()
table = adapter.match(
    data,
    pred_col="confidence",
    gt_col="is_tp",
    image_id_col="image_id",
    # image_population has one row per evaluated image, with columns
    # image_id and n_gts (plus optional class_id / weight / gt_label).
    image_meta=image_population,
)
result = precision_recall_curve(table)
```

Pass `image_meta` whenever any image may carry zero detections. Without it the
adapter derives the population by grouping the detection frame, so an image the
detector found nothing in has no metadata row at all — which deletes the
negative population and inflates recall and FP-per-image. Omitting it emits a
`UserWarning`.

Classes are keyed only when you say so: if the detections or `image_meta`
carry a `class_id` column, pass `class_col` (e.g. `class_col="class_id"`) —
the adapter raises otherwise, since the metrics join detections to the
population on `(image_id, class_id)` and a mismatched key silently counts for
nothing. Pass `iou_col` if you want to re-threshold by IoU
(`table.at_iou_threshold`, `mean_average_precision(iou_thresholds=...)`);
without one the table has no IoU and those raise.

`image_meta` is the sole source of image metadata, so it cannot be combined
with `n_gts_col`, `weight_col`, `gt_label_col` or `group_col` — those describe
how to derive metadata from the *detection* frame, and passing both raises
rather than silently ignoring them.

### ContourMatcher

For heatmap + binary mask inputs (used by FROC/LROC workflows):

```python
from polars_cv.metrics import ContourMatcher, froc_auc

matcher = ContourMatcher(iou_threshold=0.5, extraction_threshold=0.1)
table = matcher.match(data, pred_col="heatmap", gt_col="gt_mask")
auc = froc_auc(table, fp_range=(0.0, 8.0)).collect().item()
```

`ContourMatcher.match` also accepts a pre-decoded `LazyPipelineExpr` for
`pred_col`/`gt_col`, so a segmentation graph and the contour extraction can
share one decode and stream from a single collect.

Each detection is scored by the heatmap pixels in its region —
`score_reduction=` (`"max"` by default, `"mean"` or `"sum"`) over
`score_region_mode=` (`"interior"`, `"boundary"` or `"bbox"`). A heatmap that
saturates scores most detections at its peak, so under `"max"` they tie and
the FROC/LROC curve collapses to a few points; `"mean"` separates them.

`iou_threshold`, `extraction_threshold`, `min_contour_area`,
`min_contour_area_fraction` (specks relative to each image's size) and
`coverage_tolerance` each take a Polars expression, read per row — e.g. a
physical tolerance across devices:

```python
matcher = ContourMatcher(
    match_by="coverage",
    coverage_tolerance=5.0 / pl.col("spacing_mm"),
    min_contour_area_fraction=1e-4,
)
```

A literal is checked when the matcher is built; an expression when its row
runs.

#### Contour predictions

A model that emits polygons with its own per-object scores skips the heatmap:
pass a contour or contour-set `pred_col` with `score_col` — `List[float]`
aligned with a set, or a float for one contour per row — and
`auto_resize=False` (the contours are in the GT's coordinates). The contours
are the detections as they are, a 0.0 score included. A score count that does
not match the contours, or a null score, fails the query rather than dropping
detections.

```python
table = ContourMatcher(auto_resize=False).match(
    data, pred_col="pred_polygons", gt_col="gt_mask", score_col="pred_scores"
)
```

#### Line-shaped ground truth

Landmarks annotated as lines (a skinfold, a muscle edge) but predicted as thin
regions cannot be paired by IoU: a line has no area. Match them by
**coverage** instead — the fraction of each GT line's samples inside a
prediction or within `coverage_tolerance` pixels of it — and give the GT as a
contour column (open polylines, e.g. from `contour_set_from_coords(...,
closed=False)`) rather than a mask:

```python
matcher = ContourMatcher(
    iou_threshold=0.5,          # here: the minimum coverage
    match_by="coverage",
    coverage_tolerance=5.0,     # pixels; divide a physical tolerance by the spacing
    duplicates="ignore",        # a repeated hit on one GT is dropped, not an FP
    auto_resize=False,          # contour GT has no mask size to resize to
)
table = matcher.match(data, pred_col="heatmap", gt_col="gt_lines")
```

`duplicates="ignore"` follows the LUNA16/CAMELYON convention; the default,
`"false_positive"`, counts a second prediction on an already-matched GT as a
false positive. At the expression level the same pieces are
`.contour.correspond_by_coverage(...)` and the `duplicate` field of every
correspondence result.

Both columns go through `source("auto")`, so a mask may be a nested
`List`/`Array` of numbers or booleans, encoded image bytes (PNG/JPEG or a VIEW
blob), or a `String` column of paths to read.

### BBoxMatcher

For bounding-box detection inputs:

```python
from polars_cv.metrics import BBoxMatcher, precision_recall_curve

matcher = BBoxMatcher(iou_threshold=0.5)
table = matcher.match(
    data,
    pred_col="pred_bboxes",
    gt_col="gt_bboxes",
    score_col="pred_scores",
)
result = precision_recall_curve(table)
```

## Available Metrics

### Precision-Recall

```python
from polars_cv.metrics import (
    precision_recall_curve,
    average_precision,
    mean_average_precision,
    precision_at_threshold,
    recall_at_threshold,
    f1_at_threshold,
)

pr = precision_recall_curve(table)
ap = average_precision(table)
map_val = mean_average_precision(table, iou_thresholds=[0.5, 0.55, 0.6, ..., 0.95])
```

Every precision-recall metric is weighted by `image_metadata.weight`, like
FROC and LROC. Each detection carries its image's weight, precision is
`Σw·tp / Σw·(tp + fp)`, and recall is `Σw·tp / Σw·n_gts`. This matches
scikit-learn's `sample_weight`: a weight of `k` counts the image as if it
appeared `k` times. Unit weights give the plain counts, a zero weight removes an
image, and `weight_agg=` resolves duplicate keys as it does for FROC.

### FROC

The FROC metrics are **expression-valued and lazy**: `froc_auc` returns a
`LazyFrame` (one row per group), so a scalar is `.collect().item()` and grouping
is a normal `group_by` rather than a Python loop.

```python
from polars_cv.metrics import (
    froc_auc,
    froc_curve_lazy,
    froc_operating_range,
    froc_sensitivity_at_fp,
    froc_summary_table,
)

# Trapezoidal FROC AUC requires an explicit FP window (the FP/image axis is
# unbounded); the default correction="normalize" returns mean sensitivity over
# it. Pass correction=None for the raw partial area.
print(froc_auc(table, fp_range=(0, 8)).collect().item())
print(froc_sensitivity_at_fp(table, 1.0).collect()["sensitivity"].item())
print(froc_summary_table(table).collect())

# How far the curve reaches (max FP/image and the sensitivity there):
print(froc_operating_range(table).collect())

# One AUC per class, in a single lazy plan:
per_class = froc_auc(table, group_by="class_id", fp_range=(0, 8)).collect()

# Mann-Whitney AUC (detection- or image-level):
mw = froc_auc(table, method="mann_whitney", level="detection").collect().item()

# The full curve, when you need the operating points:
curve = froc_curve_lazy(table).collect()
```

A FROC curve stops at the highest FP/image any threshold reaches — detections
from a thresholded mask exist at a single operating point. Past it the
sensitivity was never observed, and every FROC reader says so the same way:
`froc_sensitivity_at_fp` and `froc_summary_table` give a null sensitivity
there, and `froc_auc` gives a null AUC for an `fp_range` that reaches beyond
it. Report `froc_operating_range` alongside them so a truncated curve is
visible. If you want the common convention of extending the last sensitivity
flat (LUNA16's `np.interp`), ask for it: every one of these functions — and
`froc_auc_ci_lazy` — takes `extrapolate="flat"` (default `"none"`). A
bootstrap interval is null when any replicate's curve stops short of the
window, rather than scoring it 0. Where an x is visited more than once, the
highest sensitivity there is returned.

### LROC

```python
from polars_cv.metrics import lroc_auc, lroc_curve_lazy, lroc_sensitivity_at_fpf

# LROC's FPF axis is bounded to [0, 1], so no range is needed — the default
# integrates the full [0, 1] domain (normalized, i.e. the standard LROC AUC).
print(lroc_auc(table).collect().item())
print(lroc_sensitivity_at_fpf(table, 0.5).collect())
curve = lroc_curve_lazy(table).collect()
```

### Confusion Matrix

```python
from polars_cv.metrics import confusion_at_threshold

counts = confusion_at_threshold(table, threshold=0.5)
counts.tp, counts.fp, counts.fn  # raw counts
counts.weighted_tp, counts.weighted_fp, counts.weighted_fn  # weighted masses
counts.precision, counts.recall, counts.f1  # derived from the weighted masses
counts.to_dict()  # {'tp': 10, 'fp': 3, 'fn': 2}
```

## Bootstrap Confidence Intervals

`bootstrap_ci(table, statistic, ...)` gives an interval for any `Statistic`.
`froc_auc_ci_lazy`, `lroc_auc_ci_lazy` and `average_precision_ci_lazy` are
this function with `FROCAUC`, `LROCAUC` and `AP`. Images are the resampling
unit. The classes and IoU thresholds that a `MeanOver` statistic averages over
are drawn together: each drawn image brings all its rows. So an mAP interval
reflects one resample per replicate, not one per class.

```python
from polars_cv.metrics import CPM, bootstrap_ci, mean_ap

bootstrap_ci(table, mean_ap("101_point"), n_bootstrap=1000, seed=1).collect()
bootstrap_ci(table, CPM(extrapolate="flat"), group_by="group_id").collect()
```

The intervals are **fully lazy and group-aware**.
Each entry point returns a `pl.LazyFrame` and never collects internally — the
whole bootstrap (resample, per-replicate metric, and the percentile bounds) is
one Polars plan the *caller* collects. That means a CI can be built at plan time
with no data present and joined onto a point-metric frame, one `ci_lower` /
`ci_upper` row per group, instead of looping over groups in Python.

```python
from polars_cv.metrics import (
    average_precision_ci_lazy,
    froc_auc_ci_lazy,
    lroc_auc_ci_lazy,
)

# Ungrouped: one row [auc, ci_lower, ci_upper]. Trapezoidal FROC needs fp_range.
froc_auc_ci_lazy(table, fp_range=(0, 8), n_bootstrap=1000, seed=42).collect()

# Group-aware: one row per group, ready to join onto the point-metric frame.
ci = froc_auc_ci_lazy(
    table, group_by="group_id", fp_range=(0, 8), n_bootstrap=1000, seed=42
)
point = froc_auc(table, group_by="group_id", fp_range=(0, 8))
point.join(ci.select("group_id", "ci_lower", "ci_upper"), on="group_id").collect()

# Entity-level resampling (e.g. by case), composing with the grouping.
froc_auc_ci_lazy(
    table, group_by="group_id", fp_range=(0, 8), seed=42, sample_col="case_id"
)

lroc_auc_ci_lazy(table, n_bootstrap=1000, seed=42)
average_precision_ci_lazy(table, group_by="group_id", n_bootstrap=1000, seed=42)
```

The resample is a **position-independent hash** of each unit's global slot,
built collect-free by cross-joining a constant-length reps frame against the
units — so it never materializes the `n_bootstrap × n_units` frame and each group
resamples within itself, stratified by `gt_label`. Each image has exactly one
draw slot, however many class rows it has. It is drawn in the positive stratum
if any of its classes is positive, and a draw brings only its own group's rows,
so `group_by="class_id"` keeps every class's replicate the size of that class's
sample. Because the draw hashes its
own slot id (never a row position), a given `seed` reproduces the interval
**bit-for-bit regardless of thread count** (`POLARS_MAX_THREADS`) or streaming
morselization, and `seed=None` is deterministic (a fixed constant). The `auc` /
`ap` column is the deterministic point estimate; only the bounds are
bootstrapped. A **degenerate group** keeps its point estimate but reports null
`ci_lower` / `ci_upper` rather than raising, so a single plan spans viable and
degenerate groups alike. Viability needs at least one positive target — and, for
`method="mann_whitney"` (a two-class rank statistic), at least one negative too.

### Weighted tables

Resampling a weighted table has to respect where the weights came from, so all
three intervals (FROC, LROC and AP) take a `weight_scheme`:

| `weight_scheme` | The weights are | The draw |
|---|---|---|
| `"reestimate"` (default) | estimated from the sample: `p / q̂` over target distributions (conditional ones included), post-stratification, raking | each `(group, weight cell)` redrawn to its own size |
| `"stratified"` | as above, and each cell's positive count was fixed by the study design | each `(group, gt_label, weight cell)` redrawn to its own size |
| `"fixed"` | known in advance: design weights, a continuous weight | `gt_label`-stratified, weights carried unchanged |

**`"reestimate"`.** When the weights are estimated from the sample itself, the
correct weights differ in every replicate, because each replicate draws a
different mix. Units whose weights agree within `weight_rtol` form a **weight
cell**, and each `(group, cell)` is redrawn to its own size. Every weighted
statistic is a ratio that does not change when all weights are rescaled, so a
weight that depends only on its cell's count does not change under this draw:
the weights given are exactly the weights re-estimated in each replicate. No
target distributions or reweighting hook are needed.

`gt_label` is **not** crossed with the cells. When the weights were estimated
over all images, each cell's positive count is random, and the weighted
statistics depend on it (the positives' mix across cells follows each cell's
observed prevalence). Fixing it too, as `"stratified"` does, drops that
variance: in simulation the bootstrap SE came out about 10% low, and 95%
intervals covered 0.90-0.93. Targets conditioned on `gt_label` give each label
its own weights, so their cells already carry the label. A group with a single
cell (unit weights) keeps the `gt_label` stratum, so its draw is the unweighted
one.

```python
# Weights computed by the caller, e.g. per (group, vendor) cell:
meta = meta.with_columns(
    weight=pl.col("vendor").replace_strict(target)
    / (pl.len().over("group_id", "vendor") / pl.len().over("group_id"))
)
froc_auc_ci_lazy(table, group_by="group_id", method="mann_whitney", seed=42)

# Two cells that happen to share a weight (e.g. several at 1.0) are merged;
# name the columns the weights were computed over to keep them apart.
froc_auc_ci_lazy(table, group_by="group_id", strata="vendor", fp_range=(0, 8))

# Coarser weights (e.g. read back from a 3-significant-figure CSV):
average_precision_ci_lazy(table, group_by="group_id", weight_rtol=1e-3)

# Known design weights, or a continuous weight (e.g. from a propensity model):
lroc_auc_ci_lazy(table, group_by="group_id", weight_scheme="fixed")
```

- **Weights computed per group or globally** are both exact. Draws never cross a
  group, and fixing every `(group, cell)` count also fixes the global count.
- **`weight_rtol`** (default `1e-6`, relative) sets how close two weights must
  be to share a cell. The sorted distinct weights split wherever consecutive
  values differ by more than that, so there is no rounding boundary for
  near-equal values to fall either side of. The default absorbs arithmetic
  noise; `0.0` compares exactly. A tolerance coarse enough to merge different
  cells keeps their combined count fixed but not the split between them.
- **Entity-level resampling (`sample_col`)** redraws whole entities,
  stratified by the weight cells of their images. Entities differ in size, so
  a replicate's image mix can still drift. Each drawn image's weight is therefore
  rescaled by `(n_c/N) / (n*_c/N*)`, its cell's full-sample share of the
  group's images over its share in the replicate. That is the `p / q̂` weight
  re-estimated on the replicate, recomputed at the coarser level and assigned
  back to the images. For image-level draws the factor is exactly 1.
  `"stratified"` draws entities the same way: an entity has no single label.
- **A weight cell holding a single unit nulls its group's bounds.** That cell has
  no bootstrap variance. A continuous weight puts every unit in its own cell and
  would otherwise report a zero-width interval: use `"fixed"` for one. A cell
  whose weights are all zero (images outside the target) is exempt, since it
  contributes nothing to any statistic.
- **A replicate that draws no positive image** (or, for Mann-Whitney, no
  negative one) nulls its group's bounds. Its statistic still has a value, but
  it describes no resample of the group. Only a draw that does not fix the
  `gt_label` counts can make one — `"reestimate"` with several cells, or
  `sample_col` — and only in small groups: each replicate draws none with
  probability about `(1 − prevalence)^n`. For a very small study whose cell
  counts were fixed by design, `"stratified"` cannot.
- **`"fixed"`** forms no cells, so it raises if given `strata` or
  `weight_rtol`. It does not rescale under `sample_col` either.

## IoU thresholds: re-matching vs re-thresholding

A matcher given several IoU thresholds matches once per threshold, in one
plan, and returns one table with an `iou_threshold` column. This is what COCO
does: at 0.75, a detection that claimed a box at 0.5 with IoU 0.6 no longer
qualifies, and a lower-scoring detection with IoU 0.8 can take the box.

```python
table = BBoxMatcher(iou_threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]).match(...)
mean_average_precision(table)        # each threshold on its own matching
table.at_iou_threshold(0.75)         # one matching, exactly
AP().by_group(table, "iou_threshold")
```

Pooling thresholds would count every ground truth once per threshold, so a
swept table refuses any evaluation that does: `table.detections`,
`average_precision(table)`, and `froc_auc(table)` without
`group_by="iou_threshold"` all raise. Group by the threshold, average over it
(`MeanOver` / `mean_ap()`), or select one with `at_iou_threshold`.

A table matched at one threshold can still be **re-thresholded**.
`at_iou_threshold(t)` compares the stored IoU with `t`, which does not
re-match: a detection that lost its box to a higher-scoring duplicate does not
get it back.

## Stratified Evaluation

`DetectionTable.filter_images` keeps a subset of images — by id, or by a
predicate over `image_metadata` — with their detections and the stored
matcher settings, so the same metrics run per device, scan type or size
bucket without re-matching:

```python
table = matcher.match(data, pred_col="heatmap", gt_col="gt_mask", group_col="device")
for device in ["a", "b"]:
    sub = table.filter_images(pl.col("group_id") == device)
    print(device, froc_auc(sub, fp_range=(0.0, 8.0)).collect().item())

small = table.filter_images(["img_001", "img_007"])
```

## Class-Aware Metrics

Matcher input for several classes is one row per (image, class). Pass the
class column as `class_col`, and each detection keeps its own row's class.
`match_detections` builds those rows for you.

```python
table = BBoxMatcher().match(data, ..., class_col="category")
ap_cat = average_precision(table, class_id="cat")
map_val = mean_average_precision(table)        # classes without truth count as 0
mean_ap().value(table)                         # COCO: classes without truth left out
AP().by_group(table, "class_id").collect()     # every class in one plan
```
