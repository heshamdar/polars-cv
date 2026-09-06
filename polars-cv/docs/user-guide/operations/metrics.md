# Detection Metrics

polars-cv provides a comprehensive suite of detection metrics built on top of
Polars lazy expressions and the polars-cv matching primitives. All curve
computations use native Polars operations.

## Architecture

The metrics system follows a three-layer architecture:

```
Input Data → Matcher → DetectionTable → Metric Function → MetricResult
```

1. **Matchers** convert raw data into a canonical `DetectionTable`.
2. **Metric functions** compute curves and scalar metrics from the table.
3. **Result objects** carry computed curves with convenience methods.

## DetectionTable

The `DetectionTable` is the canonical intermediate representation. It wraps two
aligned lazy frames:

- **detections** — one row per detection with `image_id`, `class_id`, `score`,
  `is_tp`, `gt_idx`, `iou`, `det_idx`.
- **image_metadata** — one row per (image, class) with `n_gts`, `weight`,
  `gt_label`.

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
auc = froc_auc(table).collect().item()
```

`ContourMatcher.match` also accepts a pre-decoded `LazyPipelineExpr` for
`pred_col`/`gt_col`, so a segmentation graph and the contour extraction can
share one decode and stream from a single collect.

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

### FROC

The FROC metrics are **expression-valued and lazy**: `froc_auc` returns a
`LazyFrame` (one row per group), so a scalar is `.collect().item()` and grouping
is a normal `group_by` rather than a Python loop.

```python
from polars_cv.metrics import (
    froc_auc,
    froc_curve_lazy,
    froc_sensitivity_at_fp,
    froc_summary_table,
)

print(froc_auc(table, fp_range=(0, 8)).collect().item())
print(froc_sensitivity_at_fp(table, 1.0))
print(froc_summary_table(table))

# One AUC per class, in a single lazy plan:
per_class = froc_auc(table, group_by="class_id").collect()

# Mann-Whitney AUC (detection- or image-level):
mw = froc_auc(table, method="mann_whitney", level="detection").collect().item()

# The full curve, when you need the operating points:
curve = froc_curve_lazy(table).collect()
```

`froc_sensitivity_at_fp` returns `None` — and `froc_summary_table` a null —
when the requested FP/image rate lies beyond the curve's observed range. An
operating point the detector never reaches is reported as unreachable rather
than clamped to the last value on the curve. Where an x is visited more than
once, the highest sensitivity there is returned.

### LROC

```python
from polars_cv.metrics import lroc_auc, lroc_curve_lazy, lroc_sensitivity_at_fpf

print(lroc_auc(table).collect().item())
print(lroc_sensitivity_at_fpf(table, 0.5))  # None if 0.5 FPF is off the curve
curve = lroc_curve_lazy(table).collect()
```

### Confusion Matrix

```python
from polars_cv.metrics import confusion_at_threshold

counts = confusion_at_threshold(table, threshold=0.5)
# ConfusionResult(tp=10, fp=3, fn=2)
counts.tp, counts.fp, counts.fn  # attribute access
counts.precision, counts.recall, counts.f1  # derived metrics
counts.to_dict()  # {'tp': 10, 'fp': 3, 'fn': 2}
```

## Bootstrap Confidence Intervals

The FROC / LROC / PR AUC confidence intervals are **fully lazy and group-aware**.
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

# Ungrouped: one row [auc, ci_lower, ci_upper].
froc_auc_ci_lazy(table, n_bootstrap=1000, seed=42).collect()

# Group-aware: one row per group, ready to join onto the point-metric frame.
ci = froc_auc_ci_lazy(table, group_by="group_id", n_bootstrap=1000, seed=42)
point = froc_auc(table, group_by="group_id")
point.join(ci.select("group_id", "ci_lower", "ci_upper"), on="group_id").collect()

# Entity-level resampling (e.g. by case), composing with the grouping.
froc_auc_ci_lazy(table, group_by="group_id", seed=42, sample_col="case_id")

lroc_auc_ci_lazy(table, n_bootstrap=1000, seed=42)
average_precision_ci_lazy(table, group_by="group_id", n_bootstrap=1000, seed=42)
```

The resample is a **position-independent hash** of each unit's global slot,
built collect-free by cross-joining a constant-length reps frame against the
units — so it never materializes the `n_bootstrap × n_units` frame and each group
resamples within itself, stratified by `gt_label`. Because the draw hashes its
own slot id (never a row position), a given `seed` reproduces the interval
**bit-for-bit regardless of thread count** (`POLARS_MAX_THREADS`) or streaming
morselization, and `seed=None` is deterministic (a fixed constant). The `auc` /
`ap` column is the deterministic point estimate; only the bounds are
bootstrapped. A **degenerate group** (one with no positive targets) keeps its
point estimate but reports null `ci_lower` / `ci_upper` rather than raising, so a
single plan spans viable and degenerate groups alike.

## IoU Re-thresholding

The `DetectionTable` stores raw IoU values from matching, enabling
re-thresholding without re-running the matching step:

```python
# Compute mAP across COCO IoU thresholds
map_val = mean_average_precision(
    table,
    iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
)
```

## Class-Aware Metrics

Pass a `class_col` to the matcher, then compute per-class or averaged metrics:

```python
table = adapter.match(data, ..., class_col="category")
ap_cat = average_precision(table, class_id="cat")
map_val = mean_average_precision(table)  # averages across all classes
```
