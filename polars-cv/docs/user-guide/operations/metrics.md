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
from polars_cv.metrics import ContourMatcher, froc_curve

matcher = ContourMatcher(iou_threshold=0.5, extraction_threshold=0.1)
table = matcher.match(data, pred_col="heatmap", gt_col="gt_mask")
result = froc_curve(table)
```

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

The curve carries **one point per distinct score**, not one per detection. A
threshold cannot admit one detection of a tied group and reject another, so a
run of tied scores is a single operating point — and computing it that way is
what makes AP a function of the detections alone rather than of the order they
happened to arrive in. `average_precision` applies the monotone precision
envelope and sums `Σ (Rₙ − Rₙ₋₁) · Pₙ`, matching COCO and scikit-learn.

### FROC

```python
from polars_cv.metrics import froc_curve

result = froc_curve(table)
print(result.auc(fp_range=(0, 8)))
print(result.sensitivity_at_fp(1.0))
print(result.summary_table())
```

`sensitivity_at_fp` returns `None` — and `summary_table` a null — when the
requested FP/image rate lies beyond the curve's observed range. An operating
point the detector never reaches is reported as unreachable rather than
clamped to the last value on the curve. Where an x is visited more than once,
the highest sensitivity there is returned.

### LROC

```python
from polars_cv.metrics import lroc_curve

result = lroc_curve(table)
print(result.auc())
print(result.sensitivity_at_fpf(0.5))  # None if 0.5 FPF is off the curve
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

```python
# Sequential (works with any metric)
ci = result.bootstrap_ci(n_bootstrap=1000, seed=42)

# Vectorized (faster, for PR AUC)
from polars_cv.metrics import bootstrap_pr_auc
ci = bootstrap_pr_auc(table, n_bootstrap=1000, seed=42)
```

These are not the same resampling scheme, and the difference is not only speed:

- `result.bootstrap_ci(...)` draws every sampling unit from one pool
  (**unstratified**), so a replicate's positive/negative image balance varies.
- `bootstrap_pr_auc(...)` **stratifies** draws on `gt_label`, holding that
  balance fixed across replicates.

Both are defensible estimators, but they have different variance. Do not quote a
FROC interval from the first alongside a PR interval from the second as though
they were computed alike.

Pass `sample_col` to `bootstrap_ci` when the sampling unit should be an entity
rather than an image — draws are then taken over the entity and expanded back to
the images it owns:

```python
ci = result.bootstrap_ci(n_bootstrap=1000, seed=42, sample_col="case_id")
```

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
