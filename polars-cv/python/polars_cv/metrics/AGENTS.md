# AGENTS.md — Metrics Subsystem (`polars_cv.metrics`)

> Read the [root AGENTS.md](../../../../AGENTS.md) and [`polars_cv/AGENTS.md`](../AGENTS.md) first.
> Update this file when metric APIs or behavior change.

## Purpose

Detection metrics built from polars-cv primitives and Polars lazy expressions:

- **PR**: `precision_recall_curve`, `average_precision`, `mean_average_precision`
- **Threshold**: `precision_at_threshold`, `recall_at_threshold`, `f1_at_threshold`, `confusion_at_threshold`
- **FROC/LROC**: `froc_curve`, `lroc_curve`
- **Bootstrap**: `bootstrap_metric_sequential` (general), `bootstrap_pr_auc` (vectorized fast path)
- **AUC**: `trapz_auc`, `partial_auc`, `mcclish_correction`, `mann_whitney_u_auc`, `detection_level_mann_whitney`

## Architecture

```
Input Data → Matcher → DetectionTable → Metric Function → MetricResult
```

1. **Matchers** (`_matching/`) convert raw data into a canonical `DetectionTable`. All implement the `Matcher` protocol.
2. **Metric functions** (`_metrics/`) operate on `DetectionTable`, return a `MetricResult` subclass.
3. **Result objects** (`_result.py`) carry curves with `auc(method=...)`, `interpolate()`, `summary_table()`, `bootstrap_ci()`.

### DetectionTable (`_types.py`)

Two aligned lazy frames:
- **detections** — one row per detection: `image_id`, `class_id`, `score`, `is_tp`, `gt_idx`, `iou`, `det_idx`
- **image_metadata** — one row per (image, class): `n_gts`, `weight`, `gt_label`

Supports IoU re-thresholding via `at_iou_threshold()`, class filtering via `filter_class()`, per-image aggregation via `to_per_image()`.

## Matchers

| Matcher | Input | Uses |
|---------|-------|------|
| `ContourMatcher` | heatmap + binary mask (blob, list, or array) | `Pipeline().threshold().extract_contours()`, `contour.match_detections()` |
| `BBoxMatcher` | `List[BBOX_SCHEMA]` columns | Rust `bbox_match_detections` plugin |
| `PreMatchedAdapter` | pre-computed TP/FP per detection | Direct DataFrame wrapping |

## AUC API

All result types expose AUC via `auc(method=...)`:

- **FROC/LROC**: `"trapezoidal"` (default), `"mann_whitney"`. Trapezoidal supports `correction="mcclish"|"normalize"` and range parameters (`fp_range`/`fpf_range`).
- **PR**: `"all_points"` (default, monotone envelope), `"11_point"`, `"trapezoidal"` (raw).

Mann-Whitney is a global rank statistic (`P(positive > negative)`) — no range/correction support.

## Bootstrap

`MetricResult.bootstrap_ci()` in the base class. Uses `_reconstruct(sampled_ids)` hook per subclass.

```python
result.bootstrap_ci(metric="auc")  # default
result.bootstrap_ci(
    metric="auc", metric_kwargs={"method": "mann_whitney"}
)  # Mann-Whitney
result.bootstrap_ci(
    metric={  # multi-metric
        "mw": {"metric": "auc", "method": "mann_whitney"},
        "pauc": {"metric": "auc", "fp_range": (0, 2), "correction": "mcclish"},
    }
)
result.bootstrap_ci(sample_col="case_id")  # entity-level resampling
```

`bootstrap_pr_auc` remains as a vectorized fast path for PR AUC specifically.

## File Layout

```
metrics/
├── __init__.py           # Public re-exports
├── _types.py             # DetectionTable, column constants, schema validation
├── _result.py            # MetricResult base class (auc, bootstrap_ci, interpolate)
├── _auc.py               # AUC utilities (trapz, partial, mcclish, mann_whitney)
├── _bootstrap.py         # bootstrap_metric_sequential, bootstrap_pr_auc, BootstrapResult
├── _matching/
│   ├── _protocol.py      # Matcher protocol
│   ├── _contour.py       # ContourMatcher
│   ├── _bbox.py          # BBoxMatcher
│   └── _prematched.py    # PreMatchedAdapter
└── _metrics/
    ├── _froc.py           # froc_curve, FROCResult
    ├── _lroc.py           # lroc_curve, LROCResult
    ├── _precision_recall.py  # PR curve, AP, mAP, threshold metrics
    └── _confusion.py     # confusion_at_threshold
```

## Design Principles

- **NumPy-free**: Entirely Polars-native. All numerical operations (AUC, interpolation, rank statistics, bootstrap sampling, percentile intervals) use Polars Series/expressions or pure Python.
- **No Python loops over rows**: All curve aggregation uses Polars expressions (`explode`/`group_by`/`cum_sum`/window functions).
- **Cumulative-sum curves**: FROC and LROC use sorted score buckets + cumulative sums to avoid quadratic scaling.
- **Class-aware**: `class_id` is optional; when present, metric functions include it in `group_by`.
- **IoU preservation**: IoU values from matching enable re-thresholding without re-running the matcher.
- **Streaming materialization**: Materialization points use `collect(engine="streaming")`.

## Important Patterns

### Null and edge-case handling
- Contour extraction returns `null` (not empty list) when no contours found. All matchers use `.fill_null(0)` on `list.len()` for `n_gts`.
- Zero-score contours are filtered *before* matching via `_filter_zero_score_detections` to prevent them from claiming GT objects.
- `label_reduce` with `region_mode="interior"` falls back to centroid sampling when no interior pixels exist.

### ContourMatcher behavior
- `min_contour_area` defaults to 1.0 (excludes sub-pixel contours).
- Auto-detects source format (`Binary`/`List`/`Array`) from column dtypes via `_detect_source_info`.
- `auto_resize=True` (default) resizes predictions to GT dimensions via fused pipeline. `auto_resize=False` assumes shapes match.
- Three `label_reduce` region modes: `"interior"` (default), `"boundary"` (interior + boundary pixels), `"bbox"`.

### Sorting and aggregation
- `to_per_image()` uses `sort_by()` within `group_by().agg()` — never rely on `.sort()` before `.group_by()` since Polars does not guarantee order preservation across `group_by`.
- FROC / LROC curves are returned sorted by **descending `threshold`**, which is
  ascending `fp_per_image` / `fpf` (plotting order). Sort on the threshold, never
  on the x-column: thresholds are unique so the order is total, while `fp_per_image`
  ties constantly and Polars' `sort` defaults to `maintain_order=False`, leaving
  the y at each tie boundary — and therefore the AUC — unspecified.
- Every consumer of a curve's geometry goes through `MetricResult._curve_xy`,
  which collapses tied x to the maximum y (the ROC upper envelope) before
  integrating or interpolating. `auc()` and `interpolate()` must not sort for
  themselves; a second sort is a second answer.

### IoU re-thresholding
- `at_iou_threshold()` only works reliably when *raising* the threshold. Lowering has no effect (unmatched detections lack stored IoU). A `UserWarning` is emitted.

### LROC variants
- `"best_tp"` (default): effective score = highest-scoring TP detection for positive images.
- `"top_scoring"`: effective score = single highest-scoring detection regardless of TP/FP (classical Swensson 1996).

### PreMatchedAdapter population
- Prefer `image_meta=` covering the full evaluation population. Without it the
  adapter derives metadata from detections only and silently drops images with
  zero detections (inflating recall / FP-per-image); a `UserWarning` is emitted.
- `image_meta` is the *sole* source of `image_metadata`, so combining it with
  `n_gts_col` / `weight_col` / `gt_label_col` / `group_col` raises — those
  arguments only ever described how to derive metadata from the detection
  frame, and accepting them alongside `image_meta` would silently discard them.

### The FROC evaluation unit
- An `image_metadata` row is one (image, class). The **image count** — the
  FP-per-image denominator, and `FROCResult.n_images` — is the number of
  distinct `image_id`s, read once via `_count_images`. Counting rows divides
  the false-positive rate by the number of classes.
- Bootstrap draws are renamed to distinct synthetic `image_id`s
  (`<image_id>#draw<n>`) in `FROCResult._reconstruct`, so a redraw is a
  separate evaluation unit rather than a duplicate id. Nothing downstream has
  to guess whether a repeated id is a redraw or shared ownership.

### Duplicate `image_id` in metadata
- A repeated `image_id` (and `class_id`, when present) now means only one
  thing: one rendered image owned by two cases. FROC weight lookups dedupe by
  that key so detections are not fan-out-multiplied. Equal weights are fine;
  conflicting weights raise `ValueError` (the numerator would pick an arbitrary
  row while denominators sum every row). Prefer a composite key in `image_id`
  when each ownership should be a distinct evaluation unit.
- The conflict check is on `image_id` alone, which subsumes the
  `(image_id, class_id)` check: a `weight` is a property of an *image*, and the
  FP-per-image denominator dedupes on `image_id`, so two classes of one image
  disagreeing about its weight is exactly as ill-defined.

### Interpolation beyond the curve
- `MetricResult.interpolate` / `sensitivity_at_fp` / `sensitivity_at_fpf` /
  `summary_table` return `None` / null for x-values outside the observed
  range — no endpoint clamping. `summary_table`'s y column is Float64 even when
  every point is null.
- At an x the curve visits more than once, the *highest* y there is returned:
  `sensitivity_at_fp(0.0)` is the sensitivity reachable with no false
  positives, not the origin's zero.

## Known Issues

- Bbox matching converts to contours internally. Correct but suboptimal for axis-aligned boxes.
- Score + extract cannot be merged into one graph: `label_reduce` requires contours as an expression parameter, so they must exist as a column before the scoring pipeline runs.
- `rasterize(anti_alias=)` is plumbed but view-buffer's rasterizer ignores the flag.
