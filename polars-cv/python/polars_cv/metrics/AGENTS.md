# AGENTS.md — Metrics Subsystem (`polars_cv.metrics`)

> Read the [root AGENTS.md](../../../../AGENTS.md) and [`polars_cv/AGENTS.md`](../AGENTS.md) first.
> Update this file when metric APIs or behavior change.

## Purpose

Detection metrics built from polars-cv primitives and Polars lazy expressions:

- **PR**: `precision_recall_curve`, `average_precision`, `mean_average_precision`
- **Threshold**: `precision_at_threshold`, `recall_at_threshold`, `f1_at_threshold`, `confusion_at_threshold`
- **FROC/LROC** (expression-valued, lazy, group-aware): `froc_auc`/`lroc_auc`
  (LazyFrame, one row per group), `froc_curve_lazy`/`lroc_curve_lazy`,
  `froc_sensitivity_at_fp`/`lroc_sensitivity_at_fpf`, `froc_summary_table`
- **Bootstrap** (vectorized, seed-reproducible): `bootstrap_froc_auc`,
  `bootstrap_lroc_auc`, `bootstrap_pr_auc`; `bootstrap_metric_sequential` for
  custom callbacks
- **AUC integrals**: the single authority is `_auc_expr.py`
  (`trapz_auc_expr`, `partial_auc_expr`, `mann_whitney_auc_expr`,
  `collapse_curve`); `_auc.py` keeps `trapz_auc`/`partial_auc`/`_interp`/
  `mcclish_correction` for the PR curve + `MetricResult` interpolation only

## Architecture

```
Input Data → Matcher → DetectionTable → Metric function → pl.LazyFrame / pl.Expr
```

1. **Matchers** (`_matching/`) convert raw data into a canonical `DetectionTable` (two lazy frames). All implement the `Matcher` protocol. `ContourMatcher.match` also accepts a pre-decoded `LazyPipelineExpr` (via `_SourceHandle`) so a caller's graph can share the decode.
2. **FROC/LROC metric functions** (`_metrics/`) operate on `DetectionTable` and return a `pl.LazyFrame` (`froc_auc`, `froc_curve_lazy`, …) — no result object, no eager `.item()` until the caller collects. The integral is the reusable expression in `_auc_expr.py`.
3. **PR / Confusion** still return `MetricResult` subclasses (`_result.py`) with `auc()`, `interpolate()`, `summary_table()`, `bootstrap_ci()`. The FROC/LROC curve helpers reuse `MetricResult.interpolate`/`summary_table` on a collected `*_curve_lazy` frame.

### DetectionTable (`_types.py`)

Two aligned lazy frames:
- **detections** — one row per detection: `image_id`, `class_id`, `score`, `is_tp`, `gt_idx`, `iou`, `det_idx`
- **image_metadata** — one row per (image, class): `n_gts`, `weight`, `gt_label`

Supports IoU re-thresholding via `at_iou_threshold()`, class filtering via `filter_class()`, per-image aggregation via `to_per_image()`.

## Matchers

| Matcher | Input | Uses |
|---------|-------|------|
| `ContourMatcher` | heatmap + binary mask (blob, list, or array) | `Pipeline().threshold().extract_contours()`, `contour.correspond()` |
| `BBoxMatcher` | `List[BBOX_SCHEMA]` columns | `bbox.correspond()`, ordered by confidence here |
| `PreMatchedAdapter` | pre-computed TP/FP per detection | Direct DataFrame wrapping |

## AUC API

- **FROC/LROC**: `froc_auc(table, *, method, fp_range, correction, level, group_by)`
  → `pl.LazyFrame` (`[*group_by, auc]`). `method="trapezoidal"` (default) supports
  `correction="mcclish"|"normalize"` and `fp_range`/`fpf_range`; `method="mann_whitney"`
  (`level="detection"|"image"`) is a global rank statistic (no range/correction).
  A scalar is `froc_auc(table).collect().item()`; grouping is `group_by=`.
- **PR**: `PrecisionRecallResult.auc(method=...)` — `"all_points"` (default, monotone
  envelope), `"11_point"`, `"trapezoidal"`. PR still uses `_auc.trapz_auc`.

## Bootstrap

Vectorized, seed-reproducible, all replicates in one lazy plan:

```python
bootstrap_froc_auc(table, n_bootstrap=1000, seed=42)  # detection AUC
bootstrap_froc_auc(
    table, n_bootstrap=1000, seed=42, sample_col="case_id"
)  # entity-level
bootstrap_froc_auc(table, method="mann_whitney")  # MW AUC
bootstrap_lroc_auc(table, level="image")
bootstrap_pr_auc(table, n_bootstrap=1000, seed=42)
```

Each draw gets a distinct synthetic `image_id` (`_bootstrap_table_with_draws`) so
a redraw counts once per draw. `bootstrap_metric_sequential` remains for custom
metric callbacks; `MetricResult.bootstrap_ci` (with multi-metric dict + entity-level
via `_reconstruct`) still backs PR/Confusion results.

## File Layout

```
metrics/
├── __init__.py           # Public re-exports
├── _types.py             # DetectionTable, column constants, schema validation
├── _result.py            # MetricResult base (auc, bootstrap_ci, interpolate) — PR/Confusion
├── _auc.py               # eager AUC utilities kept for PR: trapz, partial, mcclish, _interp
├── _auc_expr.py          # the FROC/LROC integral authority: *_expr + collapse_curve
├── _bootstrap.py         # bootstrap_{froc,lroc,pr}_auc, bootstrap_metric_sequential
├── _matching/
│   ├── _protocol.py      # Matcher protocol
│   ├── _contour.py       # ContourMatcher
│   ├── _bbox.py          # BBoxMatcher
│   └── _prematched.py    # PreMatchedAdapter
└── _metrics/
    ├── _froc.py           # froc_auc, froc_curve_lazy, froc_sensitivity_at_fp, froc_summary_table
    ├── _lroc.py           # lroc_auc, lroc_curve_lazy, lroc_sensitivity_at_fpf
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
  FP-per-image denominator — is the number of distinct `image_id`s: the weighted
  denominator dedupes on `image_id` per group. Counting rows divides the
  false-positive rate by the number of classes.
- Bootstrap draws are renamed to distinct synthetic `image_id`s
  (`<image_id>#d<n>`) in `_bootstrap_table_with_draws`, so a redraw is a
  separate evaluation unit rather than a duplicate id. Nothing downstream has
  to guess whether a repeated id is a redraw or shared ownership.

### Duplicate `image_id` in metadata
- A repeated `image_id` (and `class_id`, when present) now means only one
  thing: one rendered image owned by two cases. FROC weight lookups dedupe by
  that key so detections are not fan-out-multiplied. Equal weights are fine;
  conflicting weights raise (the numerator would pick an arbitrary row while
  denominators sum every row). Prefer a composite key in `image_id` when each
  ownership should be a distinct evaluation unit.
- The conflict check is on `image_id` alone, which subsumes the
  `(image_id, class_id)` check: a `weight` is a property of an *image*, and the
  FP-per-image denominator dedupes on `image_id`, so two classes of one image
  disagreeing about its weight is exactly as ill-defined.
- The guard is **deferred** (`_guarded_weight_lookup` wraps the numerator's
  weight lookup in `pl.defer`) so `froc_curve_lazy` / `froc_auc` build a
  pure-lazy plan — nothing runs until the caller collects. The conflict
  therefore surfaces on `.collect()` as a `ComputeError` wrapping the
  `ValueError` message, not at construction time. Running the check eagerly here
  is what previously made the trapezoidal FROC path execute at build time.

### Interpolation beyond the curve
- `froc_sensitivity_at_fp` / `lroc_sensitivity_at_fpf` / `froc_summary_table`
  (built on `MetricResult.interpolate` / `summary_table` over a collected
  `*_curve_lazy` frame) return `None` / null for x-values outside the observed
  range — no endpoint clamping. `froc_summary_table`'s y column is Float64 even
  when every point is null.
- At an x the curve visits more than once, the *highest* y there is returned:
  `froc_sensitivity_at_fp(table, 0.0)` is the sensitivity reachable with no
  false positives, not the origin's zero.

## Known Issues

- Bbox matching converts to contours internally. Correct but suboptimal for axis-aligned boxes.
- Score + extract cannot be merged into one graph: `label_reduce` requires contours as an expression parameter, so they must exist as a column before the scoring pipeline runs.
