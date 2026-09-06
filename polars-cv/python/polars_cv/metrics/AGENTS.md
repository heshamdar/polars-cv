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
- **Bootstrap CIs** (lazy, group-aware, seed-reproducible): `froc_auc_ci_lazy`,
  `lroc_auc_ci_lazy`, `average_precision_ci_lazy` — each returns a `LazyFrame`
  `[*group_by, <metric>, ci_lower, ci_upper]` and never collects internally
- **AUC integrals**: the single authority is `_auc_expr.py`
  (`trapz_auc_expr`, `partial_auc_expr`, `collapse_curve`; the weighted
  Mann-Whitney two-stage `collapse_scores` + `mann_whitney_auc_expr`; and the
  lazy `interpolate_curve_lazy`); `_auc.py` keeps `trapz_auc`/`partial_auc`/
  `_interp`/`mcclish_correction` for the eager PR-curve `MetricResult.auc` only —
  its `interpolate`/`summary_table` delegate to `interpolate_curve_lazy`
- **Weighted Mann-Whitney**: `froc_auc`/`lroc_auc(method="mann_whitney")` are
  weighted by `image_metadata.weight` (both `level="detection"` and
  `level="image"`), via `collapse_scores` (bucket by distinct score, carrying the
  positive/negative weight mass) then `mann_whitney_auc_expr` (weighted rank-sum).
  A pure `rank("average")` reduction can't weight ties; bucketing removes them.
  Unit weights recover the standard tie-averaged MW — one implementation, not two,
  cross-checked against the pairwise `ref_weighted_mann_whitney` oracle

## Architecture

```
Input Data → Matcher → DetectionTable → Metric function → pl.LazyFrame / pl.Expr
```

1. **Matchers** (`_matching/`) convert raw data into a canonical `DetectionTable` (two lazy frames). All implement the `Matcher` protocol. `ContourMatcher.match` also accepts a pre-decoded `LazyPipelineExpr` (via `_SourceHandle`) so a caller's graph can share the decode.
2. **FROC/LROC metric functions** (`_metrics/`) operate on `DetectionTable` and return a `pl.LazyFrame` (`froc_auc`, `froc_curve_lazy`, …) — no result object, no eager `.item()` until the caller collects. The integral is the reusable expression in `_auc_expr.py`.
3. **PR / Confusion** still return `MetricResult` subclasses (`_result.py`) with `auc()`, `interpolate()`, `summary_table()`. The FROC/LROC curve helpers reuse `MetricResult.interpolate`/`summary_table` on a collected `*_curve_lazy` frame. Confidence intervals are the free `*_ci_lazy` functions in `_bootstrap.py`, not a result-object method.

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

## Bootstrap CIs

Seed-reproducible, group-aware, and **fully lazy** — the entire bootstrap
(resample, per-replicate metric, and the percentile bounds) is one Polars plan
the caller collects. Three free functions in `_bootstrap.py` are the only way in:

```python
froc_auc_ci_lazy(table, n_bootstrap=1000, seed=42)                    # detection AUC
froc_auc_ci_lazy(table, group_by="group_id", seed=42)                 # per-group CI
froc_auc_ci_lazy(table, group_by="group_id", sample_col="case_id")    # entity-level
froc_auc_ci_lazy(table, method="mann_whitney")                        # MW AUC
lroc_auc_ci_lazy(table, level="image")
average_precision_ci_lazy(table, group_by="group_id", n_bootstrap=1000, seed=42)
```

Each returns a `LazyFrame` `[*group_by, <metric>, ci_lower, ci_upper]`: one row
per group (a single row when `group_by=None`). The `<metric>` column (`auc`/`ap`)
is the deterministic point estimate — `froc_auc`/`lroc_auc`/`all_points_ap_by_group`
grouped by the real keys — so **only the bounds are bootstrapped**. Nothing
collects internally (`tests/test_bootstrap_ci_lazy.py` pins zero
`pl.LazyFrame.collect` during plan construction), so a CI can be built at plan
time with no data and joined onto a point-metric frame.

### The lazy resampler (`_lazy_resample`)

Resampling is a **position-independent hash expression**, not a Python loop over
eager `pl.Series.sample` calls, and it is **collect-free**: a constant-length
reps frame (`int_range(0, n_bootstrap)`) is cross-joined against the base units,
so the per-replicate slot count is a window (`pl.len().over("bootstrap_id")`),
never a materialized scalar. Each draw is `hash(slot, seed) % stratum_size` where
`slot = bootstrap_id * n_units + pos`; because it depends only on the row's own
global slot id, it is **identical across thread counts and streaming morsels**
(guarded by `test_bootstrap_ci_lazy.py::TestThreadCountInvariant`) — the same
property that removes the join-order nondeterminism the old path fought (the
macOS negative-AUC comment in `_bootstrap.py`). Draws are **partitioned within
`group_keys`** (a group only ever redraws its own units) and **stratified within
`gt_label`** for image-level draws (each `(group, stratum)` redrawn to its own
size); entity-level (`sample_col`) resamples entities within group, then expands
to images with a lazy `group_by`/`explode`. An empty base or empty group
cross-joins to zero rows — it does **not** raise.

`seed=None` maps to a fixed hash constant, so the CI is **deterministic even
without an explicit seed**. Each draw gets a distinct synthetic `image_id` from
its deterministic global slot (`_bootstrap_table_with_draws`) so a redraw counts
once per draw.

### The interval (`_bootstrap_ci_from_replicates`)

Per-group bounds are a lazy `group_by(group_keys).agg(quantile(...))` over the
complete `groups × int_range(n_bootstrap)` grid: absent replicates are filled
with `empty_value` (`0.0`, or `0.5` for Mann-Whitney — a resample that drew no
detections legitimately scores that). A **degenerate group** nulls its
`ci_lower`/`ci_upper` instead of reporting a spurious interval, while keeping its
point estimate. Viability needs ≥1 positive target (`sum(gt_label) > 0`); for the
two-class rank statistics (`method="mann_whitney"`, threaded as
`require_both_classes`) it additionally needs ≥1 negative, since that AUC is
undefined without both classes. That viability rule is the one behavioral choice
worth knowing.

## File Layout

```
metrics/
├── __init__.py           # Public re-exports
├── _types.py             # DetectionTable, column constants, schema validation
├── _result.py            # MetricResult base (auc, interpolate, summary_table) — PR/Confusion
├── _auc.py               # eager AUC utilities kept for PR: trapz, partial, mcclish, _interp
├── _auc_expr.py          # the FROC/LROC integral authority: *_expr + collapse_curve
├── _bootstrap.py         # {froc_auc,lroc_auc,average_precision}_ci_lazy + lazy resampler
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
- **No Python loops over rows**: All curve aggregation uses Polars expressions (`explode`/`group_by`/`cum_sum`/window functions). Bootstrap resampling is likewise loop-free — a hash-expression draw over a lazy `int_range` skeleton (`_lazy_resample`), not a `range(n_bootstrap)` loop.
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
  denominator resolves one weight per `image_id` per group. Counting rows divides
  the false-positive rate by the number of classes.
- Bootstrap draws are renamed to distinct synthetic `image_id`s
  (`<image_id>#d<n>`) in `_bootstrap_table_with_draws`, so a redraw is a
  separate evaluation unit rather than a duplicate id. Nothing downstream has
  to guess whether a repeated id is a redraw or shared ownership.

### Duplicate `image_id` in metadata — `weight_agg`, no guard
- A repeated `(image_id[, class_id])` key means one rendered image owned by two
  cases. The per-key weight is resolved to a single value by `_weights.py`'s
  `resolve_key_weights`, keyed by the same policy for the numerator lookup and
  the summed denominators, so detections are never fan-out-multiplied and the
  result does not depend on row order.
- **There is no build-time guard.** The old `_raise_on_conflicting_weights`
  eagerly collected `image_metadata` to fail on disagreeing weights; that broke
  pure-lazy streaming (a collect before the caller asked for one). It was removed
  in favour of a `weight_agg` keyword (`"first"` default, plus `"min"`/`"max"`/
  `"mean"`/`"sum"`) on `froc_curve_lazy`/`froc_auc`/`lroc_curve_lazy`/`lroc_auc`
  and the standalone helpers. `"first"` keeps the cheap `unique(keep="first")`
  and is not guaranteed stable when weights disagree — supplying consistent
  weights is the caller's responsibility; the other policies are order-independent.

### Interpolation beyond the curve — lazy
- `froc_sensitivity_at_fp` / `lroc_sensitivity_at_fpf` / `froc_summary_table`
  return a **`LazyFrame`** (the caller collects — no method collects internally),
  built on the single lazy authority `_auc_expr.interpolate_curve_lazy`
  (`collapse_curve` + backward/forward `join_asof`). `sensitivity` is `null` for
  x-values outside the observed range — no endpoint clamping — and the summary's
  y column stays Float64 even when every point is null.
- At an x the curve visits more than once, the *highest* y there is returned:
  `froc_sensitivity_at_fp(table, 0.0).collect().item()` is the sensitivity
  reachable with no false positives, not the origin's zero.

## Known Issues

- Bbox matching converts to contours internally. Correct but suboptimal for axis-aligned boxes.
- Score + extract cannot be merged into one graph: `label_reduce` requires contours as an expression parameter, so they must exist as a column before the scoring pipeline runs.
