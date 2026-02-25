# AGENTS.md — Metrics Subsystem (`polars_cv.metrics`)

> Read the root `AGENTS.md` and `polars-cv/python/polars_cv/AGENTS.md` first.
> Update this file when metric APIs or behavior change.

## Purpose

This subpackage provides detection metrics built from polars-cv primitives
and Polars lazy expressions:

- **Precision-Recall**: `precision_recall_curve`, `average_precision`, `mean_average_precision`
- **Threshold metrics**: `precision_at_threshold`, `recall_at_threshold`, `f1_at_threshold`
- **Confusion matrix**: `confusion_at_threshold`
- **FROC** (Free-response ROC): `froc_curve`
- **LROC** (Localization ROC): `lroc_curve`
- **Bootstrap**: `bootstrap_metric_sequential` (general), `bootstrap_pr_auc` (vectorized PR fast path)
- **AUC utilities**: `trapz_auc`, `partial_auc`, `mcclish_correction`, `mann_whitney_u_auc`, `detection_level_mann_whitney` (note: `weighted_curve` was removed — weighting is now handled inline in `_curve_from_detections`)

## Architecture

```
Input Data → Matcher → DetectionTable → Metric Function → MetricResult
```

### Three-layer pipeline

1. **Matchers** (`_matching/`) convert raw data into a canonical `DetectionTable`.
   All matchers implement the `Matcher` protocol (`_matching/_protocol.py`).
2. **Metric functions** (`_metrics/`) operate purely on `DetectionTable` and
   return a `MetricResult` subclass.
3. **Result objects** (`_result.py`) carry curves with unified `auc(method=...)`,
   `interpolate()`, `summary_table()`, and `bootstrap_ci()` methods.

### DetectionTable (`_types.py`)

The canonical intermediate representation. Two aligned lazy frames:

- **detections** — one row per detection: `image_id`, `class_id`, `score`,
  `is_tp`, `gt_idx`, `iou`, `det_idx`.
- **image_metadata** — one row per (image, class): `n_gts`, `weight`, `gt_label`.

Supports IoU re-thresholding via `at_iou_threshold()` (flips `is_tp` without
re-running the matcher), class filtering via `filter_class()`, and per-image
aggregation via `to_per_image()`.

## Matchers

| Matcher | Input | Uses |
|---------|-------|------|
| `ContourMatcher` | heatmap + binary mask (any format: blob, list, array) | `Pipeline().threshold().extract_contours()`, `contour.match_detections()` |
| `BBoxMatcher` | `List[BBOX_SCHEMA]` columns | Rust `bbox_match_detections` plugin |
| `PreMatchedAdapter` | pre-computed TP/FP per detection | Direct DataFrame wrapping |

## Unified AUC API

All result types expose AUC via a single `auc(method=...)` method:

- **`FROCResult.auc(method="trapezoidal"|"mann_whitney", *, fp_range=None, correction=None, level="detection")`**
- **`LROCResult.auc(method="trapezoidal"|"mann_whitney", *, fpf_range=None, correction=None, level="image")`**
- **`PrecisionRecallResult.auc(method="all_points"|"11_point"|"trapezoidal")`**

The `correction` parameter (trapezoidal only) supports:
- `None` — raw area
- `"normalize"` — divide by x-range width
- `"mcclish"` — McClish's standardized partial AUC (maps to [0.5, 1.0])

Mann-Whitney is a global rank statistic (`P(positive > negative)`) and
does not support range/correction parameters.

## Bootstrap

Bootstrap CI is handled by `MetricResult.bootstrap_ci()` in the base class.
It uses a `_reconstruct(sampled_ids)` hook that each subclass implements
to rebuild a result from resampled image IDs.  The `_resolve_metric`
mechanism means any public method can be bootstrapped:

```python
# Bootstrap trapezoidal AUC (default)
result.bootstrap_ci(metric="auc")

# Bootstrap Mann-Whitney AUC
result.bootstrap_ci(metric="auc", metric_kwargs={"method": "mann_whitney", "level": "detection"})

# Bootstrap sensitivity at a specific FP rate
result.bootstrap_ci(metric="sensitivity_at_fp", metric_kwargs={"fp_per_image": 1.0})
```

### Multi-metric bootstrap

Pass a dict to `metric` to compute multiple metrics with shared
reconstruction (one reconstruction per iteration, zero extra overhead):

```python
cis = result.bootstrap_ci(
    n_bootstrap=1000,
    metric={
        "mw_auc": {"metric": "auc", "method": "mann_whitney"},
        "partial_auc": {"metric": "auc", "fp_range": (0, 2), "correction": "mcclish"},
    },
)
# Returns dict[str, BootstrapResult]
cis["mw_auc"].ci_lower, cis["partial_auc"].point_estimate
```

### Entity-level bootstrap

Pass `sample_col` to resample at a higher grouping level (e.g. case,
patient) instead of individual images:

```python
# Resample by case_id instead of image_id
ci = result.bootstrap_ci(sample_col="case_id")
```

The `sample_col` must exist in the image_metadata LazyFrame. Entity-level
sampling extracts unique entity values, samples them with replacement,
then expands back to all image IDs belonging to each sampled entity.

This is orthogonal to the `level` parameter on Mann-Whitney AUC, which
controls what the metric compares (individual detections vs per-image
aggregates), not what the bootstrap resamples.

The vectorized `bootstrap_pr_auc` remains available as a fast path for
PR AUC specifically.

## Design Principles

- **NumPy-free**: The entire metrics module is Polars-native with zero NumPy
  dependency. All numerical operations (AUC, interpolation, rank statistics,
  bootstrap sampling, percentile intervals) use Polars Series/expressions or
  pure Python. This follows patterns from the
  [rapidstats](https://github.com/CangyuanLi/rapidstats) library.
- All curve aggregation is done with Polars expressions (`explode`/`group_by`/
  `cum_sum`/window functions), not Python loops over rows.
- AUC computation uses `pl.Series.diff()` + `shift()` for trapezoidal
  integration, `search_sorted()` for interpolation, and `over("score")` for
  Mann-Whitney rank averaging.
- Bootstrap sampling uses `pl.Series.sample(with_replacement=True)` and
  `pl.Series.quantile()` for percentile intervals.
- Monotone-envelope AP uses `pl.Series.reverse().cum_max().reverse()`.
- Materialization points use `collect(engine="streaming")`.
- LROC materializes matched rows once and reuses for validation and
  top-detection reduction.
- FROC and LROC curve construction both use sorted score buckets + cumulative
  sums to avoid quadratic scaling (the dense cross-join grid was removed
  from FROC).
- Class-aware metrics: `class_id` is optional; when present, metric functions
  include it in `group_by` operations.
- IoU values are preserved from matching, enabling re-thresholding without
  re-running the matcher (used by `mean_average_precision`).

## Primitive Assumptions

- `extract_contours().sink("native")` returns contour sets (`List[Contour]`).
- Contour schema fields are `exterior`, `holes`, and `is_closed`.
- `BBOX_SCHEMA` is `Struct{x_min, y_min, x_max, y_max}` (all Float64).
- `MATCH_RESULT_SCHEMA` fields: `pred_idx`, `gt_idx`, `iou` (all List[Int32/Float64]).
- Bbox matching in Rust converts bboxes to rectangular contours and reuses
  contour IoU logic. A `TODO` exists for direct axis-aligned optimization.

## File Layout

```
metrics/
├── __init__.py           # Public re-exports
├── _types.py             # DetectionTable, column constants, schema validation
├── _result.py            # MetricResult base class (auc, bootstrap_ci, interpolate)
├── _auc.py               # trapz_auc, partial_auc, mcclish_correction, mann_whitney_u_auc, detection_level_mann_whitney, _interp
├── _bootstrap.py         # bootstrap_metric_sequential, bootstrap_pr_auc, BootstrapResult
├── _matching/
│   ├── __init__.py       # Matcher exports
│   ├── _protocol.py      # Matcher protocol
│   ├── _contour.py       # ContourMatcher
│   ├── _bbox.py          # BBoxMatcher
│   └── _prematched.py    # PreMatchedAdapter
└── _metrics/
    ├── __init__.py       # Metric function exports
    ├── _froc.py          # froc_curve, FROCResult
    ├── _lroc.py          # lroc_curve, LROCResult
    ├── _precision_recall.py  # PR curve, AP, mAP, precision/recall/f1 at threshold
    └── _confusion.py     # confusion_at_threshold
```

## Important Patterns

- **Null contour handling**: When contour extraction finds no contours, it
  returns `null` (not an empty list). All matchers must use `.fill_null(0)`
  on `list.len()` for `n_gts` and `gt_label` derivations. `BBoxMatcher`
  already does this; `ContourMatcher` was fixed to match.
- **`to_per_image()` sort reliability**: Uses `sort_by()` within the
  `group_by().agg()` context to guarantee the highest-scoring detection is
  selected. Never rely on a `.sort()` before `.group_by()` — Polars does
  not guarantee order preservation across `group_by`.
- **LROC image-level summarization**: `DetectionTable.to_per_image()` now
  carries a score-sorted `detections` list per image. LROC supports two
  variants via `lroc_curve(variant=...)`:
  - `"best_tp"`: effective image score = highest-scoring TP detection
    (any TP above threshold → correctly localized).
  - `"top_scoring"`: effective image score = top detection regardless of
    TP/FP status; correctly localized only if that top detection is TP
    (classical single-commitment LROC).
- **IoU re-thresholding direction**: `at_iou_threshold()` only works
  reliably when *raising* the threshold above the original matching IoU.
  Lowering it has no effect because unmatched detections have no stored
  `gt_idx`/`iou` to re-evaluate. A `UserWarning` is emitted when this
  is attempted. The `_matching_iou_threshold` field on `DetectionTable`
  tracks the matcher's original threshold.
- **ContourMatcher default `min_contour_area`**: Changed from 0.0 to 1.0.
  Sub-pixel contours from the boundary tracer are now excluded by default.
- **Pre-match zero-score filtering**: Zero-score contours (empty rasterized
  interior) are now filtered *before* matching via `_filter_zero_score_detections`,
  using `list.gather` with indices of positive scores. This prevents them
  from claiming GT objects during greedy IoU assignment — the previous
  post-explode approach could cause false negatives.
- **Centroid fallback scoring**: The Rust `label_reduce` with
  `region_mode="interior"` falls back to sampling the heatmap at the
  contour centroid when no interior pixels are found. This prevents
  sub-pixel contours from receiving a score of 0.
- **`label_reduce` region modes**: Three modes are available:
  `"interior"` (strict interior, default), `"boundary"` (interior +
  boundary pixels — avoids zero-score artifacts for small contours),
  `"bbox"` (all pixels in bounding box).
- **ContourMatcher format-agnostic input**: `ContourMatcher.match()` auto-detects
  the source format from Polars column dtypes at planning time using
  `_detect_source_info`.  Supported formats: `Binary` (blob/VIEW protocol),
  `List[List[...]]` (nested list), and `Array[...]` (fixed-size).  All
  pipeline helpers (`_extract_contours_from_col`, `_extract_with_fused_resize`,
  `_score_contours_from_heatmap`) receive the detected `_SourceInfo` to
  build pipelines with the correct source format.
- **ContourMatcher auto_resize simplification**: `auto_resize=True` is the
  default.  When enabled, predictions are always resized to GT dimensions
  via a fused pipeline (resize is a no-op if shapes already match).  GT
  dimensions are extracted via `_add_gt_shape_columns` using
  format-appropriate methods: `.list.len()` for lists,
  `Pipeline.extract_shape()` via Rust for blobs, type metadata for arrays.
  When `auto_resize=False`, shapes are assumed to match (trusting the user)
  — no shape validation or extraction is performed.
- **Unified `auc(method=...)` API**: All result types expose AUC through
  a single method with a `method` parameter.  FROC/LROC support
  `"trapezoidal"` (default) and `"mann_whitney"`.  PR supports
  `"all_points"` (default, monotone envelope), `"11_point"`, and
  `"trapezoidal"` (raw, no envelope).  The old `mann_whitney_auc()` and
  `raw_auc()` methods have been removed.
- **McClish partial AUC correction**: The `correction` parameter on
  trapezoidal AUC supports `"mcclish"` for standardized partial AUC
  (McClish 1989).  Maps raw pAUC to [0.5, 1.0] where 0.5 = chance level.
  Also supports `"normalize"` (divide by range width).
- **Base-class bootstrap**: `bootstrap_ci()` lives on `MetricResult` and
  uses `_reconstruct(sampled_ids)` / `_get_detection_table()` hooks.
  The bootstrap loop, sampling, and multi-metric dispatch all live in the
  base class. Each subclass only implements `_reconstruct()`:
  - `FROCResult`: joins sampled IDs against detections/metadata and
    re-derives the curve via `_curve_from_detections()` (cumulative sums)
  - `LROCResult`: per-image join + sampled DetectionTable
  - `PrecisionRecallResult`: full `precision_recall_curve()` on sampled table
- **FROC cumulative-sum curve**: `froc_curve()` uses a cumulative-sum
  approach (matching LROC): detections are bucketed by score, sorted
  descending, and TP/FP counts are accumulated. This replaces the previous
  O(images × thresholds) dense cross-join grid, reducing memory from
  ~8 GB to ~10 MB and time from ~15s to <1s for 41K images. The
  `_curve_from_detections()` helper handles both weighted and unweighted
  cases. `_curve_from_dense()`, `_derive_thresholds()`, and
  `weighted_curve()` have been removed.
- **FROCResult no longer stores `per_image_threshold`**: The dense grid
  field was removed. `_reconstruct()` joins against the small detections
  table and re-derives the curve via cumulative sums (~4.6K rows instead
  of 190M rows), making bootstrap ~240x faster.
- **LROC variants**: `lroc_curve(table, variant=...)` supports two
  scoring modes: `"best_tp"` (default — effective score is the best TP
  detection for positive images) and `"top_scoring"` (effective score is
  the single highest-scoring detection, localized only if that top
  detection is a TP — classical Swensson 1996 formulation).

## Known Issues

- Bbox matching in Rust converts to contours as an intermediate step. This
  works correctly but is suboptimal for axis-aligned boxes where intersection
  can be computed directly.
- **Score + extract cannot be merged today**: The scoring step
  (`label_reduce`) takes contours as an expression parameter, requiring
  them to exist as a DataFrame column before the scoring pipeline runs.
  Merging extract + score into one graph would require the graph executor
  to support binding a node's output as another node's parameter.
