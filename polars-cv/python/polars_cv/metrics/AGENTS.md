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
- **Bootstrap**: `bootstrap_metric_sequential` (general), `bootstrap_pr_auc` (vectorized)
- **AUC utilities**: `trapz_auc`, `partial_auc`

## Architecture

```
Input Data → Matcher → DetectionTable → Metric Function → MetricResult
```

### Three-layer pipeline

1. **Matchers** (`_matching/`) convert raw data into a canonical `DetectionTable`.
   All matchers implement the `Matcher` protocol (`_matching/_protocol.py`).
2. **Metric functions** (`_metrics/`) operate purely on `DetectionTable` and
   return a `MetricResult` subclass.
3. **Result objects** (`_result.py`) carry curves with `auc()`, `partial_auc()`,
   `interpolate()`, and `summary_table()` methods.

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
| `ContourMatcher` | heatmap + binary mask | `Pipeline().threshold().extract_contours()`, `contour.match_detections()` |
| `BBoxMatcher` | `List[BBOX_SCHEMA]` columns | Rust `bbox_match_detections` plugin |
| `PreMatchedAdapter` | pre-computed TP/FP per detection | Direct DataFrame wrapping |

## Design Principles

- All curve aggregation is done with Polars expressions (`explode`/`group_by`/
  `cum_sum`/window functions), not Python loops over rows.
- Materialization points use `collect(engine="streaming")`.
- LROC materializes matched rows once and reuses for validation and
  top-detection reduction.
- LROC curve construction uses sorted score buckets + cumulative sums to avoid
  quadratic scaling.
- Class-aware metrics: `class_id` is optional; when present, metric functions
  include it in `group_by` operations.
- IoU values are preserved from matching, enabling re-thresholding without
  re-running the matcher (used by `mean_average_precision`).

## Bootstrap

Two strategies:

- **`bootstrap_metric_sequential`**: General-purpose. Samples image IDs with
  replacement, rebuilds DetectionTable, applies any metric function. Sequential
  Python loop — flexible but slower.
- **`bootstrap_pr_auc`**: Vectorized Polars-native path for PR AUC. All
  bootstrap samples live in one DataFrame; uses window functions across
  `bootstrap_idx` for parallel computation. Much faster for large datasets.

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
├── _result.py            # MetricResult base class
├── _auc.py               # trapz_auc, partial_auc, weighted_curve
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
  carries a score-sorted `detections` list per image. LROC derives the
  effective image score from the best localized detection (highest-scoring TP)
  on positive images, and from the top score on negative images.
- **IoU re-thresholding direction**: `at_iou_threshold()` only works
  reliably when *raising* the threshold above the original matching IoU.
  Lowering it has no effect because unmatched detections have no stored
  `gt_idx`/`iou` to re-evaluate. A `UserWarning` is emitted when this
  is attempted. The `_matching_iou_threshold` field on `DetectionTable`
  tracks the matcher's original threshold.

## Known Issues

- `ContourMatcher` is the most complex matcher and carries legacy coupling
  to the heatmap+mask input format. It should be further decoupled.
- Bbox matching in Rust converts to contours as an intermediate step. This
  works correctly but is suboptimal for axis-aligned boxes where intersection
  can be computed directly.
