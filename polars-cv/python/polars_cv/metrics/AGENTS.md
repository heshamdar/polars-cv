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

### IoU re-thresholding
- `at_iou_threshold()` only works reliably when *raising* the threshold. Lowering has no effect (unmatched detections lack stored IoU). A `UserWarning` is emitted.

### LROC variants
- `"best_tp"` (default): effective score = highest-scoring TP detection for positive images.
- `"top_scoring"`: effective score = single highest-scoring detection regardless of TP/FP (classical Swensson 1996).

## Known Issues

- Bbox matching converts to contours internally. Correct but suboptimal for axis-aligned boxes.
- Score + extract cannot be merged into one graph: `label_reduce` requires contours as an expression parameter, so they must exist as a column before the scoring pipeline runs.
