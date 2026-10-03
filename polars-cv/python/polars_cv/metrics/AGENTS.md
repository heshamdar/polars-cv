# AGENTS.md — Metrics Subsystem (`polars_cv.metrics`)

> Read the [root AGENTS.md](../../../../AGENTS.md) and [`polars_cv/AGENTS.md`](../AGENTS.md) first.
> Update this file when metric APIs or behavior change.

## Purpose

Detection metrics built from polars-cv primitives and Polars lazy expressions:

- **PR** (weighted by `image_metadata.weight`, `sample_weight` semantics):
  `precision_recall_curve`, `average_precision`, `mean_average_precision`
- **Threshold** (weighted): `precision_at_threshold`, `recall_at_threshold`, `f1_at_threshold`, `confusion_at_threshold` (raw `tp/fp/fn` counts plus `weighted_*` masses; its derived rates read the masses)
- **FROC/LROC** (expression-valued, lazy, group-aware): `froc_auc`/`lroc_auc`
  (LazyFrame, one row per group), `froc_curve_lazy`/`lroc_curve_lazy`,
  `froc_sensitivity_at_fp`/`lroc_sensitivity_at_fpf`, `froc_summary_table`
- **Bootstrap CIs** (lazy, group-aware, seed-reproducible): `froc_auc_ci_lazy`,
  `lroc_auc_ci_lazy`, `average_precision_ci_lazy` — each returns a `LazyFrame`
  `[*group_by, <metric>, ci_lower, ci_upper]` and never collects internally
- **AUC integrals**: the single authority is `_auc_expr.py`
  (`trapz_auc`, `partial_auc`, `collapse_curve`; the weighted Mann-Whitney
  two-stage `collapse_scores` + `mann_whitney_auc`; and the lazy
  `interpolate_curve_lazy`). Each integral takes a lazy curve and `keys` and
  returns `[*keys, auc]` (one row with no keys). Every AUC — `froc_auc`/`lroc_auc`
  and the PR-curve `MetricResult.auc` (the *only* method `MetricResult` exposes) —
  reduces through them and collects streaming; there is no eager Series
  integral. They were `pl.Expr` reductions for `group_by().agg()`; a sort inside
  an aggregation is not native to the streaming engine, so they are now grouped
  scans plus exact sums (see **Streaming** below). `_auc.py` was gutted to just the `correction`
  vocabulary (`CorrectionMethod` + `validate_correction`) once the eager
  `trapz_auc`/`partial_auc` had no consumer left. The `correction` vocabulary is
  `"normalize"` or `None`;
  `validate_correction` (in `_auc.py`, imported by `_auc_expr.py`) is its single
  authority and *rejects* anything else instead of degrading to the raw area.
  McClish's standardized ROC partial-AUC correction was removed — it standardizes
  against the `y = x` chance diagonal on the unit square, valid only when the
  x-axis is a probability in `[0, 1]`; FROC's axis (FP/image) is unbounded, LROC's
  chance line is not the diagonal, and PR's is horizontal at prevalence. Its
  removal is recorded in the CHANGELOG
- **Weighted Mann-Whitney**: `froc_auc`/`lroc_auc(method="mann_whitney")` are
  weighted by `image_metadata.weight` (both `level="detection"` and
  `level="image"`), via `collapse_scores` (bucket by distinct score, carrying the
  positive/negative weight mass) then `mann_whitney_auc` (weighted rank-sum).
  A pure `rank("average")` reduction can't weight ties; bucketing removes them.
  Unit weights recover the standard tie-averaged MW — one implementation, not two,
  cross-checked against the pairwise `ref_weighted_mann_whitney` oracle

## Architecture

```
object tables ─► match_detections ─┐
heatmaps/masks ─► ContourMatcher ──┼─► DetectionTable ─► Statistic ─► value / by_group / bootstrap_ci
pre-matched ───► PreMatchedAdapter ┘                        ▲
                                     DetectionReport ───────┘   (evaluate_detections / _heatmaps)
```

Four layers; each upper one only composes the one below, never adds an
estimator, matcher or resampler of its own:

- **L0 inputs** (`_inputs.py`): `group_objects` (long object rows → one row per
  (image, class) with lists, population-complete), `bbox_from_coords`
  (`geometry/coords.py`).
- **L1 matching**: the matchers, `match_detections` (dispatch on the geometry
  dtype). Both matchers end in the one shared tail, `_matching/_table.py`
  (`matched_table` / `explode_matches`): never re-implement the explode or the
  metadata — the class-join duplication fix lives there.
- **L2 statistics** (`_statistics.py`): every metric is a `Statistic` with
  `by_group(table, keys) -> [*keys, name]`; `bootstrap_ci(table, statistic)` is
  the one CI engine (`*_ci_lazy` are thin delegations). **A new metric is a
  `Statistic` first**; its function form is its ungrouped reading.
  `tests/test_statistics.py::test_every_public_metric_is_classified` fails on a
  public function taking a `DetectionTable` that is neither a `Statistic`'s
  reading nor declared otherwise.
- **L3 reports** (`_reports.py`, `_segmentation.py`): pick statistics and show
  them; `report.metrics` maps each shown number to its statistic, which is what
  `report.ci` bootstraps.

1. **Matchers** (`_matching/`) convert raw data into a canonical `DetectionTable` (two lazy frames). All implement the `Matcher` protocol. `ContourMatcher.match` also accepts a pre-decoded `LazyPipelineExpr` (via `_SourceHandle`) so a caller's graph can share the decode.
2. **FROC/LROC metric functions** (`_metrics/`) operate on `DetectionTable` and return a `pl.LazyFrame` (`froc_auc`, `froc_curve_lazy`, …) — no result object, no eager `.item()` until the caller collects. The integral is the reusable expression in `_auc_expr.py`.
3. **PR / Confusion** still return `MetricResult` subclasses (`_result.py`), whose only method is `auc()` (eager PR-curve AUC, with optional partial-AUC range + correction). The FROC/LROC curve helpers do **not** go through `MetricResult`: `froc_sensitivity_at_fp` / `froc_summary_table` / `lroc_sensitivity_at_fpf` build on the lazy `_auc_expr.interpolate_curve_lazy` authority directly and return `LazyFrame`s (the caller collects). Confidence intervals are the free `*_ci_lazy` functions in `_bootstrap.py`, not a result-object method.

### DetectionTable (`_types.py`)

Two aligned lazy frames:
- **detections** — one row per detection: `image_id`, `class_id`, `score`, `is_tp`, `gt_idx`, `iou`, `det_idx`
- **image_metadata** — one row per (image, class): `n_gts`, `weight`, `gt_label`

Supports IoU re-thresholding via `at_iou_threshold()`, class filtering via `filter_class()`, per-image aggregation via `to_per_image()`.

**Sweeps.** A matcher given several IoU thresholds returns one table holding a
matching per threshold, told apart by an `iou_threshold` column on both frames
(`_sweep` / `iou_thresholds`). Pooling them counts every GT once per threshold,
so `DetectionTable.frames(keys)` — the one place every metric reads — refuses a
sweep unless `iou_threshold` is among the keys; `.detections` /
`.image_metadata` are `frames()` and refuse it too. Transforms that keep every
matching apart (filters, the bootstrap draw) read `_all_rows()`. Per-image work
keys on `_sweep_key()` as well as `(image_id, class_id)`.

## Matchers

| Matcher | Input | Uses |
|---------|-------|------|
| `ContourMatcher` | heatmap + binary mask (blob, list, or array) | `Pipeline().threshold().extract_contours()`, `contour.correspond()` |
| `BBoxMatcher` | `List[BBOX_SCHEMA]` columns | `bbox.correspond()`, ordered by confidence here |
| `PreMatchedAdapter` | pre-computed TP/FP per detection | Direct DataFrame wrapping |

## AUC API

- **FROC/LROC**: `froc_auc(table, *, method, fp_range, correction, level, group_by)`
  → `pl.LazyFrame` (`[*group_by, auc]`). `method="trapezoidal"` (default)
  **requires** `fp_range` for FROC (its FP/image axis is unbounded, so there is no
  reasonable default window — it raises without one) and defaults `fpf_range` to
  the full `(0.0, 1.0)` FPF domain for LROC. `correction` defaults to
  `"normalize"` (mean sensitivity over the window; pass `None` for the raw
  partial area). `method="mann_whitney"` (`level="detection"|"image"`) is a
  global rank statistic that rejects a range (correction is inert for it). A
  scalar is `froc_auc(table, fp_range=(0.0, 8.0)).collect().item()`; grouping is
  `group_by=`.
- **PR**: `PrecisionRecallResult.auc(method=...)` — `"all_points"` (default:
  the precision envelope integrated as a **step** function, `Σ ΔR·P̂`),
  `"11_point"` (VOC 2007), `"101_point"` (COCO), `"trapezoidal"` (raw). Every
  AP — scalar, grouped, bootstrap — integrates in `ap_from_points`; the N-point
  grids are `numpy.linspace`'s (`i · 1/(n−1)`), as the reference tools build
  them. Reductions feeding a reported number sum with
  `_grouped_scan.exact_sums` / `exact_mean`: an Int128 fixed-point sum, which
  does not depend on chunking, order or thread count, so bootstrap bounds
  reproduce bit for bit. It also stays on the streaming engine; a sorted
  `cum_sum` inside `agg` does not.

## Streaming

Every metrics plan stays on polars' native streaming engine. A step it cannot
run natively becomes an `in-memory-map` node that collects its whole input,
and the bootstrap's input is `n_bootstrap × detections` rows.
`tests/test_streaming_plans.py` builds every public lazy entry point and fails
on any such node (or a whole-column Python UDF) that `KNOWN_FALLBACKS` in
`tests/_streaming_guard.py` does not name, with its reason.

- **No `.over()` scans, and no sort inside `agg`.** A running sum, running max,
  lag, row index or first/last flag per group goes through
  `_grouped_scan.grouped_scan`. It sorts once by `(*keys, *by)`, runs each scan
  over the whole frame, and restarts it at group starts exactly: integer and
  Int128 fixed-point sums, rank-encoded maxima. Reductions that must be
  reproducible go through `exact_sums` / `exact_mean` (Int128 fixed point,
  NaN/±inf combined as float addition would). Aggregations that are already
  native stay as they are: `sum`, `max`, `len`, `count`, `any`, and
  `sum().over()` / `len().over()`, which polars rewrites into a group-by plus a
  join.
- **What still falls back** (all `KNOWN_FALLBACKS`, each on a per-object or
  per-image frame, never the replicate frame): `group_objects`' per-(image,
  class) object lists, which the elementwise matchers need; a sampling unit's
  set of weight cells (`_cell`); and an entity's image list (`sample_col=`).
  polars cannot build a list inside a streaming group-by.

## Bootstrap CIs

Seed-reproducible, group-aware, and **fully lazy** — the entire bootstrap
(resample, per-replicate metric, and the percentile bounds) is one Polars plan
the caller collects. Three free functions in `_bootstrap.py` are the only way in:

```python
froc_auc_ci_lazy(table, n_bootstrap=1000, seed=42)  # detection AUC
froc_auc_ci_lazy(table, group_by="group_id", seed=42)  # per-group CI
froc_auc_ci_lazy(table, group_by="group_id", sample_col="case_id")  # entity-level
froc_auc_ci_lazy(table, method="mann_whitney")  # MW AUC
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

**Every CI is weighted and stratifies on weight cells** (`_replicate_table`,
the one way replicates are built). `_sampling_units` is the single authority for
the resample's base: one row per sampling unit and group.

- At image level, a unit's `gt_label` stratum is positive if **any** of its
  `(image, class)` rows is. A mixed-label multi-class image used to sit in both
  strata, so it had two draw slots per replicate.
- Its `_cell` is the sorted distinct weight clusters (or `(*strata, cluster)`
  structs) of its rows, or of its images for an entity.
- `_with_weight_clusters` assigns each metadata row a cluster within
  `(group, *strata)` by relative gap (`weight_rtol`, default `1e-6`), with no
  rounding boundary. It is one sort plus a window, with no self-join.

The draw is stratified within `(group, gt_label, cell)` at image level, or
`(group, cell)` at entity level. Each draw carries its partition keys, and
`_bootstrap_table_with_draws` joins the metadata on `[image_id, *group_keys]`
and the detections on the group keys they carry. A draw therefore brings only
its own group's rows: under `group_by="class_id"` a draw used to bring the
image's other classes too.

At entity level, `_rescale_to_cell_shares` multiplies each drawn image's weight
by `(n_c/N) / (n*_c/N*)`, which restores each cell's share of the group's
images. That is the `p / q̂` weight re-estimated on the replicate. At image
level the factor would be exactly 1, so it is not applied.

Every weighted statistic is a weight-scale-invariant ratio, so the replicate
weights equal the re-estimated weights. That is why there is no reweight hook.
Unit weights form one cell and leave the draw and the weights bit-identical to
the unweighted resample.

**The units and the draw are `.cache()`-d** in `_replicate_table`. They stay
lazy, but without the cache the streaming engine re-runs the draw at every read
of the replicate frames, about 4x slower, because projection pushdown makes
each read a distinct subplan. This needs polars >= 1.44.2, the declared floor:
1.43.2 returned wrong rows from a cached frame read under different
projections, which put AP bounds above 1 in the floors CI job.

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
undefined without both classes. A group with any weight cell of size 1 is also
non-viable: that cell has no bootstrap variance,
and a continuous weight (all singletons) would otherwise report a zero-width
interval. That viability rule is the one behavioral choice worth knowing.

## File Layout

```
metrics/
├── __init__.py           # Public re-exports
├── _types.py             # DetectionTable, column constants, schema validation
├── _result.py            # MetricResult base (auc only) — PR/Confusion
├── _auc.py               # the correction and extrapolate vocabularies (validate_*)
├── _auc_expr.py          # the FROC/LROC integral authority: *_expr + collapse_curve
├── _bootstrap.py         # bootstrap_ci (+ the *_ci_lazy delegations) + lazy resampler
├── _statistics.py        # Statistic, AP/Recall/…/CPM/FROCAUC/LROCAUC, MeanOver, mean_ap
├── _inputs.py            # group_objects, match_detections (object tables → DetectionTable)
├── _reports.py           # evaluate_detections/_heatmaps → DetectionReport
├── _segmentation.py      # segmentation_measures, evaluate_segmentation → SegmentationReport
├── _matching/
│   ├── _protocol.py      # Matcher protocol
│   ├── _table.py         # the shared tail: thresholds, explode, metadata, sweeps
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

### All-points AP: two authorities, one estimator
- The all-points AP estimator (sort by score desc, cumulative TP/FP, monotone
  precision envelope, step integral anchored at recall 0) exists as a scalar (`_all_points_ap`,
  behind `PrecisionRecallResult.auc("all_points")`) and a vectorized grouped
  form (`all_points_ap_by_group`). Both `mean_average_precision(...,
  interpolation="all_points")` and every `*_ap` bootstrap go through the grouped
  authority; the eager per-class PR result uses the scalar one.
- The two are **identical on every curve, ties included** (pinned by
  `test_precision_recall.py::TestAllPointsAPAuthority` and, end to end, by
  `test_bootstrap_ci_lazy.py::test_pr_point_matches_average_precision`). Both
  apply the same canonical tie convention (CR-30): all detections sharing a
  score collapse into a single PR point, taken at the cumulative TP/FP *after*
  the whole tied block. So the all-points AP no longer depends on the input row
  order among equal scores, and the scalar and grouped paths no longer diverge.
  They remain two functions only because they take different inputs (a pre-built
  curve vs per-detection rows), not because they can disagree.
- Both are weighted, and both share `_score_buckets`: per-score TP/FP counts
  and weighted masses. Each accumulates the weighted masses, with precision
  `Σw·tp / Σw·(tp+fp)` and recall `Σw·tp / gt_mass`. Zero-weight detections are
  dropped first. `gt_mass` comes from `_weights.weighted_gt_mass`
  (`Σ n_gts · w`), the recall denominator for every PR metric; the grouped
  authority's input carries `weight` and `gt_mass` columns. The integer-weight
  replication oracle (`TestWeightedPrecisionRecall`) checks every PR entry
  point.
- `mean_average_precision` is `MeanOver(AP(interpolation), undefined="zero")`
  over a table stacked across its thresholds (a sweep's own matchings, or
  re-thresholded copies of one matching): one lazy plan, every interpolation.
  `undefined="zero"` keeps its historical convention (a class without GT
  averages in as 0); the reports use COCO's `"exclude"`.

### Null and edge-case handling
- Contour extraction returns an empty list when an image has no contours, and `null` only for a null input or a failed row. Matchers keep `.fill_null(0)` on `list.len()` for `n_gts`, which now only affects those null rows.
- Zero-score contours are filtered *before* matching via `_filter_zero_score_detections` to prevent them from claiming GT objects.
- `extract_contours` traces along pixel edges, so every extracted region — one pixel included — has an area equal to its pixel count and overlaps ground truth exactly as its pixels do. `label_reduce` scores a zero-area or sub-pixel contour (only user-supplied ones can be) on the pixels its outline passes through, never as 0.0.

### ContourMatcher behavior
- `min_contour_area` defaults to 0.0 and `gt_min_contour_area` to 1.0, independently. The smallest extracted region (one pixel) has area 1, so both defaults keep every region.
- Auto-detects source format (`Binary`/`List`/`Array`) from column dtypes via `_detect_source_info`.
- `auto_resize=True` (default) resizes predictions to GT dimensions via fused pipeline. `auto_resize=False` assumes shapes match.
- Three `label_reduce` region modes: `"interior"` (default), `"boundary"` (interior + boundary pixels), `"bbox"`.

### Sorting and aggregation
- `to_per_image()` picks the top detection with a `grouped_scan` over `(score desc, det_idx)` (`IsFirst`), not `sort_by()` within `group_by().agg()`: that is not native to the streaming engine, and it was not stable, so a tied top score was read arbitrarily. Never rely on `.sort()` before `.group_by()` either; polars does not keep the order across `group_by`.
- FROC / LROC curves are returned sorted by **descending `threshold`**, which is
  ascending `fp_per_image` / `fpf` (plotting order). Sort on the threshold, never
  on the x-column: thresholds are unique so the order is total, while `fp_per_image`
  ties constantly and Polars' `sort` defaults to `maintain_order=False`, leaving
  the y at each tie boundary — and therefore the AUC — unspecified.
- Every consumer of a curve's geometry goes through `_auc_expr.collapse_curve`,
  which collapses tied x to the maximum y (the ROC upper envelope) before
  integrating or interpolating. `MetricResult.auc` and the FROC/LROC paths must
  not sort for themselves; a second sort is a second answer.

### IoU re-thresholding vs re-matching
- On a sweep, `at_iou_threshold(t)` for a matched `t` is an exact slice (that
  threshold's own matching); any other `t` re-thresholds the matching at the
  highest matched threshold below it.
- Re-thresholding only works reliably when *raising* the threshold. Lowering has no effect (unmatched detections lack stored IoU). A `UserWarning` is emitted. It is not re-matching: a detection that lost its GT to a duplicate does not get it back — COCO's mAP needs a sweep.

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
- Class keys must agree: a `class_id` column on the detections or on
  `image_meta` requires `class_col`, and `class_col` requires `class_id` on
  `image_meta` (`_check_class_keys`, schema-only so nothing collects). The
  metrics join on `(image_id, class_id)`; a key matching no metadata row counts
  for nothing, which is how `lroc_auc` scored a perfect detector 0.0.
- Every named column must exist — none falls back to its default.
- Without `iou_col` the table's `iou` is null and `has_iou=False`, so
  `at_iou_threshold` (and IoU-swept mAP) raises instead of comparing a
  placeholder 0.0. `DetectionTable` copies go through `dataclasses.replace`, so
  a new field cannot be dropped by one of them.

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
  x-values outside the observed range by default, and the summary's y column
  stays Float64 even when every point is null.
- **One off-curve policy.** `partial_auc` and `interpolate_curve_lazy` are
  the only two curve readers and both take `extrapolate` (`"none"` default,
  `"flat"`; vocabulary in `_auc.py`). They used to disagree — the integral
  filled flat past the curve's end while the interpolation returned null — so
  one table had an unknown sensitivity at 2 FP/image and a defined AUC over
  0–2. Every FROC/LROC entry point and the `*_ci_lazy` functions pass the
  argument through; `froc_operating_range` reports how far a curve reaches.
  LROC appends its `(1, max sensitivity)` endpoint, so its default `[0, 1]`
  window never needs extrapolation.
- **Undefined is not empty in the bootstrap.** `_bootstrap_ci_from_replicates`
  fills an *absent* replicate (an empty draw) with `empty_value`, but a
  replicate *present with a null value* is undefined and nulls its group's
  bounds.
- At an x the curve visits more than once, the *highest* y there is returned:
  `froc_sensitivity_at_fp(table, 0.0).collect().item()` is the sensitivity
  reachable with no false positives, not the origin's zero.

### Line-shaped GT and duplicates (ContourMatcher)
- `match_by="coverage"` pairs through `.contour.correspond_by_coverage` (Rust
  `pairwise::coverage_matrix` into the same `greedy_assign`); `iou_threshold`
  is then the minimum coverage and the `iou` column holds coverage. A
  contour/contour-set `gt_col` (`_is_contour_dtype`) is used as the GT
  contours directly, and needs `auto_resize=False` — refused otherwise, since
  there is no mask size to resize to.
- `duplicates="ignore"` drops detections whose correspondence `duplicate` flag
  is set (`_explode_match_to_detections(ignore_duplicates_col=...)`), instead
  of counting them as false positives.

## Known Issues

- Score + extract cannot be merged into one graph: `label_reduce` requires contours as an expression parameter, so they must exist as a column before the scoring pipeline runs.
