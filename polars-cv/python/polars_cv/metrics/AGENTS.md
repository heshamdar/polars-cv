# AGENTS.md — Metrics Subsystem (`polars_cv.metrics`)

> Read the root `AGENTS.md` and `polars-cv/python/polars_cv/AGENTS.md` first.
> Update this file when metric APIs or behavior change.

## Purpose

This subpackage provides higher-level detection metrics built from existing
polars-cv primitives plus Polars expressions:

- FROC (Free-response ROC)
- LROC (Localization ROC)
- Bootstrap confidence intervals
- AUC and partial AUC utilities

## Design

- Input in v1 is heatmap + binary mask only.
- Both analyzers share one lazy preparation path via `_prepare.prepare_detection_table(...)`.
- Contours are extracted internally using `Pipeline().threshold().extract_contours()`.
- Matching uses `pl.Expr.contour.match_detections()`.
- Scoring uses buffer-space `Pipeline().label_reduce(contours=...)` with
  `reduction="max"` and `region_mode="interior"`.
- All curve aggregation is done with Polars expressions (`explode`/`group_by`), not
  Python loops.
- Score/match alignment is strict on match payloads: `pred_idx` and `gt_idx` list
  lengths must match; analyzers raise `ValueError` otherwise.
- Materialization points use `collect(engine="streaming")`.
- LROC positive-target validation uses extracted GT contour counts; GT contour
  extraction applies a floor of `min_area >= 1.0` to avoid noise fragments.

## Primitive Assumptions

- `extract_contours().sink("native")` returns contour sets (`List[Contour]`).
- Contour schema fields are `exterior`, `holes`, and `is_closed`.
- Contour-space `.contour.label_reduce(image=...)` remains available as a
  compatible alternative scoring path.

## Files

- `_prepare.py`: shared lazy detection table, shape handling, contour extraction/scoring
- `_froc.py`: `FROCAnalyzer`, `FROCResult`
- `_lroc.py`: `LROCAnalyzer`, `LROCResult`
- `_bootstrap.py`: `BootstrapResult`, image-level bootstrap helper
- `_auc.py`: AUC and partial AUC helpers
