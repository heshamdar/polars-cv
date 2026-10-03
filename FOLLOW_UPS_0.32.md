# Follow-ups from the 0.32.0 release review

Items the 0.32.0 pre-release review (commit `1bd81ca`) found and deliberately
left out of the release: each needs a design decision or widens scope beyond a
release fix. Same conventions as `CODE_REVIEW_FINDINGS.md`: stable IDs, edit
the status as items close, strike through rather than delete.

**Status legend:** `Open` · `In progress` · `Resolved` · `Won't fix (documented)`

| ID | Severity | Summary | Status |
|----|----------|---------|--------|
| FU-01 | Medium | Negative-size boxes are accepted and silently score as false positives | Open |
| FU-02 | Low | Sweep thresholds compared by exact float equality | Open |
| FU-03 | Low | An image missing from `images=` silently gets weight 1.0 | Open |
| FU-04 | Low | The COCO parity test never runs in CI | Open |
| FU-05 | Low | `uv.lock` pins a yanked numpy (2.4.0) | Open |

---

## FU-01 — Negative-size boxes are accepted and silently score as false positives

- **Location:** `polars-cv/python/polars_cv/geometry/coords.py::bbox_from_coords`;
  the bbox reader in `polars-cv/src/geom_columns.rs` (`BBoxColumn`).
- **Evidence:** `bbox_from_coords([[10, 10, 0, 0]], format="xyxy")` gives
  `{x: 10, y: 10, width: -10, height: -10}` with no error. Through
  `match_detections(..., box_format="xyxy")` such a prediction scores as a
  0-IoU false positive. The usual way to get here is a format mistake (xywh
  data passed as `"xyxy"`), which is exactly what the required `format=` was
  meant to catch. NaN coordinates, by contrast, are refused by the reader
  ("a bbox has a non-finite y").
- **Why deferred:** the reader accepts negative `width`/`height` everywhere,
  so refusing them changes every bbox function, not only the new constructor.
  `bbox_from_coords` is a pure polars expression and cannot raise per row
  without the plugin.
- **Proposed fix:** have the bbox reader refuse `width < 0` / `height < 0` the
  way it refuses non-finite fields (decide whether zero is allowed). Add the
  error to the `geom_columns.rs` tests and a `match_detections` test with
  swapped corners. Check first that no internal producer relies on negative
  sizes.

## FU-02 — Sweep thresholds compared by exact float equality

- **Location:** `polars-cv/python/polars_cv/metrics/_types.py`
  (`DetectionTable.at_iou_threshold`, `_slice`) and
  `polars-cv/python/polars_cv/metrics/_reports.py` (`DetectionReport.metrics`,
  `per_class`: `if t in ts`).
- **Evidence:** a sweep matched at `np.arange(0.5, 0.96, 0.05)` stores
  `0.55000000000000004`, `0.6000000000000001`, …. Then
  `at_iou_threshold(0.55)` finds no matched threshold `<= 0.55` except 0.5,
  so it re-thresholds the 0.5 matching instead of slicing the 0.55 one, with
  no warning. `DetectionReport` likewise omits `map_75` / `ap_75` when 0.75
  is stored as `0.7500000000000002`. `COCO_IOU_THRESHOLDS` is rounded, so the
  default path is unaffected.
- **Proposed fix:** normalise thresholds once where the sweep is built (round
  to e.g. 10 decimals in the matcher and `DetectionTable.stack`). Do not add
  tolerant comparisons at every reader. Add a test with `np.arange`
  thresholds that asserts `at_iou_threshold(0.55)` slices the 0.55 matching.

## FU-03 — An image missing from `images=` silently gets weight 1.0

- **Location:** `polars-cv/python/polars_cv/metrics/_inputs.py`
  (`match_detections`, `_image_columns`); policy in
  `polars-cv/python/polars_cv/metrics/_weights.py` (`fill_null(1.0)`).
- **Evidence:** with `images=pl.DataFrame({"image_id": ["a"], "w": [2.0]})`
  and predictions on images `a` and `b`, image `b`'s metadata weight is null
  and every metric reads it as 1.0. An incomplete weight table is
  indistinguishable from a deliberate unit weight.
- **Why deferred:** null → 1.0 is the existing, documented policy for missing
  weights across the matchers and `attach_resolved_weight`. It predates 0.32.
  Changing it is a behaviour change for every weighted entry point.
- **Proposed fix:** decide the policy. Either (a) under `weight=`, require
  the `images=` frame to cover every image in either table and fail the
  query otherwise (e.g. an anti-join count fed through an existing lazy
  refusal), or (b) keep 1.0 and say so explicitly in the `weight=`
  docstrings of `match_detections` / `evaluate_detections`.

## FU-04 — The COCO parity test never runs in CI

- **Location:** `polars-cv/tests/reference/test_coco_parity_ref.py`;
  `polars-cv/pyproject.toml` dev group; `.github/workflows/`.
- **Evidence:** the test skips when `pycocotools` is not importable, and
  nothing installs it, so the changelog's "agrees with COCOeval to 1e-12"
  claim is not checked by CI. Run by hand
  (`uv run --with pycocotools pytest tests/reference/test_coco_parity_ref.py`)
  it passes 4/4 at 0.32.0.
- **Proposed fix:** add `pycocotools` to the dev (or a reference-test)
  dependency group so the fast lane runs it. Alternatively, make the reference
  lane fail rather than skip when it is missing in CI (a skip there reads as
  green).

## FU-05 — `uv.lock` pins a yanked numpy (2.4.0)

- **Location:** `polars-cv/uv.lock`.
- **Evidence:** every `uv lock` / `uv run --with …` warns
  ``numpy==2.4.0 is yanked (reason: "Backward compatibility bug")``. This
  predates 0.32.0 (the 0.32.0 relock changed only the project version).
- **Proposed fix:** `uv lock --upgrade-package numpy` and run
  `scripts/verify.sh`. The `numpy>=2.0.2` floor in `pyproject.toml` is
  unaffected.
