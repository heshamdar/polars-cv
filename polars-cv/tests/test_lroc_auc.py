"""Correctness + group-awareness tests for the lazy, expression-valued LROC AUC.

Mirrors ``test_froc_auc.py``: builds ``DetectionTable``s from literal frames and
asserts ``lroc_auc(table).collect().item()`` matches an independent NumPy
reference (:mod:`tests._metric_refs`) across methods, and that a grouped result
equals the per-group AUC on the filtered sub-table.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.metrics import (
    DetectionTable,
    lroc_auc,
    lroc_sensitivity_at_fpf,
)
from polars_cv.metrics._types import (
    COL_CLASS_ID,
    COL_DET_IDX,
    COL_GT_IDX,
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IOU,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
)
from tests._metric_refs import (
    ref_froc_mw_detection,
    ref_lroc_auc,
    ref_lroc_image_mw,
    ref_lroc_sensitivity_at_fpf,
)

_TOL = 1e-9

# (image, region, score, is_tp) — one row per detection; empty images omitted.
_DETS = [
    ("a", "g1", 0.90, True),
    ("a", "g1", 0.40, False),
    ("b", "g2", 0.50, False),  # positive image, no TP
    ("c", "g1", 0.70, False),  # negative image with a detection
    ("e", "g2", 0.65, True),
    ("e", "g2", 0.30, False),
]
# (image, region, gt_label, weight)
_IMAGES = [
    ("a", "g1", True, 1.0),
    ("b", "g2", True, 1.2),
    ("c", "g1", False, 0.8),
    ("d", "g2", False, 1.0),  # negative image, no detections
    ("e", "g2", True, 0.9),
]


def _table() -> DetectionTable:
    det = pl.DataFrame(
        {
            COL_IMAGE_ID: [r[0] for r in _DETS],
            COL_CLASS_ID: ["__all__"] * len(_DETS),
            COL_SCORE: [r[2] for r in _DETS],
            COL_IS_TP: [r[3] for r in _DETS],
            COL_GT_IDX: [0 if r[3] else None for r in _DETS],
            COL_IOU: [0.7 if r[3] else 0.0 for r in _DETS],
            COL_DET_IDX: list(range(len(_DETS))),
        },
        schema={
            COL_IMAGE_ID: pl.String,
            COL_CLASS_ID: pl.String,
            COL_SCORE: pl.Float64,
            COL_IS_TP: pl.Boolean,
            COL_GT_IDX: pl.UInt32,
            COL_IOU: pl.Float64,
            COL_DET_IDX: pl.UInt32,
        },
    )
    meta = pl.DataFrame(
        {
            COL_IMAGE_ID: [r[0] for r in _IMAGES],
            COL_CLASS_ID: ["__all__"] * len(_IMAGES),
            COL_N_GTS: [1 if r[2] else 0 for r in _IMAGES],
            COL_WEIGHT: [r[3] for r in _IMAGES],
            COL_GT_LABEL: [r[2] for r in _IMAGES],
            "region": [r[1] for r in _IMAGES],
        },
        schema={
            COL_IMAGE_ID: pl.String,
            COL_CLASS_ID: pl.String,
            COL_N_GTS: pl.Int64,
            COL_WEIGHT: pl.Float64,
            COL_GT_LABEL: pl.Boolean,
            "region": pl.String,
        },
    )
    return DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)


def _sub_table(region: str) -> DetectionTable:
    """The sub-table for one region (detections filtered by that region's images)."""
    table = _table()
    meta = table.image_metadata.filter(pl.col("region") == region)
    ids = meta.select(COL_IMAGE_ID)
    det = table.detections.join(ids, on=COL_IMAGE_ID, how="semi")
    return DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)


def _v(lf: pl.LazyFrame) -> float:
    return lf.collect().item()


class TestLrocAucParity:
    @pytest.mark.parametrize("variant", ["best_tp", "top_scoring"])
    def test_trapezoidal_raw(self, variant: str) -> None:
        table = _table()
        got = _v(lroc_auc(table, variant=variant))
        want = ref_lroc_auc(table, variant=variant)
        assert got == pytest.approx(want, abs=1e-9)

    @pytest.mark.parametrize("fpf_range", [(0.0, 0.5), (0.0, 1.0), (0.25, 1.0)])
    def test_partial(self, fpf_range: tuple[float, float]) -> None:
        table = _table()
        got = _v(lroc_auc(table, fpf_range=fpf_range))
        want = ref_lroc_auc(table, fpf_range=fpf_range)
        assert got == pytest.approx(want, abs=1e-7)

    def test_mann_whitney_image(self) -> None:
        table = _table()
        got = _v(lroc_auc(table, method="mann_whitney", level="image"))
        want = ref_lroc_image_mw(table)
        assert got == pytest.approx(want, abs=1e-9)

    def test_mann_whitney_detection(self) -> None:
        table = _table()
        got = _v(lroc_auc(table, method="mann_whitney", level="detection"))
        want = ref_froc_mw_detection(table)  # same P(TP > FP) over detections
        assert got == pytest.approx(want, abs=1e-9)


class TestLrocAucGroupParity:
    def test_trapezoidal_group_by_region(self) -> None:
        table = _table().with_group("region")
        grouped = lroc_auc(table, group_by="group_id").collect()
        got = dict(zip(grouped["group_id"].to_list(), grouped["auc"].to_list()))
        for region in ("g1", "g2"):
            want = _v(lroc_auc(_sub_table(region)))
            assert got[region] == pytest.approx(want, abs=_TOL)

    def test_mann_whitney_image_group_by_region(self) -> None:
        table = _table().with_group("region")
        grouped = lroc_auc(
            table, method="mann_whitney", level="image", group_by="group_id"
        ).collect()
        got = dict(zip(grouped["group_id"].to_list(), grouped["auc"].to_list()))
        for region in ("g1", "g2"):
            want = _v(
                lroc_auc(_sub_table(region), method="mann_whitney", level="image")
            )
            assert got[region] == pytest.approx(want, abs=_TOL)

    def test_mann_whitney_detection_group_by_region(self) -> None:
        # group_id lives only on the metadata; the detection-level path must join
        # it onto the detections before grouping (regression: it used to crash).
        table = _table().with_group("region")
        grouped = lroc_auc(
            table, method="mann_whitney", level="detection", group_by="group_id"
        ).collect()
        got = dict(zip(grouped["group_id"].to_list(), grouped["auc"].to_list()))
        for region in ("g1", "g2"):
            want = _v(
                lroc_auc(_sub_table(region), method="mann_whitney", level="detection")
            )
            assert got[region] == pytest.approx(want, abs=_TOL)


class TestLrocStandaloneHelpers:
    @pytest.mark.parametrize("fpf", [0.0, 0.25, 0.5, 1.0])
    def test_sensitivity_at_fpf(self, fpf: float) -> None:
        table = _table()
        got = lroc_sensitivity_at_fpf(table, fpf)
        want = ref_lroc_sensitivity_at_fpf(table, fpf)
        assert got == want or got == pytest.approx(want, abs=1e-9)
