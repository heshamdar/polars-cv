"""Correctness + group-awareness tests for the lazy, expression-valued FROC AUC.

Builds ``DetectionTable``s from literal frames and asserts:

* ``froc_auc(table).collect().item()`` matches an independent NumPy reference
  (:mod:`tests._metric_refs`) for every method/range/correction, and
* ``froc_auc(table, group_by="class_id")`` per class equals ``froc_auc`` on
  ``table.filter_class(cid)`` — the property that replaces a per-group loop.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.metrics import (
    DetectionTable,
    froc_auc,
    froc_sensitivity_at_fp,
    froc_summary_table,
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
    ref_froc_auc,
    ref_froc_mw_detection,
    ref_froc_sensitivity_at_fp,
)

_TOL = 1e-9


def _table(*, multiclass: bool) -> DetectionTable:
    """A small two-class detection table with non-trivial weights."""
    # (image, class, score, is_tp)
    rows = [
        ("a", "x", 0.90, True),
        ("a", "x", 0.40, False),
        ("b", "x", 0.80, True),
        ("b", "x", 0.30, False),
        ("c", "x", 0.70, False),
        ("a", "y", 0.85, True),
        ("b", "y", 0.60, False),
        ("c", "y", 0.55, True),
        ("c", "y", 0.20, False),
    ]
    if not multiclass:
        rows = [(i, "__all__", s, tp) for (i, _c, s, tp) in rows]

    det = pl.DataFrame(
        {
            COL_IMAGE_ID: [r[0] for r in rows],
            COL_CLASS_ID: [r[1] for r in rows],
            COL_SCORE: [r[2] for r in rows],
            COL_IS_TP: [r[3] for r in rows],
            COL_GT_IDX: [0 if r[3] else None for r in rows],
            COL_IOU: [0.7 if r[3] else 0.0 for r in rows],
            COL_DET_IDX: list(range(len(rows))),
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

    # one metadata row per (image, class); weights vary by image
    weights = {"a": 1.0, "b": 1.2, "c": 0.8}
    classes = ["x", "y"] if multiclass else ["__all__"]
    meta_rows = [
        (img, cls, 1, weights[img], True) for img in ("a", "b", "c") for cls in classes
    ]
    meta = pl.DataFrame(
        {
            COL_IMAGE_ID: [r[0] for r in meta_rows],
            COL_CLASS_ID: [r[1] for r in meta_rows],
            COL_N_GTS: [r[2] for r in meta_rows],
            COL_WEIGHT: [r[3] for r in meta_rows],
            COL_GT_LABEL: [r[4] for r in meta_rows],
        },
        schema={
            COL_IMAGE_ID: pl.String,
            COL_CLASS_ID: pl.String,
            COL_N_GTS: pl.Int64,
            COL_WEIGHT: pl.Float64,
            COL_GT_LABEL: pl.Boolean,
        },
    )
    return DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)


def _auc_value(lf: pl.LazyFrame) -> float:
    return lf.collect().item()


class TestFrocAucParity:
    """froc_auc matches an independent NumPy reference for a pooled table."""

    def test_trapezoidal_raw(self) -> None:
        table = _table(multiclass=False)
        got = _auc_value(froc_auc(table))
        want = ref_froc_auc(table)
        assert got == pytest.approx(want, abs=1e-9)

    @pytest.mark.parametrize("fp_range", [(0.0, 1.0), (0.25, 2.0), (0.0, 8.0)])
    def test_partial(self, fp_range: tuple[float, float]) -> None:
        table = _table(multiclass=False)
        got = _auc_value(froc_auc(table, fp_range=fp_range))
        want = ref_froc_auc(table, fp_range=fp_range)
        assert got == pytest.approx(want, abs=1e-7)

    @pytest.mark.parametrize("correction", ["normalize", "mcclish"])
    def test_partial_corrections(self, correction: str) -> None:
        table = _table(multiclass=False)
        got = _auc_value(froc_auc(table, fp_range=(0.25, 2.0), correction=correction))
        want = ref_froc_auc(table, fp_range=(0.25, 2.0), correction=correction)
        assert got == pytest.approx(want, abs=1e-7)

    def test_mann_whitney_detection(self) -> None:
        table = _table(multiclass=False)
        got = _auc_value(froc_auc(table, method="mann_whitney"))
        want = ref_froc_mw_detection(table)
        assert got == pytest.approx(want, abs=1e-9)


class TestFrocAucGroupParity:
    """Grouped AUC equals per-group AUC on the filtered sub-table."""

    def test_trapezoidal_group_by_class(self) -> None:
        table = _table(multiclass=True)
        grouped = froc_auc(table, group_by="class_id").collect()
        got = dict(zip(grouped[COL_CLASS_ID].to_list(), grouped["auc"].to_list()))

        for cid in ("x", "y"):
            want = froc_auc(table.filter_class(cid)).collect().item()
            assert got[cid] == pytest.approx(want, abs=_TOL)

    def test_mann_whitney_group_by_class(self) -> None:
        table = _table(multiclass=True)
        grouped = froc_auc(table, method="mann_whitney", group_by="class_id").collect()
        got = dict(zip(grouped[COL_CLASS_ID].to_list(), grouped["auc"].to_list()))

        for cid in ("x", "y"):
            want = (
                froc_auc(table.filter_class(cid), method="mann_whitney")
                .collect()
                .item()
            )
            assert got[cid] == pytest.approx(want, abs=_TOL)

    def test_grouped_has_one_row_per_class(self) -> None:
        table = _table(multiclass=True)
        grouped = froc_auc(table, group_by="class_id").collect()
        assert sorted(grouped[COL_CLASS_ID].to_list()) == ["x", "y"]
        assert grouped.height == 2


class TestFrocStandaloneHelpers:
    """The lazy standalone helpers match the NumPy reference."""

    @pytest.mark.parametrize("fp", [0.0, 0.25, 0.5, 1.0, 2.0, 100.0])
    def test_sensitivity_at_fp(self, fp: float) -> None:
        table = _table(multiclass=False)
        got = froc_sensitivity_at_fp(table, fp)
        want = ref_froc_sensitivity_at_fp(table, fp)
        assert got == want or got == pytest.approx(want, abs=1e-9)

    def test_summary_table_interpolates_the_curve(self) -> None:
        table = _table(multiclass=False)
        got = froc_summary_table(table, fp_rates=[0.25, 0.5, 1.0])
        assert got.columns == ["fp_per_image", "sensitivity"]
        assert got["fp_per_image"].to_list() == [0.25, 0.5, 1.0]
        for fp, sens in zip(
            got["fp_per_image"].to_list(), got["sensitivity"].to_list()
        ):
            want = ref_froc_sensitivity_at_fp(table, fp)
            if want is None:
                assert sens is None
            else:
                assert sens == pytest.approx(want, abs=1e-9)
