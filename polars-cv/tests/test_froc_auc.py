"""Correctness + group-awareness tests for the lazy, expression-valued FROC AUC.

Builds ``DetectionTable``s from literal frames and asserts:

* ``froc_auc(table, fp_range=...).collect().item()`` matches an independent
  NumPy reference (:mod:`tests._metric_refs`) for every range/correction, and
* ``froc_auc(table, group_by="class_id", fp_range=...)`` per class equals
  ``froc_auc`` on ``table.filter_class(cid)`` — the property that replaces a
  per-group loop.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.metrics import (
    DetectionTable,
    froc_auc,
    froc_curve_lazy,
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

    def test_trapezoidal_requires_fp_range(self) -> None:
        """method='trapezoidal' with no fp_range raises, not silently degrades."""
        table = _table(multiclass=False)
        with pytest.raises(ValueError, match="requires an explicit fp_range"):
            froc_auc(table)

    @pytest.mark.parametrize("extrapolate", ["none", "flat"])
    @pytest.mark.parametrize("fp_range", [(0.0, 1.0), (0.25, 2.0), (0.0, 8.0)])
    def test_partial_raw(self, fp_range: tuple[float, float], extrapolate: str) -> None:
        table = _table(multiclass=False)
        got = _auc_value(
            froc_auc(table, fp_range=fp_range, correction=None, extrapolate=extrapolate)
        )
        want = ref_froc_auc(table, fp_range=fp_range, extrapolate=extrapolate)
        assert got == want or got == pytest.approx(want, abs=1e-7)

    @pytest.mark.parametrize("fp_range", [(0.0, 1.0), (0.25, 2.0), (0.0, 8.0)])
    def test_normalize_is_the_default(self, fp_range: tuple[float, float]) -> None:
        """The default correction is 'normalize' (mean sensitivity over fp_range)."""
        table = _table(multiclass=False)
        default = _auc_value(froc_auc(table, fp_range=fp_range, extrapolate="flat"))
        explicit = _auc_value(
            froc_auc(
                table, fp_range=fp_range, correction="normalize", extrapolate="flat"
            )
        )
        want = ref_froc_auc(table, fp_range=fp_range, correction="normalize")
        assert default == pytest.approx(explicit, abs=_TOL)
        assert default == pytest.approx(want, abs=1e-7)

    def test_mann_whitney_detection(self) -> None:
        table = _table(multiclass=False)
        got = _auc_value(froc_auc(table, method="mann_whitney"))
        want = ref_froc_mw_detection(table)
        assert got == pytest.approx(want, abs=1e-9)

    def test_partial_normalize_is_engine_stable(self) -> None:
        """Ungrouped normalized partial AUC is deterministic across engines.

        The normalize path guards its zero-span case with a ``pl.when``; an
        earlier form embedded a sorted reduction in the predicate, which the
        in-memory engine miscompiled into non-deterministic, sometimes-negative
        values while streaming stayed correct. Pin determinism and engine parity
        at the entry point. (The full-range trapezoidal path that first exposed
        this was removed — trapezoidal now always integrates a partial window.)
        """
        table = _table(multiclass=False)
        # The curve stops near 1.7 FP/image; "flat" keeps a value to compare.
        expr = froc_auc(table, fp_range=(0.0, 8.0), extrapolate="flat")
        in_memory = {
            round(expr.collect(engine="in-memory").item(), 12) for _ in range(12)
        }
        streaming = expr.collect(engine="streaming").item()
        assert len(in_memory) == 1, f"non-deterministic in-memory: {in_memory}"
        assert next(iter(in_memory)) == pytest.approx(streaming, abs=_TOL)


class TestFrocAucGroupParity:
    """Grouped AUC equals per-group AUC on the filtered sub-table."""

    def test_trapezoidal_group_by_class(self) -> None:
        table = _table(multiclass=True)
        grouped = froc_auc(
            table, group_by="class_id", fp_range=(0.0, 8.0), extrapolate="flat"
        ).collect()
        got = dict(zip(grouped[COL_CLASS_ID].to_list(), grouped["auc"].to_list()))

        for cid in ("x", "y"):
            want = (
                froc_auc(
                    table.filter_class(cid), fp_range=(0.0, 8.0), extrapolate="flat"
                )
                .collect()
                .item()
            )
            assert got[cid] is not None
            assert got[cid] == pytest.approx(want, abs=_TOL)

    def test_off_curve_groups_are_null_by_group(self) -> None:
        """Each group's own curve decides whether the window leaves it."""
        table = _table(multiclass=True)
        grouped = froc_auc(table, group_by="class_id", fp_range=(0.0, 8.0)).collect()
        assert grouped["auc"].null_count() == grouped.height

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
        grouped = froc_auc(table, group_by="class_id", fp_range=(0.0, 8.0)).collect()
        assert sorted(grouped[COL_CLASS_ID].to_list()) == ["x", "y"]
        assert grouped.height == 2


class TestFrocStandaloneHelpers:
    """The lazy standalone helpers match the NumPy reference."""

    @pytest.mark.parametrize("extrapolate", ["none", "flat"])
    @pytest.mark.parametrize("fp", [0.0, 0.25, 0.5, 1.0, 2.0, 100.0])
    def test_sensitivity_at_fp(self, fp: float, extrapolate: str) -> None:
        table = _table(multiclass=False)
        got = (
            froc_sensitivity_at_fp(table, fp, extrapolate=extrapolate)
            .collect()["sensitivity"]
            .item()
        )
        want = ref_froc_sensitivity_at_fp(table, fp, extrapolate=extrapolate)
        assert got == want or got == pytest.approx(want, abs=1e-9)

    def test_summary_table_interpolates_the_curve(self) -> None:
        table = _table(multiclass=False)
        got = froc_summary_table(table, fp_rates=[0.25, 0.5, 1.0]).collect()
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


class TestFrocThresholds:
    """``thresholds=`` keeps exactly the curve rows at those scores."""

    def test_keeps_the_listed_thresholds(self) -> None:
        table = _table(multiclass=False)
        got = froc_curve_lazy(table, thresholds=[0.9, 0.4]).collect()
        assert got["threshold"].to_list() == [0.9, 0.4]

    def test_integer_thresholds_are_accepted(self) -> None:
        # polars 2.0's `is_in` refuses to compare Float64 data with an Int64
        # list, so a plain `thresholds=[1]` used to raise.
        table = _table(multiclass=False)
        got = froc_curve_lazy(table, thresholds=[1, 0.9]).collect()
        assert got["threshold"].to_list() == [0.9]
        assert froc_curve_lazy(table, thresholds=[1]).collect().height == 0


def _one_operating_point() -> DetectionTable:
    """The reporter's case: 100 images, 50 GTs, detections at one operating
    point (30 TPs, 5 FPs), so the curve stops at 0.05 FP/image, sensitivity 0.6."""
    from polars_cv.metrics import PreMatchedAdapter

    detections = pl.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(30)]
            + [f"img{i}" for i in range(60, 65)],
            "score": [0.9 - i * 0.01 for i in range(30)]
            + [0.5 - i * 0.01 for i in range(5)],
            "is_tp": [True] * 30 + [False] * 5,
        }
    )
    image_meta = pl.DataFrame(
        {"image_id": [f"img{i}" for i in range(100)], "n_gts": [1] * 50 + [0] * 50}
    )
    return PreMatchedAdapter().match(
        detections, image_id_col="image_id", image_meta=image_meta
    )


class TestFrocOffCurve:
    """One policy past the curve's end, for every FROC reader.

    ``froc_summary_table`` said the sensitivity at 2 FP/image was unknown while
    ``froc_auc(fp_range=(0, 2))`` reported 0.6 for the same table — a curve
    observed only up to 0.05 FP/image, silently extended.
    """

    def test_the_auc_past_the_curve_is_null(self) -> None:
        table = _one_operating_point()
        assert froc_auc(table, fp_range=(0.0, 2.0)).collect().item() is None

    def test_the_auc_within_the_curve_is_defined(self) -> None:
        table = _one_operating_point()
        # Every TP outranks every FP: the curve is at 0.6 from fp = 0.
        got = froc_auc(table, fp_range=(0.0, 0.05)).collect().item()
        assert got == pytest.approx(0.6)

    def test_flat_is_the_explicit_convention(self) -> None:
        table = _one_operating_point()
        auc = froc_auc(table, fp_range=(0.0, 2.0), extrapolate="flat")
        assert auc.collect().item() == pytest.approx(0.6)
        summary = froc_summary_table(table, [0.05, 1.0, 2.0], extrapolate="flat")
        assert summary.collect()["sensitivity"].to_list() == pytest.approx(
            [0.6, 0.6, 0.6]
        )

    def test_summary_and_auc_agree_by_default(self) -> None:
        table = _one_operating_point()
        summary = froc_summary_table(table, [0.05, 1.0, 2.0]).collect()
        assert summary["sensitivity"].to_list() == [pytest.approx(0.6), None, None]

    def test_the_operating_range_shows_where_the_curve_stops(self) -> None:
        from polars_cv.metrics import froc_operating_range

        got = froc_operating_range(_one_operating_point()).collect()
        assert got.columns == ["max_fp_per_image", "max_sensitivity"]
        assert got.row(0) == pytest.approx((0.05, 0.6))

    def test_the_operating_range_is_per_group(self) -> None:
        from polars_cv.metrics import froc_operating_range

        table = _table(multiclass=True)
        got = (
            froc_operating_range(table, group_by="class_id").collect().sort("class_id")
        )
        for cid, row in zip(got["class_id"], got.iter_rows(named=True)):
            curve = froc_curve_lazy(table.filter_class(cid)).collect()
            assert row["max_fp_per_image"] == pytest.approx(curve["fp_per_image"].max())
            assert row["max_sensitivity"] == pytest.approx(curve["sensitivity"].max())

    def test_bootstrap_bounds_are_null_when_a_replicate_is_off_curve(self) -> None:
        """An undefined replicate is not scored as an empty draw's 0.0."""
        from polars_cv.metrics import froc_auc_ci_lazy

        table = _one_operating_point()
        out = froc_auc_ci_lazy(
            table, fp_range=(0.0, 2.0), n_bootstrap=20, seed=0
        ).collect()
        assert out.row(0) == (None, None, None)
        flat = froc_auc_ci_lazy(
            table, fp_range=(0.0, 2.0), n_bootstrap=20, seed=0, extrapolate="flat"
        ).collect()
        assert flat["ci_lower"].item() is not None
        assert flat["ci_lower"].item() <= flat["auc"].item() <= flat["ci_upper"].item()


class TestFrocIsPureLazy:
    """No FROC entry point may collect during plan construction.

    The trapezoidal path used to run an eager ``image_metadata.collect(...)`` at
    build time (to feed a conflicting-weight guard); the guard is gone and the
    weight is resolved lazily instead. These pin that ``froc_curve_lazy`` /
    ``froc_auc`` and the standalone helpers build a plan without materialising
    anything — the caller owns the collect. Watched failing against the pre-fix
    code (3 collects).
    """

    @staticmethod
    def _collects_during(build) -> int:
        calls = {"n": 0}
        original = pl.LazyFrame.collect

        def counting(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return original(self, *args, **kwargs)

        pl.LazyFrame.collect = counting  # type: ignore[assignment]
        try:
            build()
        finally:
            pl.LazyFrame.collect = original  # type: ignore[assignment]
        return calls["n"]

    def test_froc_curve_lazy_builds_without_collecting(self) -> None:
        table = _table(multiclass=False)
        assert self._collects_during(lambda: froc_curve_lazy(table)) == 0

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda t: froc_auc(t, fp_range=(0.0, 8.0)), id="trapezoidal"),
            pytest.param(lambda t: froc_auc(t, fp_range=(0.0, 2.0)), id="partial"),
            pytest.param(
                lambda t: froc_auc(t, fp_range=(0.0, 8.0), correction="normalize"),
                id="normalize",
            ),
            pytest.param(
                lambda t: froc_auc(t, method="mann_whitney", level="detection"),
                id="mw-detection",
            ),
            pytest.param(
                lambda t: froc_auc(t, method="mann_whitney", level="image"),
                id="mw-image",
            ),
            pytest.param(lambda t: froc_sensitivity_at_fp(t, 1.0), id="sensitivity"),
            pytest.param(
                lambda t: froc_summary_table(t, fp_rates=[0.5, 1.0]), id="summary"
            ),
        ],
    )
    def test_pooled_entry_points_build_without_collecting(self, build) -> None:
        table = _table(multiclass=False)
        assert self._collects_during(lambda: build(table)) == 0

    def test_grouped_froc_auc_builds_without_collecting(self) -> None:
        table = _table(multiclass=True)
        assert (
            self._collects_during(
                lambda: froc_auc(table, group_by="class_id", fp_range=(0.0, 8.0))
            )
            == 0
        )
