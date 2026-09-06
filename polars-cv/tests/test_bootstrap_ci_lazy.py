"""Lazy, group-aware bootstrap confidence intervals.

The public CI seam is three free functions that return a ``pl.LazyFrame`` and
never collect internally:

* :func:`froc_auc_ci_lazy`  → ``[*group_by, auc, ci_lower, ci_upper]``
* :func:`lroc_auc_ci_lazy`  → ``[*group_by, auc, ci_lower, ci_upper]``
* :func:`average_precision_ci_lazy` → ``[*group_by, ap, ci_lower, ci_upper]``

These replace the eager ``bootstrap_{froc,lroc,pr}_auc`` scalar path. A downstream
compiler builds its plan with no data present, so the CI must stay a LazyFrame
until the caller's final ``.collect()`` and must carry one bound row per group so
it can be *joined* onto a point-metric frame rather than looped over in Python.

These tests pin: LazyFrame return, zero-collect plan construction, group-aware
schema, ``ci_lower <= point <= ci_upper``, seed reproducibility, thread-count
invariance, degenerate groups nulling their bounds (never raising), entity-level
resampling, and the Mann-Whitney / partial-range variants.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import polars as pl
import pytest

from polars_cv.metrics import (
    DetectionTable,
    average_precision,
    average_precision_ci_lazy,
    froc_auc,
    froc_auc_ci_lazy,
    lroc_auc,
    lroc_auc_ci_lazy,
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

# (image_id, score, is_tp, group)
Row = tuple[str, float, bool, str]


def _table(rows: list[Row], *, cases: dict[str, str] | None = None) -> DetectionTable:
    """One detection per image; ``gt_label``/``n_gts`` derive from ``is_tp``.

    ``group`` becomes a ``group_id`` metadata column. ``cases`` optionally maps
    each image to an entity id (``case_id``) for entity-level resampling.
    """
    det = pl.DataFrame(
        {
            COL_IMAGE_ID: [r[0] for r in rows],
            COL_CLASS_ID: ["__all__"] * len(rows),
            COL_SCORE: [r[1] for r in rows],
            COL_IS_TP: [r[2] for r in rows],
            COL_GT_IDX: [0 if r[2] else None for r in rows],
            COL_IOU: [0.7 if r[2] else 0.0 for r in rows],
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
    meta_cols = {
        COL_IMAGE_ID: [r[0] for r in rows],
        COL_CLASS_ID: ["__all__"] * len(rows),
        COL_N_GTS: [1 if r[2] else 0 for r in rows],
        COL_WEIGHT: [1.0] * len(rows),
        COL_GT_LABEL: [r[2] for r in rows],
        "group_id": [r[3] for r in rows],
    }
    schema = {
        COL_IMAGE_ID: pl.String,
        COL_CLASS_ID: pl.String,
        COL_N_GTS: pl.Int64,
        COL_WEIGHT: pl.Float64,
        COL_GT_LABEL: pl.Boolean,
        "group_id": pl.String,
    }
    if cases is not None:
        meta_cols["case_id"] = [cases[r[0]] for r in rows]
        schema["case_id"] = pl.String
    meta = pl.DataFrame(meta_cols, schema=schema)
    return DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)


def _mixed() -> DetectionTable:
    """A single-group table with both classes of image."""
    return _table(
        [
            ("a", 0.9, True, "g1"),
            ("b", 0.8, True, "g1"),
            ("c", 0.7, False, "g1"),
            ("d", 0.6, True, "g1"),
            ("e", 0.5, False, "g1"),
            ("f", 0.4, True, "g1"),
        ]
    )


def _two_groups() -> DetectionTable:
    """Two viable groups, each with positives and negatives."""
    return _table(
        [
            ("a", 0.9, True, "g1"),
            ("b", 0.8, False, "g1"),
            ("c", 0.7, True, "g1"),
            ("d", 0.6, False, "g1"),
            ("e", 0.85, True, "g2"),
            ("f", 0.5, False, "g2"),
            ("g", 0.65, True, "g2"),
            ("h", 0.3, False, "g2"),
        ]
    )


# --- entry-point registry so families share the property tests ----------------

_CI_FUNCS = {
    "froc": (froc_auc_ci_lazy, "auc"),
    "lroc": (lroc_auc_ci_lazy, "auc"),
    "pr": (average_precision_ci_lazy, "ap"),
}


def _collects_during(build) -> int:
    """Count ``pl.LazyFrame.collect`` calls made while ``build`` runs."""
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


class TestReturnsLazyAndZeroCollect:
    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_returns_lazyframe(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        out = fn(_mixed(), n_bootstrap=20, seed=1)
        assert isinstance(out, pl.LazyFrame)

    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_ungrouped_builds_without_collecting(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        table = _mixed()
        assert _collects_during(lambda: fn(table, n_bootstrap=20, seed=1)) == 0

    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_grouped_builds_without_collecting(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        table = _two_groups()
        assert (
            _collects_during(
                lambda: fn(table, group_by="group_id", n_bootstrap=20, seed=1)
            )
            == 0
        )


class TestSchemaAndShape:
    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_ungrouped_single_row(self, family: str) -> None:
        fn, value_col = _CI_FUNCS[family]
        out = fn(_mixed(), n_bootstrap=50, seed=1).collect()
        assert out.columns == [value_col, "ci_lower", "ci_upper"]
        assert out.height == 1

    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_grouped_one_row_per_group(self, family: str) -> None:
        fn, value_col = _CI_FUNCS[family]
        out = fn(_two_groups(), group_by="group_id", n_bootstrap=50, seed=1).collect()
        assert out.columns == ["group_id", value_col, "ci_lower", "ci_upper"]
        assert sorted(out["group_id"].to_list()) == ["g1", "g2"]

    def test_grouped_ci_frame_joins_onto_point_frame(self) -> None:
        # The headline downstream use: join bounds onto the point metric by group.
        table = _two_groups()
        point = froc_auc(table, group_by="group_id")
        ci = froc_auc_ci_lazy(
            table, group_by="group_id", n_bootstrap=50, seed=1
        ).select("group_id", "ci_lower", "ci_upper")
        joined = point.join(ci, on="group_id", how="left").collect()
        assert joined.height == 2
        assert set(joined.columns) == {"group_id", "auc", "ci_lower", "ci_upper"}


class TestBracketsPoint:
    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_ungrouped_brackets_point(self, family: str) -> None:
        fn, value_col = _CI_FUNCS[family]
        out = fn(_mixed(), n_bootstrap=200, seed=3).collect()
        lo = out["ci_lower"].item()
        hi = out["ci_upper"].item()
        point = out[value_col].item()
        assert lo <= point <= hi

    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_grouped_brackets_point_per_group(self, family: str) -> None:
        fn, value_col = _CI_FUNCS[family]
        out = fn(_two_groups(), group_by="group_id", n_bootstrap=200, seed=3).collect()
        for row in out.iter_rows(named=True):
            assert row["ci_lower"] <= row[value_col] <= row["ci_upper"]


class TestPointColumnParity:
    """The point column is the deterministic lazy metric, not a bootstrap mean."""

    def test_froc_point_matches_froc_auc(self) -> None:
        table = _mixed()
        got = froc_auc_ci_lazy(table, n_bootstrap=50, seed=1).collect()["auc"].item()
        assert got == pytest.approx(froc_auc(table).collect().item())

    def test_lroc_point_matches_lroc_auc(self) -> None:
        table = _mixed()
        got = lroc_auc_ci_lazy(table, n_bootstrap=50, seed=1).collect()["auc"].item()
        assert got == pytest.approx(lroc_auc(table).collect().item())

    def test_pr_point_matches_average_precision(self) -> None:
        table = _mixed()
        got = (
            average_precision_ci_lazy(table, n_bootstrap=50, seed=1)
            .collect()["ap"]
            .item()
        )
        assert got == pytest.approx(average_precision(table))

    def test_grouped_froc_point_matches_per_group_auc(self) -> None:
        table = _two_groups()
        ci = (
            froc_auc_ci_lazy(table, group_by="group_id", n_bootstrap=50, seed=1)
            .collect()
            .sort("group_id")
        )
        ref = froc_auc(table, group_by="group_id").collect().sort("group_id")
        for c, r in zip(ci["auc"].to_list(), ref["auc"].to_list()):
            assert c == pytest.approx(r)


class TestReproducible:
    @pytest.mark.parametrize("family", list(_CI_FUNCS))
    def test_same_seed_is_identical(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        table = _two_groups()
        a = fn(table, group_by="group_id", n_bootstrap=100, seed=7).collect()
        b = fn(table, group_by="group_id", n_bootstrap=100, seed=7).collect()
        assert a.sort("group_id").equals(b.sort("group_id"))

    def test_seedless_is_deterministic(self) -> None:
        # seed=None maps to a fixed constant, so even without a seed the bounds
        # are reproducible (a deliberate property of the lazy resampler).
        table = _mixed()
        a = froc_auc_ci_lazy(table, n_bootstrap=80).collect()
        b = froc_auc_ci_lazy(table, n_bootstrap=80).collect()
        assert a.equals(b)


class TestDegenerateGroups:
    """A degenerate group nulls its bounds without killing the plan."""

    def _table_with_degenerate_group(self) -> DetectionTable:
        # g1 is viable; g2 has no positive targets at all.
        return _table(
            [
                ("a", 0.9, True, "g1"),
                ("b", 0.8, False, "g1"),
                ("c", 0.7, True, "g1"),
                ("d", 0.6, False, "g2"),
                ("e", 0.5, False, "g2"),
            ]
        )

    def test_degenerate_group_nulls_bounds_not_point(self) -> None:
        table = self._table_with_degenerate_group()
        out = (
            froc_auc_ci_lazy(table, group_by="group_id", n_bootstrap=50, seed=1)
            .collect()
            .sort("group_id")
        )
        by_group = {row["group_id"]: row for row in out.iter_rows(named=True)}
        # Viable group: bounds present and bracket the point.
        g1 = by_group["g1"]
        assert g1["ci_lower"] is not None and g1["ci_upper"] is not None
        # Degenerate group (no positives): bounds null, but point still reported.
        g2 = by_group["g2"]
        assert g2["ci_lower"] is None
        assert g2["ci_upper"] is None
        assert g2["auc"] is not None

    def test_empty_table_yields_empty_frame(self) -> None:
        empty = _table([])
        out = froc_auc_ci_lazy(empty, group_by="group_id", n_bootstrap=10, seed=1)
        assert isinstance(out, pl.LazyFrame)
        assert out.collect().height == 0  # no raise

    def test_mann_whitney_requires_both_classes(self) -> None:
        """Mann-Whitney AUC is a two-class rank statistic, undefined without both
        classes: a group with positives but no negatives nulls its bounds, while
        the trapezoidal path (which needs only positives) keeps them."""
        # g1 viable (both classes); g2 has positives only (no negatives).
        table = _table(
            [
                ("a", 0.9, True, "g1"),
                ("b", 0.8, False, "g1"),
                ("c", 0.7, True, "g1"),
                ("d", 0.6, True, "g2"),
                ("e", 0.5, True, "g2"),
            ]
        )
        mw = (
            froc_auc_ci_lazy(
                table,
                group_by="group_id",
                n_bootstrap=50,
                seed=1,
                method="mann_whitney",
            )
            .collect()
            .sort("group_id")
        )
        mw_by = {r["group_id"]: r for r in mw.iter_rows(named=True)}
        assert mw_by["g1"]["ci_lower"] is not None
        assert mw_by["g2"]["ci_lower"] is None  # one-class group → null under MW
        assert mw_by["g2"]["ci_upper"] is None
        assert mw_by["g2"]["auc"] is not None  # point still reported

        # Trapezoidal only needs positives, so g2 stays viable there.
        trap = (
            froc_auc_ci_lazy(table, group_by="group_id", n_bootstrap=50, seed=1)
            .collect()
            .sort("group_id")
        )
        trap_by = {r["group_id"]: r for r in trap.iter_rows(named=True)}
        assert trap_by["g2"]["ci_lower"] is not None


class TestEntityLevel:
    """``sample_col`` resamples entities, composing with ``group_by``."""

    def _cased(self) -> DetectionTable:
        # two images per case; cases c1,c2 in g1 and c3,c4 in g2
        rows = [
            ("a", 0.9, True, "g1"),
            ("b", 0.8, False, "g1"),
            ("c", 0.7, True, "g1"),
            ("d", 0.6, False, "g1"),
            ("e", 0.85, True, "g2"),
            ("f", 0.5, False, "g2"),
            ("g", 0.65, True, "g2"),
            ("h", 0.3, False, "g2"),
        ]
        cases = {
            "a": "c1",
            "b": "c1",
            "c": "c2",
            "d": "c2",
            "e": "c3",
            "f": "c3",
            "g": "c4",
            "h": "c4",
        }
        return _table(rows, cases=cases)

    def test_entity_level_reproducible_and_grouped(self) -> None:
        table = self._cased()
        a = froc_auc_ci_lazy(
            table, group_by="group_id", n_bootstrap=100, seed=5, sample_col="case_id"
        ).collect()
        b = froc_auc_ci_lazy(
            table, group_by="group_id", n_bootstrap=100, seed=5, sample_col="case_id"
        ).collect()
        assert a.sort("group_id").equals(b.sort("group_id"))
        assert sorted(a["group_id"].to_list()) == ["g1", "g2"]
        for row in a.iter_rows(named=True):
            assert row["ci_lower"] <= row["auc"] <= row["ci_upper"]


class TestVariants:
    @pytest.mark.parametrize("level", ["detection", "image"])
    def test_froc_mann_whitney(self, level: str) -> None:
        table = _mixed()
        out = froc_auc_ci_lazy(
            table, n_bootstrap=50, seed=2, method="mann_whitney", level=level
        ).collect()
        assert out["auc"].item() == pytest.approx(
            froc_auc(table, method="mann_whitney", level=level).collect().item()
        )
        assert out["ci_lower"].item() <= out["ci_upper"].item()

    def test_froc_partial_range(self) -> None:
        table = _mixed()
        out = froc_auc_ci_lazy(
            table, n_bootstrap=50, seed=2, fp_range=(0.0, 2.0)
        ).collect()
        assert out["auc"].item() == pytest.approx(
            froc_auc(table, fp_range=(0.0, 2.0)).collect().item()
        )

    def test_lroc_mann_whitney(self) -> None:
        table = _mixed()
        out = lroc_auc_ci_lazy(
            table, n_bootstrap=50, seed=2, method="mann_whitney"
        ).collect()
        assert out["auc"].item() == pytest.approx(
            lroc_auc(table, method="mann_whitney").collect().item()
        )


class TestThreadCountInvariant:
    """A fixed seed gives identical bounds regardless of ``POLARS_MAX_THREADS``.

    The draw is a position-free hash of each row's global slot id, so it does not
    depend on how the streaming engine splits work. Run in subprocesses because
    Polars reads the thread count once at import.
    """

    _SNIPPET = textwrap.dedent(
        """
        import polars as pl
        from polars_cv.metrics import DetectionTable, froc_auc_ci_lazy
        from polars_cv.metrics._types import (
            COL_CLASS_ID, COL_DET_IDX, COL_GT_IDX, COL_GT_LABEL, COL_IMAGE_ID,
            COL_IOU, COL_IS_TP, COL_N_GTS, COL_SCORE, COL_WEIGHT,
        )
        rows = [("a",0.9,True,"g1"),("b",0.8,False,"g1"),("c",0.7,True,"g1"),
                ("d",0.6,False,"g1"),("e",0.85,True,"g2"),("f",0.5,False,"g2"),
                ("g",0.65,True,"g2"),("h",0.3,False,"g2")]
        det = pl.DataFrame(
            {COL_IMAGE_ID:[r[0] for r in rows], COL_CLASS_ID:["__all__"]*len(rows),
             COL_SCORE:[r[1] for r in rows], COL_IS_TP:[r[2] for r in rows],
             COL_GT_IDX:[0 if r[2] else None for r in rows],
             COL_IOU:[0.7 if r[2] else 0.0 for r in rows],
             COL_DET_IDX:list(range(len(rows)))},
            schema={COL_IMAGE_ID:pl.String, COL_CLASS_ID:pl.String, COL_SCORE:pl.Float64,
                    COL_IS_TP:pl.Boolean, COL_GT_IDX:pl.UInt32, COL_IOU:pl.Float64,
                    COL_DET_IDX:pl.UInt32})
        meta = pl.DataFrame(
            {COL_IMAGE_ID:[r[0] for r in rows], COL_CLASS_ID:["__all__"]*len(rows),
             COL_N_GTS:[1 if r[2] else 0 for r in rows], COL_WEIGHT:[1.0]*len(rows),
             COL_GT_LABEL:[r[2] for r in rows], "group_id":[r[3] for r in rows]},
            schema={COL_IMAGE_ID:pl.String, COL_CLASS_ID:pl.String, COL_N_GTS:pl.Int64,
                    COL_WEIGHT:pl.Float64, COL_GT_LABEL:pl.Boolean, "group_id":pl.String})
        t = DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)
        out = froc_auc_ci_lazy(t, group_by="group_id", n_bootstrap=200, seed=123)
        print(out.collect().sort("group_id").write_json())
        """
    )

    def _run(self, threads: int) -> str:
        import os

        env = dict(os.environ, POLARS_MAX_THREADS=str(threads))
        out = subprocess.run(
            [sys.executable, "-c", self._SNIPPET],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return out.stdout.strip()

    def test_bounds_match_across_thread_counts(self) -> None:
        assert self._run(1) == self._run(4)
