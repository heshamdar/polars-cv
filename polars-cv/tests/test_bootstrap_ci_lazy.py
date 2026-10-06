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

import functools
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_cv.metrics as M
from polars_cv.metrics import (
    DetectionTable,
    PreMatchedAdapter,
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


def _table(
    rows: list[Row],
    *,
    cases: dict[str, str] | None = None,
    extra: dict[str, dict[str, str]] | None = None,
) -> DetectionTable:
    """One detection per image; ``gt_label``/``n_gts`` derive from ``is_tp``.

    ``group`` becomes a ``group_id`` metadata column. ``cases`` optionally maps
    each image to an entity id (``case_id``) for entity-level resampling.
    ``extra`` adds String metadata columns, each mapping image → value.
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
    for name, values in (extra or {}).items():
        meta_cols[name] = [values[r[0]] for r in rows]
        schema[name] = pl.String
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

# FROC AUC now requires an explicit FP window, so the shared property tests bind
# one (the point column is froc_auc over the same window, so any fixed window
# keeps CI-vs-point parity).
_FROC_FP_RANGE = (0.0, 8.0)

_CI_FUNCS = {
    "froc": (
        functools.partial(
            froc_auc_ci_lazy, fp_range=_FROC_FP_RANGE, extrapolate="flat"
        ),
        "auc",
    ),
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
        point = froc_auc(
            table, group_by="group_id", fp_range=_FROC_FP_RANGE, extrapolate="flat"
        )
        ci = froc_auc_ci_lazy(
            table,
            group_by="group_id",
            n_bootstrap=50,
            seed=1,
            fp_range=_FROC_FP_RANGE,
            extrapolate="flat",
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
        got = (
            froc_auc_ci_lazy(
                table,
                n_bootstrap=50,
                seed=1,
                fp_range=_FROC_FP_RANGE,
                extrapolate="flat",
            )
            .collect()["auc"]
            .item()
        )
        assert got == pytest.approx(
            froc_auc(table, fp_range=_FROC_FP_RANGE, extrapolate="flat")
            .collect()
            .item()
        )

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
            froc_auc_ci_lazy(
                table,
                group_by="group_id",
                n_bootstrap=50,
                seed=1,
                fp_range=_FROC_FP_RANGE,
                extrapolate="flat",
            )
            .collect()
            .sort("group_id")
        )
        ref = (
            froc_auc(
                table, group_by="group_id", fp_range=_FROC_FP_RANGE, extrapolate="flat"
            )
            .collect()
            .sort("group_id")
        )
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
        a = froc_auc_ci_lazy(
            table, n_bootstrap=80, fp_range=_FROC_FP_RANGE, extrapolate="flat"
        ).collect()
        b = froc_auc_ci_lazy(
            table, n_bootstrap=80, fp_range=_FROC_FP_RANGE, extrapolate="flat"
        ).collect()
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
            froc_auc_ci_lazy(
                table,
                group_by="group_id",
                n_bootstrap=50,
                seed=1,
                fp_range=_FROC_FP_RANGE,
                extrapolate="flat",
            )
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
        out = froc_auc_ci_lazy(
            empty,
            group_by="group_id",
            n_bootstrap=10,
            seed=1,
            fp_range=_FROC_FP_RANGE,
            extrapolate="flat",
        )
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
            froc_auc_ci_lazy(
                table,
                group_by="group_id",
                n_bootstrap=50,
                seed=1,
                fp_range=_FROC_FP_RANGE,
                extrapolate="flat",
            )
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
            table,
            group_by="group_id",
            n_bootstrap=100,
            seed=5,
            sample_col="case_id",
            fp_range=_FROC_FP_RANGE,
            extrapolate="flat",
        ).collect()
        b = froc_auc_ci_lazy(
            table,
            group_by="group_id",
            n_bootstrap=100,
            seed=5,
            sample_col="case_id",
            fp_range=_FROC_FP_RANGE,
            extrapolate="flat",
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
            table, n_bootstrap=50, seed=2, fp_range=(0.0, 2.0), extrapolate="flat"
        ).collect()
        assert out["auc"].item() == pytest.approx(
            froc_auc(table, fp_range=(0.0, 2.0), extrapolate="flat").collect().item()
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
        out = froc_auc_ci_lazy(
            t, group_by="group_id", n_bootstrap=200, seed=123, fp_range=(0.0, 8.0), extrapolate="flat"
        )
        print(out.collect().sort("group_id").write_json())
        # Weighted: the resample is also stratified by weight cell.
        tw = DetectionTable.from_matched(
            det, meta.with_columns(pl.Series(COL_WEIGHT, [1.0, 2.0] * 4)),
            matching_iou_threshold=0.5,
        )
        out = froc_auc_ci_lazy(
            tw, group_by="group_id", n_bootstrap=200, seed=123, method="mann_whitney"
        )
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
        one = self._run(1)
        assert one.count("\n") == 1  # unweighted + weighted outputs
        assert one == self._run(4)


# --- weight-cell stratification ---------------------------------------------------

# Target vendor mix the importance weights reweight to: w = p_v / q̂_v.
_TARGET = {"A": 0.3, "B": 0.7}


# Cases of uneven size, each within one vendor: (group, first image, size).
_UNEVEN_CASES = {
    "g1": [(0, 1), (1, 2), (3, 3), (6, 1), (7, 3)],
    "g2": [(0, 2), (2, 1), (3, 1), (4, 3), (7, 1), (8, 2)],
}


def _vendor_table(*, uneven_cases: bool = False) -> DetectionTable:
    """Two groups of ten images with different vendor mixes (g1 6A/4B, g2 4A/6B).

    Unit weights; :func:`_importance_weighted` sets ``p / q̂``. Each ``case_id``
    pairs two images of one vendor, so an entity-level draw keeps the image mix —
    unless ``uneven_cases``, whose cases hold one to three images each.
    """
    rows: list[Row] = []
    vendor: dict[str, str] = {}
    cases: dict[str, str] = {}
    for group, shift, n_a in (("g1", 0, 6), ("g2", 3, 4)):
        for i in range(10):
            image = f"{group}_{i}"
            rows.append((image, ((i * 7 + shift) % 10) / 10 + 0.05, i % 2 == 0, group))
            vendor[image] = "A" if i < n_a else "B"
            cases[image] = f"{group}_c{i // 2}"
    if uneven_cases:
        for group, spans in _UNEVEN_CASES.items():
            for first, size in spans:
                for i in range(first, first + size):
                    cases[f"{group}_{i}"] = f"{group}_c{first}"
    return _table(rows, cases=cases, extra={"vendor": vendor})


def _p_over_q(over: list[str]) -> pl.Expr:
    """``p_v / q̂_v`` with ``q̂_v`` the vendor's share within ``over``."""
    p = pl.col("vendor").replace_strict(_TARGET, return_dtype=pl.Float64)
    share = (
        pl.len().over(*over, "vendor") / pl.len().over(*over)
        if over
        else (pl.len().over("vendor") / pl.len())
    )
    return (p / share).alias(COL_WEIGHT)


def _with_meta(table: DetectionTable, meta: pl.LazyFrame) -> DetectionTable:
    return DetectionTable.from_matched(
        table.detections, meta, matching_iou_threshold=0.5
    )


def _importance_weighted(scope: str, *, uneven_cases: bool = False) -> DetectionTable:
    """``_vendor_table`` weighted to ``_TARGET``, ``q̂`` per group or global."""
    table = _vendor_table(uneven_cases=uneven_cases)
    over = ["group_id"] if scope == "per_group" else []
    return _with_meta(table, table.image_metadata.with_columns(_p_over_q(over)))


def _replicates(
    table: DetectionTable,
    *,
    group_keys: list[str],
    sample_col: str | None = None,
    strata: list[str] | None = (),  # type: ignore[assignment]
    weight_rtol: float | None = None,
    n_bootstrap: int = 60,
    weight_scheme: str = "reestimate",
    seed: int = 3,
) -> DetectionTable:
    """The CI path's replicate table; ``strata=None`` is a plain, cell-blind draw."""
    from polars_cv.metrics._bootstrap import (
        _bootstrap_table_with_draws,
        _replicate_tables,
        _resolve_bootstrap_samples,
    )

    if strata is None:
        samples = _resolve_bootstrap_samples(
            table,
            sample_col=sample_col,
            n_bootstrap=n_bootstrap,
            seed=seed,
            group_keys=group_keys,
        )
        return _bootstrap_table_with_draws(table, samples, group_keys=group_keys)
    (boot,), _ = _replicate_tables(
        table,
        group_keys=group_keys,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata=list(strata),
        weight_rtol=weight_rtol,
        weight_scheme=weight_scheme,
        batch=None,
    )
    return boot


def _replicate_drift(boot: DetectionTable, scope: str, metric: str) -> float:
    """Max per-replicate |static − re-estimated-weight| metric difference."""
    over = ["bootstrap_id", "group_id"] if scope == "per_group" else ["bootstrap_id"]
    re_estimated = _with_meta(boot, boot.image_metadata.with_columns(_p_over_q(over)))
    keys = ["group_id", "bootstrap_id"]

    def run(tbl: DetectionTable) -> pl.DataFrame:
        if metric == "froc_mw":
            out = froc_auc(tbl, method="mann_whitney", group_by=keys)
        elif metric == "froc_trap":
            out = froc_auc(tbl, fp_range=(0.0, 1.0), extrapolate="flat", group_by=keys)
        else:
            out = lroc_auc(tbl, group_by=keys)
        return out.collect().sort(keys)

    static, fresh = run(boot), run(re_estimated)
    assert static.select(keys).equals(fresh.select(keys))
    return float((static["auc"] - fresh["auc"]).abs().max())  # type: ignore[arg-type]


class TestWeightCellStratification:
    """Weighted FROC/LROC CIs stratify the resample on weight cells.

    The weighted statistics are weight-scale-invariant ratios, so an importance
    weight ``p / q̂`` matters only through its cell's count. Redrawing every cell
    to its own size keeps that count — so the full-sample weights *are* the
    per-replicate re-estimated weights, and no reweight hook is needed.
    """

    @pytest.mark.parametrize("metric", ["froc_mw", "froc_trap", "lroc"])
    @pytest.mark.parametrize("scope", ["per_group", "global"])
    def test_static_weights_equal_re_estimated_weights(
        self, scope: str, metric: str
    ) -> None:
        table = _importance_weighted(scope)
        boot = _replicates(table, group_keys=["group_id"])
        assert _replicate_drift(boot, scope, metric) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("scope", ["per_group", "global"])
    def test_unstratified_draw_drifts(self, scope: str) -> None:
        # The discriminating half: without weight cells the vendor mix wanders,
        # so held-fixed weights are not the re-estimated ones.
        table = _importance_weighted(scope)
        boot = _replicates(table, group_keys=["group_id"], strata=None)
        assert _replicate_drift(boot, scope, "froc_mw") > 0.01

    def test_ungrouped_static_weights_equal_re_estimated_weights(self) -> None:
        table = _importance_weighted("global")
        boot = _replicates(table, group_keys=[])
        meta = boot.image_metadata.with_columns(_p_over_q(["bootstrap_id"]))

        def run(tbl: DetectionTable) -> pl.DataFrame:
            return (
                froc_auc(tbl, method="mann_whitney", group_by="bootstrap_id")
                .collect()
                .sort("bootstrap_id")
            )

        assert run(boot)["auc"].to_list() == pytest.approx(
            run(_with_meta(boot, meta))["auc"].to_list(), abs=1e-12
        )

    def test_entity_level_static_weights_equal_re_estimated_weights(self) -> None:
        table = _importance_weighted("per_group")
        boot = _replicates(table, group_keys=["group_id"], sample_col="case_id")
        assert _replicate_drift(boot, "per_group", "froc_mw") == pytest.approx(
            0.0, abs=1e-12
        )

    @pytest.mark.parametrize("sample_col", [None, "case_id"])
    def test_every_replicate_keeps_the_vendor_mix(self, sample_col: str | None) -> None:
        table = _importance_weighted("per_group")
        boot = _replicates(table, group_keys=["group_id"], sample_col=sample_col)
        full = (
            table.image_metadata.group_by("group_id", "vendor")
            .len()
            .collect()
            .sort("group_id", "vendor")
        )
        per_rep = boot.image_metadata.group_by(
            "bootstrap_id", "group_id", "vendor"
        ).len()
        mismatched = (
            per_rep.join(full.lazy(), on=["group_id", "vendor"], suffix="_full")
            .filter(pl.col("len") != pl.col("len_full"))
            .collect()
        )
        assert per_rep.select(pl.len()).collect().item() == 60 * full.height
        assert mismatched.height == 0

    # Bounds of `_vendor_table` (unit weights) frozen from the pre-stratification
    # resampler: one weight cell per group must leave every draw unchanged.
    _UNIT_WEIGHT_BOUNDS = {
        "froc_trap_grouped": [
            ("g1", 0.96875, 0.95125, 0.99003125),
            ("g2", 0.98, 0.9599687499999999, 0.99875),
        ],
        "froc_mw": [(0.5, 0.25462500000000005, 0.735125)],
        "lroc_grouped": [
            ("g1", 0.5, 0.22, 0.8405000000000001),
            ("g2", 0.68, 0.35950000000000004, 0.98),
        ],
        "froc_entity": [
            ("g1", 0.4, 0.04, 0.9210000000000003),
            ("g2", 0.6, 0.23800000000000032, 1.0),
        ],
    }

    def _unit_weight_bounds(self) -> dict[str, list[tuple]]:
        table = _vendor_table()
        kw = {"n_bootstrap": 200, "seed": 11}
        return {
            "froc_trap_grouped": froc_auc_ci_lazy(
                table,
                group_by="group_id",
                fp_range=_FROC_FP_RANGE,
                extrapolate="flat",
                **kw,
            )
            .collect()
            .sort("group_id")
            .rows(),
            "froc_mw": froc_auc_ci_lazy(table, method="mann_whitney", **kw)
            .collect()
            .rows(),
            "lroc_grouped": lroc_auc_ci_lazy(table, group_by="group_id", **kw)
            .collect()
            .sort("group_id")
            .rows(),
            "froc_entity": froc_auc_ci_lazy(
                table,
                group_by="group_id",
                method="mann_whitney",
                sample_col="case_id",
                **kw,
            )
            .collect()
            .sort("group_id")
            .rows(),
        }

    def test_unit_weights_are_unchanged(self) -> None:
        assert self._unit_weight_bounds() == self._UNIT_WEIGHT_BOUNDS

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_singleton_weight_cells_null_the_bounds(self, family: str) -> None:
        # A continuous weight puts every image in its own cell: every replicate
        # would be the original sample, a zero-width interval. Null instead.
        fn, value_col = _CI_FUNCS[family]
        table = _vendor_table()
        distinct = (pl.int_range(pl.len()).cast(pl.Float64) + 1.0).alias(COL_WEIGHT)
        weighted = _with_meta(table, table.image_metadata.with_columns(distinct))
        out = fn(weighted, n_bootstrap=50, seed=1).collect()
        assert out["ci_lower"].item() is None
        assert out["ci_upper"].item() is None
        point = fn(table, n_bootstrap=50, seed=1).collect()[value_col]
        assert out[value_col].item() is not None
        assert point.item() is not None

    def test_singleton_rule_is_per_group(self) -> None:
        table = _vendor_table()
        g2_distinct = (
            pl.when(pl.col("group_id") == "g2")
            .then(pl.int_range(pl.len()).cast(pl.Float64) + 1.0)
            .otherwise(1.0)
            .alias(COL_WEIGHT)
        )
        weighted = _with_meta(table, table.image_metadata.with_columns(g2_distinct))
        out = {
            r["group_id"]: r
            for r in froc_auc_ci_lazy(
                weighted,
                group_by="group_id",
                n_bootstrap=50,
                seed=1,
                method="mann_whitney",
            )
            .collect()
            .iter_rows(named=True)
        }
        assert out["g1"]["ci_lower"] is not None
        assert out["g2"]["ci_lower"] is None
        assert out["g2"]["auc"] is not None

    def test_strata_separates_cells_sharing_a_weight(self) -> None:
        # Both vendors at w = 1.0 share one weight cell; naming the column keeps
        # each vendor's count fixed, which the weight alone cannot.
        table = _vendor_table()
        full = (
            table.image_metadata.group_by("group_id", "vendor")
            .len()
            .collect()
            .sort("group_id", "vendor")
        )

        def mismatches(strata: list[str]) -> int:
            boot = _replicates(table, group_keys=["group_id"], strata=strata)
            return (
                boot.image_metadata.group_by("bootstrap_id", "group_id", "vendor")
                .len()
                .join(full.lazy(), on=["group_id", "vendor"], how="full", suffix="_f")
                .filter(pl.col("len").fill_null(0) != pl.col("len_f").fill_null(0))
                .collect()
                .height
            )

        assert mismatches([]) > 0
        assert mismatches(["vendor"]) == 0

    @pytest.mark.parametrize("family", ["froc", "lroc"])
    def test_strata_keeps_the_point_and_brackets_it(self, family: str) -> None:
        fn, value_col = _CI_FUNCS[family]
        table = _importance_weighted("per_group")
        plain = fn(table, group_by="group_id", n_bootstrap=100, seed=2).collect()
        out = fn(
            table, group_by="group_id", strata="vendor", n_bootstrap=100, seed=2
        ).collect()
        assert out.sort("group_id")[value_col].equals(plain.sort("group_id")[value_col])
        for row in out.iter_rows(named=True):
            assert row["ci_lower"] <= row[value_col] <= row["ci_upper"]

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_unknown_strata_column_raises(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        with pytest.raises(ValueError, match="strata.*'scanner'"):
            fn(_vendor_table(), strata="scanner", n_bootstrap=10, seed=1)

    def test_average_precision_static_weights_equal_re_estimated_weights(
        self,
    ) -> None:
        # AP is weighted too, so it shares the weight-cell draw.
        from polars_cv.metrics._statistics import AP

        table = _importance_weighted("per_group")
        boot = _replicates(table, group_keys=["group_id"])
        meta = boot.image_metadata.with_columns(_p_over_q(["bootstrap_id", "group_id"]))
        keys = ["group_id", "bootstrap_id"]
        static = AP().by_group(boot, keys).collect().sort(keys)
        fresh = AP().by_group(_with_meta(boot, meta), keys).collect().sort(keys)
        # A replicate that drew no positives has AP NaN under both weightings
        # (the interval treats it as degenerate; see TestWeightScheme).
        assert static["ap"].to_list() == pytest.approx(
            fresh["ap"].to_list(), abs=1e-12, nan_ok=True
        )

    def test_average_precision_singleton_cells_null_the_bounds(self) -> None:
        table = _vendor_table()
        distinct = (pl.int_range(pl.len()).cast(pl.Float64) + 1.0).alias(COL_WEIGHT)
        weighted = _with_meta(table, table.image_metadata.with_columns(distinct))
        out = average_precision_ci_lazy(weighted, n_bootstrap=20, seed=4).collect()
        assert out["ci_lower"].item() is None
        assert out["ap"].item() == pytest.approx(average_precision(weighted))

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_weighted_build_does_not_collect(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        table = _importance_weighted("per_group")
        assert (
            _collects_during(
                lambda: fn(
                    table,
                    group_by="group_id",
                    strata="vendor",
                    sample_col="case_id",
                    n_bootstrap=20,
                    seed=1,
                )
            )
            == 0
        )


def _count_mismatches(
    table: DetectionTable, boot: DetectionTable, by: list[str]
) -> int:
    """Replicate ``(group_id, *by)`` counts that differ from the full sample's."""
    full = table.image_metadata.group_by("group_id", *by).len().collect()
    return (
        boot.image_metadata.group_by("bootstrap_id", "group_id", *by)
        .len()
        .join(full.lazy(), on=["group_id", *by], how="full", suffix="_f")
        .filter(pl.col("len").fill_null(0) != pl.col("len_f").fill_null(0))
        .collect()
        .height
    )


def _continuous(table: DetectionTable) -> DetectionTable:
    """``table`` with a distinct weight on every image (a continuous weight)."""
    distinct = (pl.int_range(pl.len()).cast(pl.Float64) + 1.0).alias(COL_WEIGHT)
    return _with_meta(table, table.image_metadata.with_columns(distinct))


class TestWeightScheme:
    """``weight_scheme`` says what the weights are, and so how the draw treats them.

    * ``"reestimate"`` (default): ``p / q̂`` weights estimated from the sample.
      Each weight cell is redrawn to its own size, so the weights are their
      per-replicate re-estimates, but ``gt_label`` is not crossed with the cells:
      each cell's positive count is random in the population, and the weighted
      statistics depend on it. A group with a single cell keeps the ``gt_label``
      stratum.
    * ``"stratified"``: each cell's positive count was fixed by the study design;
      the draw keeps ``(gt_label, cell)`` counts.
    * ``"fixed"``: known (design) weights, carried unchanged; no weight cells.
    """

    def test_reestimate_lets_each_cells_label_count_vary(self) -> None:
        table = _importance_weighted("per_group")
        boot = _replicates(table, group_keys=["group_id"])
        assert _count_mismatches(table, boot, ["vendor"]) == 0
        assert _count_mismatches(table, boot, [COL_GT_LABEL, "vendor"]) > 0

    def test_stratified_freezes_each_cells_label_count(self) -> None:
        table = _importance_weighted("per_group")
        boot = _replicates(table, group_keys=["group_id"], weight_scheme="stratified")
        assert _count_mismatches(table, boot, [COL_GT_LABEL, "vendor"]) == 0

    def test_reestimate_keeps_the_label_stratum_in_a_single_cell_group(self) -> None:
        # g1 is importance-weighted (two cells), g2 carries unit weights (one cell).
        table = _importance_weighted("per_group")
        meta = table.image_metadata.with_columns(
            pl.when(pl.col("group_id") == "g2")
            .then(1.0)
            .otherwise(pl.col(COL_WEIGHT))
            .alias(COL_WEIGHT)
        )
        mixed = _with_meta(table, meta)
        boot = _replicates(mixed, group_keys=["group_id"])
        full = mixed.image_metadata.group_by("group_id", COL_GT_LABEL).len().collect()
        per_rep = (
            boot.image_metadata.group_by("bootstrap_id", "group_id", COL_GT_LABEL)
            .len()
            .join(full.lazy(), on=["group_id", COL_GT_LABEL], suffix="_f")
            .filter(pl.col("len") != pl.col("len_f"))
            .collect()
        )
        assert set(per_rep["group_id"]) == {"g1"}

    def test_fixed_carries_weights_on_a_label_stratified_draw(self) -> None:
        table = _continuous(_vendor_table())
        boot = _replicates(table, group_keys=["group_id"], weight_scheme="fixed")
        assert _count_mismatches(table, boot, [COL_GT_LABEL]) == 0
        assert _count_mismatches(table, boot, ["vendor"]) > 0
        source = table.image_metadata.select(
            COL_IMAGE_ID, pl.col(COL_WEIGHT).alias("_w0")
        )
        drawn = boot.image_metadata.with_columns(
            pl.col(COL_IMAGE_ID).str.replace(r"#d\d+$", "")
        ).join(source, on=COL_IMAGE_ID)
        assert drawn.filter(pl.col(COL_WEIGHT) != pl.col("_w0")).collect().height == 0

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_fixed_bounds_a_continuous_weight(self, family: str) -> None:
        fn, value_col = _CI_FUNCS[family]
        table = _continuous(_vendor_table())
        out = fn(table, n_bootstrap=50, seed=1, weight_scheme="fixed").collect()
        assert out["ci_lower"].item() <= out[value_col].item() <= out["ci_upper"].item()
        assert out["ci_lower"].item() < out["ci_upper"].item()
        # The default reads a continuous weight as all-singleton cells: no bounds.
        assert fn(table, n_bootstrap=50, seed=1).collect()["ci_lower"].item() is None

    @pytest.mark.parametrize("scheme", ["reestimate", "stratified", "fixed"])
    def test_unit_weights_draw_alike_under_every_scheme(self, scheme: str) -> None:
        table = _vendor_table()
        kw = {"group_by": "group_id", "n_bootstrap": 100, "seed": 7}
        base = lroc_auc_ci_lazy(table, **kw).collect().sort("group_id")
        out = lroc_auc_ci_lazy(table, weight_scheme=scheme, **kw).collect()
        assert_frame_equal(out.sort("group_id"), base)

    @pytest.mark.parametrize("scheme", ["reestimate", "stratified"])
    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_a_zero_weight_singleton_does_not_null_the_bounds(
        self, family: str, scheme: str
    ) -> None:
        # An image outside the target (weight 0) is its own cell but contributes
        # nothing to any statistic: it has no variance to hide.
        fn, _ = _CI_FUNCS[family]
        table = _vendor_table()
        meta = table.image_metadata.with_columns(
            pl.when(pl.col(COL_IMAGE_ID) == "g1_9")
            .then(0.0)
            .otherwise(pl.col(COL_WEIGHT))
            .alias(COL_WEIGHT)
        )
        out = fn(
            _with_meta(table, meta), n_bootstrap=50, seed=1, weight_scheme=scheme
        ).collect()
        assert out["ci_lower"].item() is not None

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"strata": "vendor"}, "strata"), ({"weight_rtol": 0.0}, "weight_rtol")],
    )
    def test_fixed_rejects_cell_arguments(
        self, family: str, kwargs: dict, match: str
    ) -> None:
        fn, _ = _CI_FUNCS[family]
        with pytest.raises(ValueError, match=match):
            fn(_vendor_table(), weight_scheme="fixed", n_bootstrap=10, **kwargs)

    def test_a_replicate_without_positives_nulls_its_groups_bounds(self) -> None:
        # Under "reestimate" g1's replicate 30 draws no positive image. Its
        # statistic still has a value (0.0 here) that describes no resample of
        # the group, so it nulls g1's bounds instead of being scored.
        table = TestWeightCellTolerance._noisy()
        kw = {"group_by": "group_id", "n_bootstrap": 50, "seed": 1}
        boot = _replicates(table, group_keys=["group_id"], n_bootstrap=50, seed=1)
        drew_none = (
            boot.image_metadata.group_by("bootstrap_id", "group_id")
            .agg(pl.col(COL_GT_LABEL).any())
            .filter(~pl.col(COL_GT_LABEL))
            .collect()
        )
        assert set(drew_none["group_id"]) == {"g1"}
        out = {
            r["group_id"]: r
            for r in lroc_auc_ci_lazy(table, **kw).collect().iter_rows(named=True)
        }
        assert out["g1"]["ci_lower"] is None
        assert out["g1"]["auc"] is not None
        assert out["g2"]["ci_lower"] is not None
        kept = lroc_auc_ci_lazy(table, weight_scheme="stratified", **kw).collect()
        assert kept["ci_lower"].null_count() == 0

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_unknown_scheme_raises(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        with pytest.raises(ValueError, match="weight_scheme.*'design'"):
            fn(_vendor_table(), weight_scheme="design", n_bootstrap=10)


class TestWeightCellTolerance:
    """Weights within ``weight_rtol`` (relative, default ``1e-6``) share a cell."""

    @staticmethod
    def _noisy() -> DetectionTable:
        # One image per (group, vendor) cell recomputes its weight along another
        # arithmetic path: equal in intent, unequal in the last bits.
        table = _importance_weighted("per_group")
        noise = pl.when(pl.col(COL_IMAGE_ID).is_in(["g1_0", "g1_9", "g2_0", "g2_9"]))
        meta = table.image_metadata.with_columns(
            noise.then(pl.col(COL_WEIGHT) * (1.0 + 1e-12)).otherwise(pl.col(COL_WEIGHT))
        )
        return _with_meta(table, meta)

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_float_noise_does_not_split_a_cell(self, family: str) -> None:
        # "stratified" keeps every replicate's gt_label counts, so a null bound
        # here can only come from a cell split, not a degenerate replicate.
        fn = functools.partial(_CI_FUNCS[family][0], weight_scheme="stratified")
        table = self._noisy()
        default = fn(table, group_by="group_id", n_bootstrap=50, seed=1).collect()
        assert default["ci_lower"].null_count() == 0
        # Exact comparison isolates each perturbed image as a singleton cell.
        exact = fn(
            table, group_by="group_id", n_bootstrap=50, seed=1, weight_rtol=0.0
        ).collect()
        assert exact["ci_lower"].null_count() == 2

    def test_tolerance_merges_only_weights_within_it(self) -> None:
        # Vendors at 1.0 and 1.01: apart at 1e-3, merged at 5e-2.
        table = _vendor_table()
        meta = table.image_metadata.with_columns(
            pl.when(pl.col("vendor") == "A").then(1.0).otherwise(1.01).alias(COL_WEIGHT)
        )
        weighted = _with_meta(table, meta)
        full = weighted.image_metadata.group_by("group_id", "vendor").len().collect()

        def mismatches(rtol: float) -> int:
            boot = _replicates(weighted, group_keys=["group_id"], weight_rtol=rtol)
            return (
                boot.image_metadata.group_by("bootstrap_id", "group_id", "vendor")
                .len()
                .join(full.lazy(), on=["group_id", "vendor"], how="full", suffix="_f")
                .filter(pl.col("len").fill_null(0) != pl.col("len_f").fill_null(0))
                .collect()
                .height
            )

        assert mismatches(1e-3) == 0
        assert mismatches(5e-2) > 0

    @pytest.mark.parametrize("bad", [-1e-6, float("nan"), float("inf")])
    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_invalid_tolerance_raises(self, family: str, bad: float) -> None:
        fn, _ = _CI_FUNCS[family]
        with pytest.raises(ValueError, match="weight_rtol"):
            fn(_vendor_table(), weight_rtol=bad, n_bootstrap=10, seed=1)


class TestEntityLevelReweighting:
    """A coarser draw (``sample_col``) re-estimates each replicate's weights.

    Entities of uneven size let a replicate's image mix wander even though its
    entity mix is fixed; each image's weight is rescaled by
    ``(n_c/N) / (n*_c/N*)`` so every ``(group, cell)`` keeps its full-sample share
    of the group's images — what re-estimating ``p / q̂`` on the replicate gives.
    """

    @pytest.mark.parametrize("metric", ["froc_mw", "froc_trap", "lroc"])
    def test_rescaled_weights_equal_re_estimated_weights(self, metric: str) -> None:
        table = _importance_weighted("per_group", uneven_cases=True)
        boot = _replicates(table, group_keys=["group_id"], sample_col="case_id")
        assert _replicate_drift(boot, "per_group", metric) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_every_replicate_keeps_the_weighted_vendor_share(self) -> None:
        table = _importance_weighted("per_group", uneven_cases=True)
        boot = _replicates(table, group_keys=["group_id"], sample_col="case_id")

        def shares(meta: pl.LazyFrame, by: list[str]) -> pl.LazyFrame:
            mass = pl.col(COL_WEIGHT).sum()
            return (
                meta.group_by(*by, "vendor")
                .agg(mass.alias("_m"))
                .with_columns(share=pl.col("_m") / pl.col("_m").sum().over(by))
            )

        full = shares(table.image_metadata, ["group_id"]).select(
            "group_id", "vendor", "share"
        )
        rep = shares(boot.image_metadata, ["bootstrap_id", "group_id"])
        diff = (
            rep.join(full, on=["group_id", "vendor"], suffix="_full")
            .select((pl.col("share") - pl.col("share_full")).abs().max())
            .collect()
            .item()
        )
        assert diff == pytest.approx(0.0, abs=1e-12)

    def test_image_level_weights_are_untouched(self) -> None:
        # Stratified image draws keep every cell's count: the factor is exactly 1.
        table = _importance_weighted("per_group")
        boot = _replicates(table, group_keys=["group_id"])
        original = table.image_metadata.select(COL_IMAGE_ID, COL_WEIGHT)
        drawn = boot.image_metadata.with_columns(
            pl.col(COL_IMAGE_ID).str.split("#d").list.first()
        ).join(original, on=COL_IMAGE_ID, suffix="_full")
        assert (
            drawn.filter(pl.col(COL_WEIGHT) != pl.col(f"{COL_WEIGHT}_full"))
            .collect()
            .height
            == 0
        )


class TestMultiClassDraw:
    """One draw slot per image, however many class rows it has.

    ``image_metadata`` has a row per ``(image, class)``. An image positive for
    one class and negative for another used to sit in both ``gt_label`` strata,
    so it had two draw chances per replicate and replicates held more images
    than the sample. An image's stratum is now positive if any class is.
    """

    @staticmethod
    def _two_class() -> DetectionTable:
        # Six images; a, c, e positive for "car" and negative for "ped".
        images = ["a", "b", "c", "d", "e", "f"]
        car_pos = {"a", "b", "c", "e"}
        ped_pos = {"b", "d"}
        det_rows = [
            (img, cls, 0.9 - 0.1 * k, img in pos)
            for k, img in enumerate(images)
            for cls, pos in (("car", car_pos), ("ped", ped_pos))
        ]
        det = pl.DataFrame(
            {
                COL_IMAGE_ID: [r[0] for r in det_rows],
                COL_CLASS_ID: [r[1] for r in det_rows],
                COL_SCORE: [r[2] for r in det_rows],
                COL_IS_TP: [r[3] for r in det_rows],
                COL_GT_IDX: [0 if r[3] else None for r in det_rows],
                COL_IOU: [0.7 if r[3] else 0.0 for r in det_rows],
                COL_DET_IDX: list(range(len(det_rows))),
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
                COL_IMAGE_ID: [r[0] for r in det_rows],
                COL_CLASS_ID: [r[1] for r in det_rows],
                COL_N_GTS: [int(r[3]) for r in det_rows],
                COL_WEIGHT: [1.0] * len(det_rows),
                COL_GT_LABEL: [r[3] for r in det_rows],
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

    @pytest.mark.parametrize("weighted", [False, True])
    def test_every_replicate_draws_one_slot_per_image(self, weighted: bool) -> None:
        table = self._two_class()
        boot = _replicates(table, group_keys=[], strata=[] if weighted else None)
        draws = (
            boot.image_metadata.group_by("bootstrap_id")
            .agg(pl.col(COL_IMAGE_ID).n_unique().alias("n"))
            .collect()
        )
        assert draws["n"].unique().to_list() == [6]
        # Each draw carries all of its image's class rows, once.
        rows = boot.image_metadata.select(pl.len()).collect().item()
        assert rows == 60 * 12

    def test_class_partitioned_draw_is_unchanged(self) -> None:
        # Grouped by class, every partition already has one row per image.
        table = self._two_class()
        boot = _replicates(table, group_keys=[COL_CLASS_ID])
        draws = (
            boot.image_metadata.group_by("bootstrap_id", COL_CLASS_ID)
            .agg(pl.col(COL_IMAGE_ID).n_unique().alias("n"))
            .collect()
        )
        assert draws["n"].unique().to_list() == [6]

    @pytest.mark.parametrize("family", ["froc", "lroc", "pr"])
    def test_ci_brackets_point(self, family: str) -> None:
        fn, value_col = _CI_FUNCS[family]
        out = fn(self._two_class(), n_bootstrap=100, seed=2).collect()
        assert out["ci_lower"].item() <= out[value_col].item() <= out["ci_upper"].item()


@pytest.mark.parametrize("n_bootstrap", [1, 2, 7, 40, 1000])
def test_percentile_bounds_are_polars_linear_quantiles_bit_for_bit(
    n_bootstrap: int,
) -> None:
    """The rank-read bounds equal ``quantile(q, "linear")`` exactly.

    They replaced that aggregation because it is not native to the streaming
    engine; bootstrap bounds are compared bit for bit across runs, so the
    replacement must not move a single bit.
    """
    from polars_cv.metrics._bootstrap import _percentile_bounds

    rng = np.random.default_rng(n_bootstrap)
    groups = ["a", "b", "c"]
    joined = pl.DataFrame(
        {
            "g": np.repeat(groups, n_bootstrap),
            "v": rng.lognormal(0.0, 3.0, n_bootstrap * len(groups)),
            "_present": 1,
            "_undefined": 0,
            "_viable": True,
        }
    ).sample(fraction=1.0, shuffle=True, seed=0)
    alpha = 0.025
    got = (
        _percentile_bounds(joined.lazy(), ["g"], "v", n_bootstrap, alpha)
        .collect(engine="streaming")
        .sort("g")
    )
    want = (
        joined.group_by("g")
        .agg(
            ci_lower=pl.col("v").quantile(alpha, "linear"),
            ci_upper=pl.col("v").quantile(1.0 - alpha, "linear"),
        )
        .sort("g")
    )
    assert got["ci_lower"].to_list() == want["ci_lower"].to_list()
    assert got["ci_upper"].to_list() == want["ci_upper"].to_list()
    ungrouped = _percentile_bounds(
        joined.filter(pl.col("g") == "a").lazy(), [], "v", n_bootstrap, alpha
    ).collect()
    assert ungrouped["ci_lower"].to_list() == want["ci_lower"].to_list()[:1]


class TestReplicateBatches:
    """Batching the replicates bounds memory and changes no bit of any bound.

    Each batch draws its own ``bootstrap_id`` range, and a draw hashes its
    global slot id, so the batches are slices of the one whole-range draw.
    """

    @staticmethod
    def _table() -> DetectionTable:
        rng = np.random.default_rng(3)
        n_img, per = 40, 4
        dets = pl.DataFrame(
            {
                "image_id": np.repeat([f"i{i}" for i in range(n_img)], per),
                "class_id": rng.choice(["a", "b"], n_img * per),
                "score": rng.random(n_img * per),
                "is_tp": rng.random(n_img * per) < 0.4,
            }
        )
        images = pl.DataFrame(
            {
                "image_id": [f"i{i}" for i in range(n_img)],
                "weight": rng.choice([0.5, 1.0, 2.0], n_img),
            }
        )
        meta = images.join(pl.DataFrame({"class_id": ["a", "b"]}), how="cross")
        meta = meta.with_columns(n_gts=pl.Series(rng.integers(0, 3, meta.height)))
        return PreMatchedAdapter().match(
            dets,
            image_id_col="image_id",
            class_col="class_id",
            image_meta=meta,
        )

    @pytest.mark.parametrize(
        "ci",
        [
            pytest.param(
                lambda t: M.bootstrap_ci(t, M.mean_ap(), n_bootstrap=23, seed=5),
                id="mean_ap",
            ),
            pytest.param(
                lambda t: M.average_precision_ci_lazy(
                    t, group_by="class_id", n_bootstrap=23, seed=5, strata="weight"
                ),
                id="ap-grouped-stratified",
            ),
            pytest.param(
                lambda t: M.froc_auc_ci_lazy(
                    t, n_bootstrap=23, seed=5, fp_range=(0.0, 1.0), extrapolate="flat"
                ),
                id="froc",
            ),
        ],
    )
    def test_any_batch_size_gives_the_same_bounds(
        self, ci, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from polars_cv.metrics import _bootstrap

        results = []
        for batch in (None, 1, 7, 50):
            monkeypatch.setattr(_bootstrap, "_REPLICATES_PER_BATCH", batch)
            results.append(ci(self._table()).collect().sort(pl.all()))
        for got in results[1:]:
            assert_frame_equal(got, results[0])
