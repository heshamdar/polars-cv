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


def _vendor_table() -> DetectionTable:
    """Two groups of ten images with different vendor mixes (g1 6A/4B, g2 4A/6B).

    Unit weights; :func:`_importance_weighted` sets ``p / q̂``. Each ``case_id``
    pairs two images of one vendor, so an entity-level draw keeps the image mix.
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


def _importance_weighted(scope: str) -> DetectionTable:
    """``_vendor_table`` weighted to ``_TARGET``, ``q̂`` per group or global."""
    table = _vendor_table()
    over = ["group_id"] if scope == "per_group" else []
    return _with_meta(table, table.image_metadata.with_columns(_p_over_q(over)))


def _replicates(
    table: DetectionTable,
    *,
    group_keys: list[str],
    sample_col: str | None = None,
    strata: list[str] | None = (),  # type: ignore[assignment]
    n_bootstrap: int = 60,
) -> DetectionTable:
    """The CI path's replicate table; ``strata=None`` is the unstratified draw."""
    from polars_cv.metrics._bootstrap import (
        _bootstrap_table_with_draws,
        _resolve_bootstrap_samples,
        _sampling_cells,
    )

    cells = (
        None
        if strata is None
        else _sampling_cells(
            table, sample_col=sample_col, group_keys=group_keys, strata=list(strata)
        )
    )
    samples = _resolve_bootstrap_samples(
        table,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        seed=3,
        group_keys=group_keys,
        cells=cells,
    )
    return _bootstrap_table_with_draws(table, samples)


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

    @pytest.mark.parametrize("family", ["froc", "lroc"])
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

    @pytest.mark.parametrize("family", ["froc", "lroc"])
    def test_unknown_strata_column_raises(self, family: str) -> None:
        fn, _ = _CI_FUNCS[family]
        with pytest.raises(ValueError, match="strata.*'scanner'"):
            fn(_vendor_table(), strata="scanner", n_bootstrap=10, seed=1)

    def test_average_precision_is_not_weight_stratified(self) -> None:
        # AP never reads `weight`, so its weight cells are no part of the
        # estimator: distinct weights neither change nor null its bounds.
        table = _vendor_table()
        distinct = (pl.int_range(pl.len()).cast(pl.Float64) + 1.0).alias(COL_WEIGHT)
        weighted = _with_meta(table, table.image_metadata.with_columns(distinct))
        a = average_precision_ci_lazy(
            table, group_by="group_id", n_bootstrap=100, seed=4
        )
        b = average_precision_ci_lazy(
            weighted, group_by="group_id", n_bootstrap=100, seed=4
        )
        out = b.collect().sort("group_id")
        assert out.equals(a.collect().sort("group_id"))
        assert out["ci_lower"].null_count() == 0

    @pytest.mark.parametrize("family", ["froc", "lroc"])
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
