"""The ``Statistic`` layer: one grouped estimator per metric.

Every public metric that reads a ``DetectionTable`` is either a reading of a
:class:`~polars_cv.metrics.Statistic` (and must agree with it) or is declared
here as something else (a curve, a result object, a CI). A new public metric
that is neither fails :func:`test_every_public_metric_is_classified`, so the
grouped/bootstrap path cannot silently miss it.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable

import polars as pl
import pytest

import polars_cv.metrics as M
from polars_cv.metrics import (
    AP,
    CPM,
    FROCAUC,
    LROCAUC,
    DetectionTable,
    F1At,
    FROCSensitivity,
    LROCSensitivity,
    MeanOver,
    PrecisionAt,
    Recall,
    RecallAt,
    Statistic,
    bootstrap_ci,
    mean_ap,
)

# ---------------------------------------------------------------------------
# Fixture tables
# ---------------------------------------------------------------------------


def _table(
    dets: list[tuple[str, str, float, bool]],
    meta: list[tuple[str, str, int]],
    weights: dict[str, float] | None = None,
) -> DetectionTable:
    """``dets``: (image, class, score, is_tp); ``meta``: (image, class, n_gts)."""
    w = weights or {}
    det = pl.DataFrame(
        {
            "image_id": [d[0] for d in dets],
            "class_id": [d[1] for d in dets],
            "score": [d[2] for d in dets],
            "is_tp": [d[3] for d in dets],
            "gt_idx": [0 if d[3] else None for d in dets],
            "iou": [0.8 if d[3] else None for d in dets],
            "det_idx": list(range(len(dets))),
        },
        schema={
            "image_id": pl.String,
            "class_id": pl.String,
            "score": pl.Float64,
            "is_tp": pl.Boolean,
            "gt_idx": pl.UInt32,
            "iou": pl.Float64,
            "det_idx": pl.UInt32,
        },
    )
    md = pl.DataFrame(
        {
            "image_id": [m[0] for m in meta],
            "class_id": [m[1] for m in meta],
            "n_gts": [m[2] for m in meta],
            "weight": [w.get(m[0], 1.0) for m in meta],
            "gt_label": [m[2] > 0 for m in meta],
            "group_id": ["g" + str(int(m[0][1:]) % 2) for m in meta],
        },
        schema_overrides={"n_gts": pl.Int64},
    )
    return DetectionTable.from_matched(det, md)


def _multiclass(weighted: bool = False) -> DetectionTable:
    dets = [
        ("i0", "cat", 0.95, True),
        ("i0", "cat", 0.60, False),
        ("i0", "dog", 0.90, True),
        ("i1", "cat", 0.80, True),
        ("i1", "cat", 0.80, False),  # a tie mixing TP and FP
        ("i1", "dog", 0.40, False),
        ("i2", "cat", 0.30, False),
        ("i2", "dog", 0.85, True),
        ("i2", "dog", 0.20, True),
        ("i3", "dog", 0.70, False),
        ("i4", "cat", 0.50, True),
        ("i5", "dog", 0.65, True),
    ]
    meta = [
        ("i0", "cat", 2),
        ("i0", "dog", 1),
        ("i1", "cat", 1),
        ("i1", "dog", 0),
        ("i2", "cat", 1),
        ("i2", "dog", 3),
        ("i3", "cat", 0),
        ("i3", "dog", 0),
        ("i4", "cat", 2),
        ("i4", "dog", 1),
        ("i5", "cat", 0),
        ("i5", "dog", 1),
    ]
    weights = {"i0": 2.0, "i1": 0.5, "i4": 3.0} if weighted else None
    return _table(dets, meta, weights)


TABLES = {"unit": _multiclass(), "weighted": _multiclass(weighted=True)}


def _scalar(frame: pl.LazyFrame, col: str) -> float:
    v = frame.collect().item(0, col)
    return float("nan") if v is None else float(v)


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# Classification of the public metric surface
# ---------------------------------------------------------------------------

#: Public functions that are a reading of a statistic:
#: name -> (call on a table, the statistic it reads).
STATISTIC_VIEWS: dict[str, tuple[Callable[[DetectionTable], float], Statistic]] = {
    "average_precision": (lambda t: M.average_precision(t), AP()),
    "precision_at_threshold": (
        lambda t: M.precision_at_threshold(t, 0.6),
        PrecisionAt(0.6),
    ),
    "recall_at_threshold": (lambda t: M.recall_at_threshold(t, 0.6), RecallAt(0.6)),
    "f1_at_threshold": (lambda t: M.f1_at_threshold(t, 0.6), F1At(0.6)),
    "froc_auc": (
        lambda t: _scalar(M.froc_auc(t, fp_range=(0.0, 0.5)), "auc"),
        FROCAUC(fp_range=(0.0, 0.5)),
    ),
    "froc_sensitivity_at_fp": (
        lambda t: _scalar(M.froc_sensitivity_at_fp(t, 0.25), "sensitivity"),
        FROCSensitivity(0.25),
    ),
    "lroc_auc": (lambda t: _scalar(M.lroc_auc(t), "auc"), LROCAUC()),
    "lroc_sensitivity_at_fpf": (
        lambda t: _scalar(M.lroc_sensitivity_at_fpf(t, 0.3), "sensitivity"),
        LROCSensitivity(0.3),
    ),
    "mean_average_precision": (
        lambda t: M.mean_average_precision(t),
        mean_ap(undefined="zero"),
    ),
}

#: Public functions taking a DetectionTable that are not one statistic, and why.
NOT_A_STATISTIC = {
    "precision_recall_curve": "a curve (PrecisionRecallResult)",
    "froc_curve_lazy": "a curve",
    "lroc_curve_lazy": "a curve",
    "froc_summary_table": "FROCSensitivity at several rates (tested below)",
    "froc_operating_range": "the curve's extent, not a metric",
    "confusion_at_threshold": "raw counts (ConfusionResult)",
    "bootstrap_ci": "the CI engine over statistics",
    "froc_auc_ci_lazy": "bootstrap_ci(FROCAUC)",
    "lroc_auc_ci_lazy": "bootstrap_ci(LROCAUC)",
    "average_precision_ci_lazy": "bootstrap_ci(AP)",
}


def _takes_a_table(obj: object) -> bool:
    if not inspect.isfunction(obj):
        return False
    params = list(inspect.signature(obj).parameters.values())
    return bool(params) and "DetectionTable" in str(params[0].annotation)


def test_every_public_metric_is_classified() -> None:
    public = {n for n in M.__all__ if _takes_a_table(getattr(M, n))}
    assert public, "no public metric found: the scan matched nothing"
    unclassified = public - set(STATISTIC_VIEWS) - set(NOT_A_STATISTIC)
    assert not unclassified, (
        f"{sorted(unclassified)} take a DetectionTable but are neither a reading "
        "of a Statistic (STATISTIC_VIEWS) nor declared otherwise "
        "(NOT_A_STATISTIC). Give the metric a Statistic so it can be grouped "
        "and bootstrapped."
    )
    stale = (set(STATISTIC_VIEWS) | set(NOT_A_STATISTIC)) - public
    assert not stale, f"classified but not public: {sorted(stale)}"


@pytest.mark.parametrize("table_kind", sorted(TABLES))
@pytest.mark.parametrize("name", sorted(STATISTIC_VIEWS))
def test_a_public_metric_agrees_with_its_statistic(name: str, table_kind: str) -> None:
    call, statistic = STATISTIC_VIEWS[name]
    table = TABLES[table_kind]
    assert _same(call(table), statistic.value(table))


ALL_STATISTICS: list[Statistic] = [
    AP(),
    AP("11_point"),
    AP("101_point"),
    Recall(),
    PrecisionAt(0.6),
    RecallAt(0.6),
    F1At(0.6),
    FROCSensitivity(0.25),
    FROCSensitivity(0.25, extrapolate="flat"),
    CPM(extrapolate="flat"),
    FROCAUC(fp_range=(0.0, 0.5)),
    FROCAUC(method="mann_whitney"),
    LROCAUC(),
    LROCSensitivity(0.3),
]


@pytest.mark.parametrize("table_kind", sorted(TABLES))
@pytest.mark.parametrize("statistic", ALL_STATISTICS, ids=repr)
def test_a_group_equals_its_filtered_sub_table(
    statistic: Statistic, table_kind: str
) -> None:
    """Grouping is filtering: the row for a class is the statistic on that
    class's sub-table, for every statistic."""
    table = TABLES[table_kind]
    grouped = statistic.by_group(table, "class_id").collect()
    for cid in ("cat", "dog"):
        row = grouped.filter(pl.col("class_id") == cid)
        got = row.item(0, statistic.name) if row.height else statistic.empty_value
        got = float("nan") if got is None else float(got)
        assert _same(got, statistic.value(table.filter_class(cid))), cid


@pytest.mark.parametrize("statistic", ALL_STATISTICS, ids=repr)
def test_a_metadata_only_key_groups(statistic: Statistic) -> None:
    table = TABLES["weighted"]
    grouped = statistic.by_group(table, "group_id").collect()
    for gid in ("g0", "g1"):
        row = grouped.filter(pl.col("group_id") == gid)
        sub = table.filter_images(pl.col("group_id") == gid)
        got = row.item(0, statistic.name) if row.height else statistic.empty_value
        got = float("nan") if got is None else float(got)
        assert _same(got, statistic.value(sub)), gid


def test_froc_summary_table_reads_froc_sensitivity() -> None:
    table = TABLES["weighted"]
    rates = [0.125, 0.25, 0.5]
    summary = M.froc_summary_table(table, rates, group_by="class_id").collect()
    for cid in ("cat", "dog"):
        for rate in rates:
            got = summary.filter(
                (pl.col("class_id") == cid) & (pl.col("fp_per_image") == rate)
            ).item(0, "sensitivity")
            got = float("nan") if got is None else float(got)
            want = FROCSensitivity(rate).value(table.filter_class(cid))
            assert _same(got, want), (cid, rate)


def test_cpm_is_the_mean_sensitivity_over_its_rates() -> None:
    table = TABLES["unit"]
    rates = (0.125, 0.25, 0.5)
    sens = [FROCSensitivity(r, "flat").value(table) for r in rates]
    assert CPM(rates, "flat").value(table) == pytest.approx(sum(sens) / len(sens))


def test_cpm_off_the_curve_is_null_not_averaged() -> None:
    assert math.isnan(CPM((0.125, 100.0)).value(TABLES["unit"]))


# ---------------------------------------------------------------------------
# MeanOver
# ---------------------------------------------------------------------------


def _with_gtless_class() -> DetectionTable:
    """``bird`` has predictions but no ground truth anywhere."""
    return _table(
        [
            ("i0", "cat", 0.9, True),
            ("i0", "bird", 0.8, False),
            ("i1", "cat", 0.7, False),
        ],
        [
            ("i0", "cat", 1),
            ("i0", "bird", 0),
            ("i1", "cat", 0),
            ("i1", "bird", 0),
        ],
    )


def test_mean_over_classes_by_policy() -> None:
    table = _with_gtless_class()
    cat = AP().value(table.filter_class("cat"))
    assert cat == pytest.approx(1.0)
    # COCO: a class with no ground truth is not averaged in.
    assert mean_ap().value(table) == pytest.approx(cat)
    # The historical mean_average_precision convention averages it in as 0.
    assert mean_ap(undefined="zero").value(table) == pytest.approx(cat / 2)
    assert M.mean_average_precision(table) == pytest.approx(cat / 2)


def test_a_class_with_truth_but_no_detections_scores_zero_either_way() -> None:
    table = _table(
        [("i0", "cat", 0.9, True)],
        [("i0", "cat", 1), ("i0", "dog", 2)],
    )
    for policy in ("exclude", "zero"):
        assert mean_ap(undefined=policy).value(table) == pytest.approx(0.5)


def test_mean_over_names_and_per_facet() -> None:
    table = TABLES["unit"]
    stat = MeanOver(Recall(), undefined="exclude")
    assert stat.name == "mean_recall"
    per = stat.per_facet(table).collect().sort("class_id")
    assert per["class_id"].to_list() == ["cat", "dog"]
    assert stat.value(table) == pytest.approx(per["recall"].mean())


# ---------------------------------------------------------------------------
# bootstrap_ci over statistics
# ---------------------------------------------------------------------------


def test_the_ci_wrappers_are_bootstrap_ci() -> None:
    table = TABLES["weighted"]
    for wrapper, statistic, kwargs in [
        (M.average_precision_ci_lazy, AP(), {}),
        (M.froc_auc_ci_lazy, FROCAUC(fp_range=(0.0, 0.5)), {"fp_range": (0.0, 0.5)}),
        (M.lroc_auc_ci_lazy, LROCAUC(), {}),
    ]:
        a = wrapper(table, group_by="class_id", n_bootstrap=50, seed=3, **kwargs)
        b = bootstrap_ci(table, statistic, group_by="class_id", n_bootstrap=50, seed=3)
        assert a.collect().sort("class_id").equals(b.collect().sort("class_id"))


def test_a_map_interval_brackets_its_point_and_is_reproducible() -> None:
    # Unit weights: the weighted fixture has single-image weight cells, whose
    # bounds the engine nulls by design (no bootstrap variance in a cell of 1).
    table = TABLES["unit"]
    stat = mean_ap("101_point")
    a = bootstrap_ci(table, stat, n_bootstrap=200, seed=11).collect()
    b = bootstrap_ci(table, stat, n_bootstrap=200, seed=11).collect()
    assert a.equals(b)
    row = a.row(0, named=True)
    assert row["map"] == pytest.approx(stat.value(table))
    assert row["ci_lower"] <= row["map"] <= row["ci_upper"]


def test_facets_are_paired_within_a_replicate() -> None:
    """A replicate's mAP is the mean of its per-class APs *on the same draws*:
    each drawn image brings every class row, so the classes of one replicate
    come from one resample (not one resample per class)."""
    from polars_cv.metrics._bootstrap import _COL_BOOT, _replicate_tables

    table = TABLES["weighted"]
    (boot,), _ = _replicate_tables(
        table,
        group_keys=[],
        sample_col=None,
        n_bootstrap=30,
        seed=5,
        strata=[],
        weight_rtol=1e-6,
        weight_scheme="reestimate",
        batch=None,
    )
    _, meta = boot.frames([_COL_BOOT])
    meta = meta.collect()
    # Every draw (synthetic image id) carries both of its source image's classes.
    per_draw = meta.group_by(_COL_BOOT, "image_id").agg(pl.col("class_id").n_unique())
    assert per_draw["class_id"].min() == 2

    stat = mean_ap(undefined="exclude")
    outer = stat.by_group(boot, _COL_BOOT).collect().sort(_COL_BOOT)
    inner = stat.per_facet(boot, _COL_BOOT).collect()
    by_hand = inner.group_by(_COL_BOOT).agg(pl.col("ap").mean()).sort(_COL_BOOT)
    assert outer["map"].to_list() == pytest.approx(by_hand["ap"].to_list())
