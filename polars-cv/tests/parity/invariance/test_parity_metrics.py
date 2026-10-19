"""The metrics subsystem, over drawn detection tables.

``polars_cv.metrics`` is pure Polars (no compiled plugin), so these properties
need no build. Each draws a *frame* of detections and an image population, wraps
it through ``PreMatchedAdapter`` and asks two questions:

* **Oracle** — is the number right? AP, the PR curve and the threshold metrics
  against a small NumPy reference written here; FROC / LROC AUC (trapezoidal and
  Mann-Whitney) against ``tests/_metric_refs.py``. Compared to 1e-12 relative.
* **Invariance** — does the number depend on how it was asked? Row order,
  chunking, the collecting engine, a strictly increasing transform of the
  scores, grouping against filtering, an integer weight against duplicated
  images, and the bootstrap's seed.

The frames are drawn to be awkward on purpose: images with no detections, images
with no ground truth, scores sampled from a three-value set (so ties are the
rule), on a 1/1024 grid, or arbitrary floats, and image weights that are unit,
integer or dyadic. Weights and grid scores are exactly representable, so every
sum the metrics take is exact and the *invariance* properties compare with
``==`` (bit for bit, as the exact-sums design promises); only the oracle
comparisons, whose references accumulate differently, use a relative tolerance.

Scope notes (what is deliberately not covered here):

* ``level="image"`` FROC Mann-Whitney has no independent reference in
  ``_metric_refs.py``; it is covered by the invariance properties only.
* FROC / LROC references are single-class (pooled) by construction, so the
  oracle properties draw one class; the multi-class behaviour is held by the
  grouped-equals-filtered property.
* The bootstrap's *bounds* have no oracle (they are a percentile of random
  draws); only reproducibility, engine and chunk invariance and ``lower <=
  upper`` are asserted. ``ci_lower <= point <= ci_upper`` is not guaranteed.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import example
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from polars_cv.metrics import (
    AP,
    FROCAUC,
    LROCAUC,
    DetectionTable,
    F1At,
    PrecisionAt,
    PreMatchedAdapter,
    Recall,
    RecallAt,
    Statistic,
    average_precision,
    average_precision_ci_lazy,
    bootstrap_ci,
    f1_at_threshold,
    froc_auc,
    froc_auc_ci_lazy,
    lroc_auc,
    lroc_auc_ci_lazy,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
)
from tests._metric_refs import (
    ref_froc_auc,
    ref_froc_mw_detection,
    ref_lroc_auc,
    ref_lroc_image_mw,
)
from tests.parity.framework.budget import property_lanes

CLASSES = ("a", "b", "c")
GROUPS = ("g0", "g1")
#: Scores that collide on purpose.
TIE_SCORES = (0.2, 0.5, 0.8)
#: Weights exactly representable, so sums over them are exact in any order.
DYADIC_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
INTERPOLATIONS = ("all_points", "11_point", "101_point")
REL = 1e-12


# ---------------------------------------------------------------------------
# Drawn frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A drawn evaluation: detections (one row each) and the image population."""

    dets: pl.DataFrame  # image_id, class_id, score, is_tp, det_idx
    meta: pl.DataFrame  # image_id, class_id, n_gts, [weight], group_id

    @property
    def weighted(self) -> bool:
        return "weight" in self.meta.columns

    def image_weights(self) -> dict[str, float]:
        if not self.weighted:
            return dict.fromkeys(self.meta["image_id"].to_list(), 1.0)
        pairs = zip(self.meta["image_id"], self.meta["weight"])
        return {i: float(w) for i, w in pairs}


def build(
    frame: Frame,
    *,
    det_idx: bool = False,
    dets: pl.DataFrame | None = None,
    meta: pl.DataFrame | None = None,
) -> DetectionTable:
    """The one way every property wraps a frame, through ``PreMatchedAdapter``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # image_meta is always given: no warning
        return PreMatchedAdapter().match(
            frame.dets if dets is None else dets,
            image_id_col="image_id",
            class_col="class_id",
            image_meta=frame.meta if meta is None else meta,
            det_idx_col="det_idx" if det_idx else None,
        )


def _score(kind: str, draw: st.DrawFn) -> float:
    ties = st.sampled_from(TIE_SCORES)
    grid = st.integers(1, 1023).map(lambda k: k / 1024)
    floats = st.floats(0.001, 0.999, allow_nan=False, allow_infinity=False)
    strategy = {
        "ties": ties,
        "grid": grid,
        "floats": floats,
        "mixed": st.one_of(ties, grid),
        "any": st.one_of(ties, grid, floats),
    }[kind]
    return draw(strategy)


@st.composite
def frames(
    draw: st.DrawFn,
    *,
    weights: str = "dyadic",
    single_class: bool = False,
    both_labels: bool = False,
    need_gt: bool = False,
    scores: str = "mixed",
    max_images: int = 6,
    sort_by_score: bool = False,
) -> Frame:
    """Draw a frame.

    Args:
        weights: ``"none"`` (no weight column), ``"unit"``, ``"int"`` (1-3) or
            ``"dyadic"``. One weight per image, shared by its classes.
        single_class: One class only (what the FROC / LROC references take).
        both_labels: Force a positive and a negative image (LROC's rank
            statistics are undefined without both).
        need_gt: Force at least one ground truth.
        scores: ``"ties"``, ``"grid"``, ``"floats"``, ``"mixed"`` or ``"any"``.
        sort_by_score: Order detections by descending score (the order the
            LROC reference reads a "first" detection in).
    """
    n_images = draw(st.integers(2 if both_labels else 1, max_images))
    n_classes = 1 if single_class else draw(st.integers(1, len(CLASSES)))
    classes = CLASSES[:n_classes]
    weight_choices = {
        "unit": (1.0,),
        "int": (1.0, 2.0, 3.0),
        "dyadic": DYADIC_WEIGHTS,
        "none": (1.0,),
    }[weights]

    image_ids = [f"img{i}" for i in range(n_images)]
    group_of = {i: draw(st.sampled_from(GROUPS)) for i in image_ids}
    weight_of = {i: draw(st.sampled_from(weight_choices)) for i in image_ids}

    meta_rows: list[dict[str, Any]] = []
    det_rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        for class_id in classes:
            n_gts = draw(st.integers(0, 3))
            meta_rows.append(
                {
                    "image_id": image_id,
                    "class_id": class_id,
                    "n_gts": n_gts,
                    "weight": weight_of[image_id],
                    "group_id": group_of[image_id],
                }
            )
            tps = 0
            for _ in range(draw(st.integers(0, 4))):  # 0 detections is common
                is_tp = draw(st.booleans())
                if is_tp and tps >= n_gts:
                    is_tp = False  # a TP claims a distinct ground truth
                tps += is_tp
                det_rows.append(
                    {
                        "image_id": image_id,
                        "class_id": class_id,
                        "score": _score(scores, draw),
                        "is_tp": is_tp,
                    }
                )

    def set_gts(image_id: str, n_gts: int) -> None:
        for row in meta_rows:
            if row["image_id"] == image_id and row["class_id"] == classes[0]:
                row["n_gts"] = n_gts
        if n_gts == 0:  # a negative image has no TP
            for d in det_rows:
                if d["image_id"] == image_id and d["class_id"] == classes[0]:
                    d["is_tp"] = False

    if both_labels:
        # one image positive, another negative, in every class
        for row in meta_rows:
            if row["image_id"] == image_ids[0]:
                row["n_gts"] = max(row["n_gts"], 1)
            if row["image_id"] == image_ids[1]:
                row["n_gts"] = 0
        for d in det_rows:
            if d["image_id"] == image_ids[1]:
                d["is_tp"] = False
    elif need_gt and not any(r["n_gts"] for r in meta_rows):
        set_gts(image_ids[0], 1)

    if sort_by_score:
        det_rows.sort(key=lambda d: -d["score"])  # stable: ties keep draw order

    dets = pl.DataFrame(
        det_rows,
        schema={
            "image_id": pl.String,
            "class_id": pl.String,
            "score": pl.Float64,
            "is_tp": pl.Boolean,
        },
    )
    # The ordinal of each detection within its (image, class), in frame order.
    dets = dets.with_columns(
        pl.int_range(pl.len())
        .over("image_id", "class_id")
        .cast(pl.UInt32)
        .alias("det_idx")
    )
    meta = pl.DataFrame(
        meta_rows,
        schema={
            "image_id": pl.String,
            "class_id": pl.String,
            "n_gts": pl.Int64,
            "weight": pl.Float64,
            "group_id": pl.String,
        },
    )
    if weights == "none":
        meta = meta.drop("weight")
    return Frame(dets=dets, meta=meta)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _missing(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def same(a: float | None, b: float | None) -> bool:
    """Bit-for-bit equality; null and NaN both read as "undefined"."""
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return a == b


def close(a: float | None, b: float | None, rel: float = REL) -> bool:
    """Relative closeness (1e-12); undefined only matches undefined."""
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return abs(a - b) <= rel * max(abs(a), abs(b))


def scalar(lf: pl.LazyFrame | pl.DataFrame) -> float | None:
    df = lf.collect() if isinstance(lf, pl.LazyFrame) else lf
    return df.item() if df.height else None


def scalar_or_empty(lf: pl.LazyFrame, empty: float) -> float | None:
    """The scalar, or the statistic's documented ``empty_value`` for no row.

    A table without a single detection yields no row at all
    (``Statistic.value`` documents it as ``empty_value``: 0.5 for the rank
    statistics, which "a resample that drew no detections legitimately scores").
    """
    df = lf.collect()
    return df.item() if df.height else empty


def collect_sorted(
    stat: Statistic,
    table: DetectionTable,
    keys: str | list[str] | None = None,
    engine: str | None = None,
) -> pl.DataFrame:
    lf = stat.by_group(table, keys)
    out = lf.collect() if engine is None else lf.collect(engine=engine)
    order = out.columns[:-1] if keys else []
    return out.sort(order) if order else out


def assert_same_frames(a: pl.DataFrame, b: pl.DataFrame, what: str = "") -> None:
    try:
        assert_frame_equal(a, b, check_exact=True, check_row_order=True)
    except AssertionError as exc:
        raise AssertionError(f"{what}\n{a}\nvs\n{b}\n{exc}") from None


# ---------------------------------------------------------------------------
# Statistics under test (every one a Statistic: by_group / bootstrap-able)
# ---------------------------------------------------------------------------

STATS: dict[str, Statistic] = {
    "ap": AP(),
    "ap11": AP("11_point"),
    "ap101": AP("101_point"),
    "recall": Recall(),
    "precision_at": PrecisionAt(0.5),
    "recall_at": RecallAt(0.5),
    "f1_at": F1At(0.5),
    "froc_trap": FROCAUC(fp_range=(0.0, 2.0), extrapolate="flat"),
    "froc_trap_raw": FROCAUC(fp_range=(0.5, 3.0), correction=None, extrapolate="flat"),
    "froc_mw": FROCAUC(method="mann_whitney"),
    "froc_mw_image": FROCAUC(method="mann_whitney", level="image"),
    "lroc_trap": LROCAUC(),
    "lroc_trap_range": LROCAUC(fpf_range=(0.25, 0.75), correction=None),
    "lroc_top": LROCAUC(variant="top_scoring"),
    "lroc_mw_image": LROCAUC(method="mann_whitney", level="image"),
    "lroc_mw_det": LROCAUC(method="mann_whitney", level="detection"),
}
#: Statistics that read only the *ranking* of the scores (every one that does
#: not compare a score with a fixed threshold).
RANK_STATS = sorted(k for k in STATS if not k.endswith("_at"))
#: ... and whose value does not depend on which of two tied detections is first.
ORDER_FREE_STATS = sorted(k for k in STATS if k != "lroc_top")

stat_names = st.sampled_from(sorted(STATS))


def _cut(df: pl.DataFrame, parts: int) -> pl.DataFrame:
    """The frame as *parts* separate chunks (no rechunk)."""
    if df.height == 0 or parts <= 1:
        return df
    bounds = np.linspace(0, df.height, parts + 1).astype(int)
    slices = [df.slice(int(a), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:])]
    return pl.concat(slices, rechunk=False)


# ---------------------------------------------------------------------------
# Oracle: AP, PR curve and threshold metrics (NumPy reference)
# ---------------------------------------------------------------------------


def ref_pr_points(
    frame: Frame, class_id: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-distinct-score PR points of the pooled (or one-class) detections.

    Returns ``(score, precision, recall, cum_tp, cum_fp, gt_mass)``, score
    descending, one point after each whole tied block.
    """
    dets, meta = frame.dets, frame.meta
    if class_id is not None:
        dets = dets.filter(pl.col("class_id") == class_id)
        meta = meta.filter(pl.col("class_id") == class_id)
    weight = frame.image_weights()
    gt_mass = float(sum(n * weight[i] for i, n in zip(meta["image_id"], meta["n_gts"])))
    rows = [
        (s, tp, weight[i])
        for i, s, tp in zip(dets["image_id"], dets["score"], dets["is_tp"])
    ]
    scores = sorted({s for s, _, _ in rows}, reverse=True)
    out = []
    wtp = wfp = 0.0
    ctp = cfp = 0
    for s in scores:
        for ss, tp, w in rows:
            if ss == s:
                if tp:
                    wtp += w
                    ctp += 1
                else:
                    wfp += w
                    cfp += 1
        out.append(
            (s, wtp / (wtp + wfp), wtp / gt_mass if gt_mass else math.nan, ctp, cfp)
        )
    if not out:
        z = np.array([])
        return z, z, z, z, z, gt_mass
    s, p, r, a, b = (np.array(c, dtype=float) for c in zip(*out))
    return s, p, r, a, b, gt_mass


def ref_ap(precision: np.ndarray, recall: np.ndarray, interpolation: str) -> float:
    """AP from a PR curve: step-integrated envelope, or its N-point mean."""
    if precision.size == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    if interpolation == "all_points":
        previous = np.concatenate([[0.0], recall[:-1]])
        return float(np.sum((recall - previous) * envelope))
    n = {"11_point": 11, "101_point": 101}[interpolation]
    total = 0.0
    for t in np.linspace(0.0, 1.0, n):
        reached = recall >= t
        total += float(envelope[reached][0]) if reached.any() else 0.0
    return total / n


def _reference_ap(frame: Frame, interpolation: str, class_id: str | None) -> float:
    _, precision, recall, *_, gt_mass = ref_pr_points(frame, class_id)
    if gt_mass == 0.0:
        return 0.0  # no ground truth: nothing to average
    return ref_ap(precision, recall, interpolation)


def _known_class(frame: Frame, class_id: str | None) -> str | None:
    present = frame.meta["class_id"].to_list()
    return class_id if class_id in present else None


#: A recall that lands on an N-point grid value: 3 / 5 is 0.6 and the grid's
#: 6 * 0.1 is 0.6000000000000001, so the recall does not reach it. Weighted
#: image img2 (w=3) holds the only TP, img0 (w=1, two GTs) a miss.
GRID_EDGE = Frame(
    dets=pl.DataFrame(
        {
            "image_id": ["img0", "img0", "img2", "img2"],
            "class_id": ["a"] * 4,
            "score": [0.2, 0.2, 0.5, 0.8],
            "is_tp": [False, False, True, False],
            "det_idx": pl.Series([0, 1, 0, 1], dtype=pl.UInt32),
        }
    ),
    meta=pl.DataFrame(
        {
            "image_id": ["img0", "img1", "img2", "img3"],
            "class_id": ["a"] * 4,
            "n_gts": [2, 0, 1, 0],
            "weight": [1.0, 0.25, 3.0, 0.25],
            "group_id": ["g0"] * 4,
        }
    ),
)


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic", scores="any"),
    interpolation=st.sampled_from(INTERPOLATIONS),
    class_id=st.sampled_from([None, *CLASSES]),
)
def test_ap_statistic_and_pr_curve_match_reference(
    frame: Frame, interpolation: str, class_id: str | None
) -> None:
    """``AP`` (the grouped authority) and the PR curve against a NumPy AP."""
    class_id = _known_class(frame, class_id)
    table = build(frame)
    _, precision, recall, ctp, cfp, gt_mass = ref_pr_points(frame, class_id)
    if gt_mass > 0.0:  # without ground truth: test_ap_without_ground_truth_...
        want = _reference_ap(frame, interpolation, class_id)
        scoped = table if class_id is None else table.filter_class(class_id)
        got = scalar(AP(interpolation).by_group(scoped).select("ap").collect())
        assert close(got if got is not None else 0.0, want), (got, want)

    # the curve itself, one point per distinct score
    curve = precision_recall_curve(table, class_id=class_id).curve
    if gt_mass == 0.0 or precision.size == 0:
        assert curve.height == 0
        return
    assert curve["precision"].to_list() == pytest.approx(precision, rel=REL, abs=0)
    assert curve["recall"].to_list() == pytest.approx(recall, rel=REL, abs=0)
    assert curve["cum_tp"].to_list() == ctp.astype(int).tolist()
    assert curve["cum_fp"].to_list() == cfp.astype(int).tolist()


#: One false positive, no ground truth anywhere.
NO_GT = Frame(
    dets=pl.DataFrame(
        {
            "image_id": ["img0"],
            "class_id": ["a"],
            "score": [0.2],
            "is_tp": [False],
            "det_idx": pl.Series([0], dtype=pl.UInt32),
        }
    ),
    meta=pl.DataFrame(
        {
            "image_id": ["img0"],
            "class_id": ["a"],
            "n_gts": [0],
            "weight": [0.25],
            "group_id": ["g0"],
        }
    ),
)


@property_lanes(
    weight=1,
    marks=(
        pytest.mark.xfail(
            strict=True,
            reason=(
                "LIBRARY BUG (inconsistency): on a table with detections and no "
                "ground truth, average_precision() is 0.0 but the same "
                "estimator as a Statistic, AP().by_group(), is NaN (recall is "
                "0/0). metrics/AGENTS.md calls them 'identical on every curve'."
            ),
        ),
    ),
    frame=frames(weights="dyadic", scores="any").map(
        lambda f: Frame(
            dets=f.dets.with_columns(pl.lit(False).alias("is_tp")),
            meta=f.meta.with_columns(pl.lit(0).alias("n_gts")),
        )
    ),
    interpolation=st.sampled_from(INTERPOLATIONS),
)
@example(frame=NO_GT, interpolation="all_points")
def test_ap_without_ground_truth_is_the_same_everywhere(
    frame: Frame, interpolation: str
) -> None:
    table = build(frame)
    as_function = average_precision(table, interpolation=interpolation)
    as_statistic = AP(interpolation).value(table)
    assert same(as_function, as_statistic), (as_function, as_statistic)


@property_lanes(
    weight=2,
    marks=(
        pytest.mark.xfail(
            strict=True,
            reason=(
                "LIBRARY BUG: precision_recall_curve computes recall as "
                "`cum_weighted_tp / pl.lit(gt_mass)`, which polars evaluates as a "
                "multiplication by the reciprocal (3 * 0.2 = 0.6000000000000001, "
                "not 3 / 5 = 0.6). A recall on an N-point grid value then "
                "compares differently from the grouped authority's, so "
                "average_precision(t, interpolation='11_point') != "
                "AP('11_point').by_group(t) (and a numpy/COCO reference) on the "
                "same table, against the documented 'identical on every curve'."
            ),
        ),
    ),
    frame=frames(weights="dyadic", scores="any"),
    interpolation=st.sampled_from(INTERPOLATIONS),
    class_id=st.sampled_from([None, *CLASSES]),
)
@example(frame=GRID_EDGE, interpolation="11_point", class_id=None)
def test_average_precision_function_matches_reference(
    frame: Frame, interpolation: str, class_id: str | None
) -> None:
    class_id = _known_class(frame, class_id)
    got = average_precision(
        build(frame), class_id=class_id, interpolation=interpolation
    )
    want = _reference_ap(frame, interpolation, class_id)
    assert close(got, want), (got, want)


@property_lanes(
    weight=1,
    frame=frames(weights="dyadic", scores="mixed"),
    threshold=st.sampled_from([0.0, 0.2, 0.5, 0.8, 0.9, 1.1]),
    class_id=st.sampled_from([None, *CLASSES]),
)
def test_threshold_metrics_match_reference(
    frame: Frame, threshold: float, class_id: str | None
) -> None:
    if class_id is not None and class_id not in frame.meta["class_id"].to_list():
        class_id = None
    table = build(frame)
    dets, meta = frame.dets, frame.meta
    if class_id is not None:
        dets = dets.filter(pl.col("class_id") == class_id)
        meta = meta.filter(pl.col("class_id") == class_id)
    w = frame.image_weights()
    gt_mass = sum(n * w[i] for i, n in zip(meta["image_id"], meta["n_gts"]))
    kept = [
        (tp, w[i])
        for i, s, tp in zip(dets["image_id"], dets["score"], dets["is_tp"])
        if s >= threshold
    ]
    wtp = sum(x for tp, x in kept if tp)
    wfp = sum(x for tp, x in kept if not tp)
    precision = 1.0 if wtp + wfp == 0 else wtp / (wtp + wfp)
    recall = wtp / gt_mass if gt_mass > 0 else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    got = (
        precision_at_threshold(table, threshold, class_id=class_id),
        recall_at_threshold(table, threshold, class_id=class_id),
        f1_at_threshold(table, threshold, class_id=class_id),
    )
    for g, want in zip(got, (precision, recall, f1)):
        assert close(g, want), (got, (precision, recall, f1))


# ---------------------------------------------------------------------------
# Oracle: FROC / LROC AUC (tests/_metric_refs.py)
# ---------------------------------------------------------------------------


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic", single_class=True, need_gt=True, scores="any"),
    lo=st.sampled_from([0.0, 0.25, 0.5, 1.0]),
    width=st.sampled_from([0.25, 0.5, 1.0, 2.0, 4.0]),
    correction=st.sampled_from(["normalize", None]),
    extrapolate=st.sampled_from(["flat", "none"]),
)
def test_froc_trapezoidal_matches_reference(
    frame: Frame, lo: float, width: float, correction: str | None, extrapolate: str
) -> None:
    table = build(frame)
    fp_range = (lo, lo + width)
    got = scalar(
        froc_auc(
            table, fp_range=fp_range, correction=correction, extrapolate=extrapolate
        )
    )
    want = ref_froc_auc(
        table, fp_range=fp_range, correction=correction, extrapolate=extrapolate
    )
    assert close(got, want), (fp_range, correction, extrapolate, got, want)


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic", scores="any"),
)
def test_froc_mann_whitney_matches_reference(frame: Frame) -> None:
    table = build(frame)
    got = scalar_or_empty(froc_auc(table, method="mann_whitney"), 0.5)
    want = ref_froc_mw_detection(table)
    assert close(got, want), (got, want)


@property_lanes(
    weight=2,
    frame=frames(
        weights="dyadic",
        single_class=True,
        both_labels=True,
        scores="any",
        sort_by_score=True,
    ),
    variant=st.sampled_from(["best_tp", "top_scoring"]),
    lo=st.sampled_from([0.0, 0.25, 0.5]),
    width=st.sampled_from([0.25, 0.5, 1.0]),
    correction=st.sampled_from(["normalize", None]),
)
def test_lroc_trapezoidal_matches_reference(
    frame: Frame, variant: str, lo: float, width: float, correction: str | None
) -> None:
    table = build(frame)
    hi = min(lo + width, 1.0)
    if hi <= lo:
        lo, hi = 0.0, 1.0
    # the default window is the full [0, 1] FPF domain
    got = scalar(lroc_auc(table, variant=variant))
    assert close(got, ref_lroc_auc(table, variant=variant)), (variant, got)
    got = scalar(
        lroc_auc(
            table,
            variant=variant,
            fpf_range=(lo, hi),
            correction=correction,
            extrapolate="flat",
        )
    )
    want = ref_lroc_auc(
        table, variant=variant, fpf_range=(lo, hi), correction=correction
    )
    assert close(got, want), (variant, (lo, hi), correction, got, want)


@property_lanes(
    weight=2,
    frame=frames(
        weights="dyadic",
        single_class=True,
        both_labels=True,
        scores="any",
        sort_by_score=True,
    ),
    variant=st.sampled_from(["best_tp", "top_scoring"]),
)
def test_lroc_mann_whitney_matches_reference(frame: Frame, variant: str) -> None:
    table = build(frame)
    got = scalar_or_empty(
        lroc_auc(table, variant=variant, method="mann_whitney", level="image"), 0.5
    )
    assert close(got, ref_lroc_image_mw(table, variant)), (variant, got)
    got = scalar_or_empty(
        lroc_auc(table, variant=variant, method="mann_whitney", level="detection"),
        0.5,
    )
    # the detection-level LROC rank statistic is P(TP > FP) over detections
    assert close(got, ref_froc_mw_detection(table)), (variant, got)


@property_lanes(
    weight=2,
    frame=frames(weights="none", scores="any"),
    stat=stat_names,
    scale=st.sampled_from([1.0, 0.25, 0.5, 2.0, 4.0]),
)
def test_unit_weights_equal_unweighted_and_scale_invariant(
    frame: Frame, stat: str, scale: float
) -> None:
    """No weight column == explicit 1.0 == any constant weight (a ratio)."""
    statistic = STATS[stat]
    absent = frame
    explicit = Frame(
        dets=frame.dets,
        meta=frame.meta.with_columns(pl.lit(1.0).alias("weight")),
    )
    scaled = Frame(
        dets=frame.dets,
        meta=frame.meta.with_columns(pl.lit(scale).alias("weight")),
    )
    base = collect_sorted(statistic, build(absent), "class_id")
    assert_same_frames(
        base, collect_sorted(statistic, build(explicit), "class_id"), stat
    )
    got = collect_sorted(statistic, build(scaled), "class_id")
    assert got.columns == base.columns
    assert got["class_id"].to_list() == base["class_id"].to_list()
    name = statistic.name
    for g, b in zip(got[name].to_list(), base[name].to_list()):
        assert close(g, b), (stat, scale, g, b)


# ---------------------------------------------------------------------------
# Invariance
# ---------------------------------------------------------------------------


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic"),
    stat=stat_names,
    with_det_idx=st.booleans(),
    data=st.data(),
)
def test_row_order_does_not_matter(
    frame: Frame, stat: str, with_det_idx: bool, data: st.DataObject
) -> None:
    """Shuffling the detection rows (and the metadata rows) changes nothing.

    ``top_scoring`` LROC reads the first of several tied top detections, which
    the adapter's ordinal (``det_idx``) orders. Row order is then the ordinal's
    only source, so it is held to the shuffle only when the ordinal rides with
    each row (``det_idx_col``).
    """
    if stat == "lroc_top" and not with_det_idx:
        return
    statistic = STATS[stat]
    perm = data.draw(st.permutations(range(frame.dets.height)), label="detections")
    mperm = data.draw(st.permutations(range(frame.meta.height)), label="metadata")
    shuffled = Frame(
        dets=frame.dets[list(perm)] if perm else frame.dets,
        meta=frame.meta[list(mperm)] if mperm else frame.meta,
    )
    for keys in (None, "class_id"):
        want = collect_sorted(statistic, build(frame, det_idx=with_det_idx), keys)
        got = collect_sorted(statistic, build(shuffled, det_idx=with_det_idx), keys)
        assert_same_frames(want, got, f"{stat} by {keys}")


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic"),
    stat=stat_names,
    det_parts=st.integers(2, 5),
    meta_parts=st.integers(2, 4),
)
def test_chunking_does_not_matter(
    frame: Frame, stat: str, det_parts: int, meta_parts: int
) -> None:
    """The same rows in several chunks (``pl.concat(..., rechunk=False)``)."""
    statistic = STATS[stat]
    dets = _cut(frame.dets, det_parts)
    meta = _cut(frame.meta, meta_parts)
    for keys in (None, "class_id"):
        want = collect_sorted(statistic, build(frame, det_idx=True), keys)
        got = collect_sorted(
            statistic, build(frame, det_idx=True, dets=dets, meta=meta), keys
        )
        assert_same_frames(want, got, f"{stat} by {keys}")
    # a rechunked frame is the same frame again
    again = build(frame, det_idx=True, dets=dets.rechunk(), meta=meta.rechunk())
    assert_same_frames(
        collect_sorted(statistic, build(frame, det_idx=True)),
        collect_sorted(statistic, again),
        f"{stat} rechunked",
    )


#: AP on an N-point recall grid steps when a recall lands on a grid value, so a
#: 1-ulp difference in recall is not a 1-ulp difference in AP. They get their
#: own engine property (below), where that difference is the question.
GRID_AP = ("ap11", "ap101")


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic"),
    stat=st.sampled_from([k for k in sorted(STATS) if k not in GRID_AP]),
    keys=st.sampled_from([None, "class_id", "group_id"]),
)
def test_streaming_and_in_memory_engines_agree(
    frame: Frame, stat: str, keys: str | None
) -> None:
    """The two engines agree to 1e-12 relative.

    Not bit for bit: the in-memory engine divides by a broadcast scalar as a
    multiplication by its reciprocal and the streaming engine does not, so the
    last bit of a recall (and of a sum of products of recalls) differs. Only
    the Int128 sums are promised bit-exact, and they sum what the engines
    computed. Anything coarser than that is a real divergence.
    """
    statistic = STATS[stat]
    table = build(frame, det_idx=True)
    want = collect_sorted(statistic, table, keys, engine="in-memory")
    got = collect_sorted(statistic, table, keys, engine="streaming")
    assert got.columns == want.columns and got.height == want.height
    for col in want.columns[:-1]:
        assert got[col].to_list() == want[col].to_list()
    name = statistic.name
    for g, w in zip(got[name].to_list(), want[name].to_list()):
        assert close(g, w), (stat, keys, g, w)


@property_lanes(
    weight=1,
    marks=(
        pytest.mark.xfail(
            strict=True,
            reason=(
                "LIBRARY BUG: AP('11_point') of one table depends on the "
                "engine. The in-memory engine computes recall = cum_wtp / "
                "gt_mass as a multiplication by the reciprocal (3 * 0.2 = "
                "0.6000000000000001) where streaming divides (3 / 5 = 0.6), "
                "and a recall of 0.6 does not reach the grid value "
                "6 * 0.1 = 0.6000000000000001. GRID_EDGE: in-memory 3.5/11, "
                "streaming 3/11 (the numpy / COCO answer)."
            ),
        ),
    ),
    frame=frames(weights="dyadic", scores="any"),
    stat=st.sampled_from(GRID_AP),
    keys=st.sampled_from([None, "class_id"]),
)
@example(frame=GRID_EDGE, stat="ap11", keys=None)
def test_grid_ap_is_engine_independent(
    frame: Frame, stat: str, keys: str | None
) -> None:
    statistic = STATS[stat]
    table = build(frame, det_idx=True)
    want = collect_sorted(statistic, table, keys, engine="in-memory")
    got = collect_sorted(statistic, table, keys, engine="streaming")
    assert got.height == want.height
    for g, w in zip(got["ap"].to_list(), want["ap"].to_list()):
        assert close(g, w), (stat, keys, g, w)


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic", max_images=7),
    stat=stat_names,
    key=st.sampled_from(["class_id", "group_id"]),
)
def test_group_by_equals_the_group_filtered_alone(
    frame: Frame, stat: str, key: str
) -> None:
    """``by_group(key)`` for a group == the statistic on that group's rows alone."""
    if stat == "froc_mw_image" and key == "class_id":
        return  # held by test_froc_image_level_mann_whitney_by_class_...
    statistic = STATS[stat]
    grouped = collect_sorted(statistic, build(frame, det_idx=True), key)
    name = statistic.name
    group_of_image = dict(zip(frame.meta["image_id"], frame.meta["group_id"]))
    for value in sorted(set(frame.meta[key].to_list())):
        if key == "class_id":
            dets = frame.dets.filter(pl.col("class_id") == value)
            meta = frame.meta.filter(pl.col("class_id") == value)
        else:
            ids = [i for i, g in group_of_image.items() if g == value]
            dets = frame.dets.filter(pl.col("image_id").is_in(ids))
            meta = frame.meta.filter(pl.col("image_id").is_in(ids))
        alone = collect_sorted(
            statistic, build(frame, det_idx=True, dets=dets, meta=meta)
        )
        row = grouped.filter(pl.col(key) == value)
        # present in one reading exactly when present in the other ...
        assert row.height == alone.height, (stat, key, value, grouped, alone)
        if alone.height:  # ... and equal
            assert same(row[name].item(), alone[name].item()), (
                stat,
                key,
                value,
                row[name].item(),
                alone[name].item(),
            )


#: Image i0 is positive for class a and has nothing; i1's only detection is a
#: class-b false positive. Class a, read alone, has no detections at all.
CROSS_CLASS = Frame(
    dets=pl.DataFrame(
        {
            "image_id": ["i1"],
            "class_id": ["b"],
            "score": [0.2],
            "is_tp": [False],
            "det_idx": pl.Series([0], dtype=pl.UInt32),
        }
    ),
    meta=pl.DataFrame(
        {
            "image_id": ["i0", "i1", "i0", "i1"],
            "class_id": ["a", "a", "b", "b"],
            "n_gts": [1, 0, 0, 0],
            "weight": [1.0] * 4,
            "group_id": ["g0"] * 4,
        }
    ),
)


def _class_alone(frame: Frame, class_id: str) -> DetectionTable:
    return build(
        frame,
        det_idx=True,
        dets=frame.dets.filter(pl.col("class_id") == class_id),
        meta=frame.meta.filter(pl.col("class_id") == class_id),
    )


@property_lanes(weight=1, frame=frames(weights="dyadic", max_images=7))
@example(frame=CROSS_CLASS)
def test_froc_image_level_mann_whitney_by_class_ignores_other_classes(
    frame: Frame,
) -> None:
    """Each class's image-level FROC Mann-Whitney AUC is the one its rows give
    alone: an image is scored from its own class's detections."""
    statistic = STATS["froc_mw_image"]
    grouped = collect_sorted(statistic, build(frame, det_idx=True), "class_id")
    for class_id in sorted(set(frame.meta["class_id"].to_list())):
        alone = collect_sorted(statistic, _class_alone(frame, class_id))
        row = grouped.filter(pl.col("class_id") == class_id)
        assert row.height == alone.height
        if alone.height:
            assert same(row["auc"].item(), alone["auc"].item()), class_id


@property_lanes(
    weight=2,
    frame=frames(weights="dyadic", scores="grid"),
    stat=st.sampled_from(RANK_STATS),
    transform=st.sampled_from(["affine", "exp", "square"]),
    slope=st.sampled_from([0.5, 2.0, 4.0, 8.0]),
    offset=st.sampled_from([0.0, 1.0, 7.0]),
)
def test_increasing_score_transform_leaves_rank_metrics_unchanged(
    frame: Frame, stat: str, transform: str, slope: float, offset: float
) -> None:
    """AP, FROC / LROC AUC and Mann-Whitney read only the ranking of the scores.

    Scores are on the 1/1024 grid and the affine maps are exact, so the map is
    strictly increasing in floating point as well as in the reals (an ``exp``
    of two near floats could otherwise merge them into a tie). Offsets are
    non-negative: the image-level rank statistics score an image with nothing
    to commit as 0.0, which presumes scores above 0 (see the negative-score
    property below).
    """
    statistic = STATS[stat]
    expr = {
        "affine": pl.col("score") * slope + offset,
        "exp": pl.col("score").exp(),
        "square": pl.col("score") ** 2,
    }[transform]
    moved = frame.dets.with_columns(expr.alias("score"))
    assert moved["score"].n_unique() == frame.dets["score"].n_unique()
    for keys in (None, "class_id"):
        want = collect_sorted(statistic, build(frame, det_idx=True), keys)
        got = collect_sorted(statistic, build(frame, det_idx=True, dets=moved), keys)
        assert_same_frames(want, got, f"{stat} under {transform}")


#: A positive image with no detection and a negative one whose only detection
#: scores 0.001: shifted by -3 the detection sits below the 0.0 a missed image
#: is given.
NEGATIVE_SCORES = Frame(
    dets=pl.DataFrame(
        {
            "image_id": ["n"],
            "class_id": ["a"],
            "score": [0.001],
            "is_tp": [False],
            "det_idx": pl.Series([0], dtype=pl.UInt32),
        }
    ),
    meta=pl.DataFrame(
        {
            "image_id": ["p", "n"],
            "class_id": ["a", "a"],
            "n_gts": [1, 0],
            "weight": [1.0, 1.0],
            "group_id": ["g0", "g0"],
        }
    ),
)


@property_lanes(
    weight=1,
    marks=(
        pytest.mark.xfail(
            strict=True,
            reason=(
                "LIBRARY BUG (silent wrong answer outside an implicit domain): "
                "the image-level Mann-Whitney AUCs give an image with no "
                "detection the score 0.0 (`fill_null(0.0)`), so they are only "
                "rank-based while every score is above 0. Shifting the scores "
                "of the same ranking below 0 (a logit, say) changes the AUC "
                "from 0.0 to 1.0 instead of being refused."
            ),
        ),
    ),
    frame=frames(weights="dyadic", scores="grid"),
    stat=st.sampled_from(["froc_mw_image", "lroc_mw_image"]),
    shift=st.sampled_from([-0.5, -1.0, -3.0]),
)
@example(frame=NEGATIVE_SCORES, stat="froc_mw_image", shift=-3.0)
def test_image_level_rank_statistics_survive_negative_scores(
    frame: Frame, stat: str, shift: float
) -> None:
    statistic = STATS[stat]
    moved = frame.dets.with_columns((pl.col("score") + shift).alias("score"))
    want = collect_sorted(statistic, build(frame, det_idx=True))
    got = collect_sorted(statistic, build(frame, det_idx=True, dets=moved))
    assert_same_frames(want, got, f"{stat} shifted by {shift}")


@property_lanes(
    weight=2,
    frame=frames(weights="int"),
    stat=stat_names,
    keys=st.sampled_from([None, "class_id"]),
)
def test_integer_weight_equals_duplicated_images(
    frame: Frame, stat: str, keys: str | None
) -> None:
    """An image of integer weight ``w`` == ``w`` copies of it under fresh ids.

    The documented ``sample_weight`` semantics (``metrics/AGENTS.md``): every
    weighted statistic is a ratio of weighted masses, so a weight of ``w`` and
    ``w`` replicas carry the same mass.
    """
    statistic = STATS[stat]
    weight = frame.image_weights()

    def copies(df: pl.DataFrame, *, weighted: bool) -> pl.DataFrame:
        parts = []
        for image_id, w in weight.items():
            rows = df.filter(pl.col("image_id") == image_id)
            for j in range(int(w)):
                parts.append(
                    rows.with_columns((pl.col("image_id") + f"#r{j}").alias("image_id"))
                )
        out = pl.concat(parts) if parts else df.clear()
        return out.with_columns(pl.lit(1.0).alias("weight")) if weighted else out

    dup = Frame(
        dets=copies(frame.dets, weighted=False),
        meta=copies(frame.meta, weighted=True),
    )
    want = collect_sorted(statistic, build(dup, det_idx=True), keys)
    got = collect_sorted(statistic, build(frame, det_idx=True), keys)
    name = statistic.name
    assert got.columns == want.columns and got.height == want.height, (got, want)
    for key_col in want.columns[:-1]:
        assert got[key_col].to_list() == want[key_col].to_list()
    for g, w in zip(got[name].to_list(), want[name].to_list()):
        assert close(g, w), (stat, g, w)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

#: Each builder is one public lazy CI entry point (``[*keys, <metric>, lo, hi]``).
CI_BUILDERS = {
    "ap": lambda t, **kw: average_precision_ci_lazy(t, **kw),
    "froc": lambda t, **kw: froc_auc_ci_lazy(
        t, fp_range=(0.0, 2.0), extrapolate="flat", **kw
    ),
    "froc_mw": lambda t, **kw: froc_auc_ci_lazy(t, method="mann_whitney", **kw),
    "lroc": lambda t, **kw: lroc_auc_ci_lazy(t, **kw),
    "lroc_mw_image": lambda t, **kw: lroc_auc_ci_lazy(
        t, method="mann_whitney", level="image", **kw
    ),
    "bootstrap_ci_ap101": lambda t, **kw: bootstrap_ci(t, AP("101_point"), **kw),
}


def _ci_frames(
    make: Any,
    frame: Frame,
    group_by: str | None,
    seed: int,
    n_bootstrap: int,
) -> tuple[DetectionTable, Any]:
    table = build(frame, det_idx=True)

    def run(t: DetectionTable = table, engine: str | None = None) -> pl.DataFrame:
        lf = make(t, group_by=group_by, seed=seed, n_bootstrap=n_bootstrap)
        out = lf.collect() if engine is None else lf.collect(engine=engine)
        return out.sort(group_by) if group_by else out

    return table, run


def _assert_interval_ordered(ci: pl.DataFrame) -> None:
    """lower <= upper wherever defined (``lower <= point <= upper`` is *not*
    guaranteed, so it is not asserted); a degenerate group nulls both bounds."""
    for lo, hi in zip(ci["ci_lower"].to_list(), ci["ci_upper"].to_list()):
        if not _missing(lo) and not _missing(hi):
            assert lo <= hi, (lo, hi)
        else:
            assert _missing(lo) and _missing(hi), (lo, hi)


@property_lanes(
    weight=0.5,
    frame=frames(weights="unit", max_images=6),
    which=st.sampled_from(sorted(CI_BUILDERS)),
    group_by=st.sampled_from([None, "class_id"]),
    seed=st.integers(0, 2**32 - 1),
    n_bootstrap=st.integers(1, 12),
    det_parts=st.integers(2, 4),
)
def test_bootstrap_same_seed_is_bit_identical_across_runs_and_chunks(
    frame: Frame,
    which: str,
    group_by: str | None,
    seed: int,
    n_bootstrap: int,
    det_parts: int,
) -> None:
    make = CI_BUILDERS[which]
    _, run = _ci_frames(make, frame, group_by, seed, n_bootstrap)
    first = run()
    assert_same_frames(first, run(), "second run")
    chunked = build(
        frame,
        det_idx=True,
        dets=_cut(frame.dets, det_parts),
        meta=_cut(frame.meta, det_parts),
    )
    assert_same_frames(first, run(chunked), "chunked input")
    _assert_interval_ordered(first)


#: Bootstrap bounds that are bit-identical across engines (rank statistics).
ENGINE_STABLE_CIS = ("froc_mw", "lroc_mw_image")
#: ... and the ones that are not (see the xfail below).
ENGINE_DIVERGENT_CIS = ("ap", "froc", "lroc", "bootstrap_ci_ap101")

#: Six unit-weight images, one of two GTs and one of one, both found; the AP
#: and FROC bounds of seed 7 differ in the last bit between the engines.
BOOT_EDGE = Frame(
    dets=pl.DataFrame(
        {
            "image_id": ["img1", "img5"],
            "class_id": ["a", "a"],
            "score": [0.2, 0.5],
            "is_tp": [True, True],
            "det_idx": pl.Series([0, 0], dtype=pl.UInt32),
        }
    ),
    meta=pl.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(6)],
            "class_id": ["a"] * 6,
            "n_gts": [0, 2, 0, 0, 1, 1],
            "weight": [1.0] * 6,
            "group_id": ["g0"] * 6,
        }
    ),
)


def _engines_agree(which: str, frame: Frame, group_by, seed, n_bootstrap) -> None:
    _, run = _ci_frames(CI_BUILDERS[which], frame, group_by, seed, n_bootstrap)
    assert_same_frames(
        run(engine="in-memory"), run(engine="streaming"), f"{which} engines"
    )


@property_lanes(
    weight=0.5,
    frame=frames(weights="unit", max_images=6),
    which=st.sampled_from(ENGINE_STABLE_CIS),
    group_by=st.sampled_from([None, "class_id"]),
    seed=st.integers(0, 2**32 - 1),
    n_bootstrap=st.integers(1, 12),
)
def test_bootstrap_rank_statistic_bounds_are_engine_independent(
    frame: Frame, which: str, group_by: str | None, seed: int, n_bootstrap: int
) -> None:
    _engines_agree(which, frame, group_by, seed, n_bootstrap)


@property_lanes(
    weight=0.5,
    marks=(
        pytest.mark.xfail(
            strict=True,
            reason=(
                "LIBRARY BUG: the AP and trapezoidal FROC / LROC bootstrap bounds are "
                "not bit-identical across engines: the in-memory engine's "
                "scalar division (a multiplication by the reciprocal) gives "
                "0.6000000000000001 where streaming gives 0.6 for the same "
                "replicate. BOOT_EDGE, seed 7, n_bootstrap 1."
            ),
        ),
    ),
    frame=frames(weights="unit", max_images=6),
    which=st.sampled_from(ENGINE_DIVERGENT_CIS),
    group_by=st.sampled_from([None, "class_id"]),
    seed=st.integers(0, 2**32 - 1),
    n_bootstrap=st.integers(1, 12),
)
@example(frame=BOOT_EDGE, which="ap", group_by=None, seed=7, n_bootstrap=1)
def test_bootstrap_pr_and_froc_bounds_are_engine_independent(
    frame: Frame, which: str, group_by: str | None, seed: int, n_bootstrap: int
) -> None:
    _engines_agree(which, frame, group_by, seed, n_bootstrap)
