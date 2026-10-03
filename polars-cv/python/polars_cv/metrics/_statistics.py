"""Statistics: every detection metric as one grouped, lazy reduction.

A :class:`Statistic` turns a :class:`DetectionTable` into a ``LazyFrame`` with
one row per group, ``[*keys, <name>]``. That one shape is what lets every
metric be computed per class, per IoU threshold, per subgroup or per bootstrap
replicate by the same code: the scalar functions (``average_precision``,
``froc_sensitivity_at_fp``, …) are its ungrouped reading, a report's tables are
its grouped readings, and :func:`~polars_cv.metrics.bootstrap_ci` is its
reading per replicate. A statistic computes nothing itself beyond composing
the existing authorities (the PR, FROC and LROC modules); it adds no second
estimator.

:class:`MeanOver` averages a statistic over *facets* — keys such as
``class_id`` and ``iou_threshold`` that partition the evaluation of one image
population. Facets are not partition keys of the bootstrap: a replicate draws
images and brings each image's rows for every facet, so the facets of one
replicate are paired, which is what a confidence interval for mAP needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import polars as pl

from ._auc_expr import ordered_mean
from ._types import COL_CLASS_ID, COL_IMAGE_ID, COL_IS_TP, COL_SCORE, COL_WEIGHT
from ._weights import WeightAgg, attach_resolved_weight, weighted_gt_mass

if TYPE_CHECKING:
    from ._auc import CorrectionMethod, Extrapolate
    from ._metrics._precision_recall import APInterpolation
    from ._types import DetectionTable

# A dummy key for the authorities that need a non-empty grouping; dropped from
# every result.
_ONE = "_stat_one"

#: The standard radiology FROC operating points (false positives per image).
FROC_RATES: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def normalize_keys(keys: str | Sequence[str] | None) -> list[str]:
    """``None`` / one name / several names → a list of key columns."""
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    return list(keys)


def _frames(
    table: DetectionTable, keys: list[str]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """The table's ``(detections, image_metadata)`` for a grouping by ``keys``.

    The one place a statistic reads the frames, so a property of the table
    that constrains which groupings are meaningful is enforced for every
    statistic at once.
    """
    return table.frames(keys)


def _with_meta_keys(
    det: pl.LazyFrame, meta: pl.LazyFrame, keys: list[str]
) -> pl.LazyFrame:
    """Attach the grouping keys that live only on the metadata (``group_id``)."""
    names = set(det.collect_schema().names())
    meta_only = [k for k in keys if k not in names]
    if not meta_only:
        return det
    return det.join(
        meta.select(COL_IMAGE_ID, *meta_only).unique(), on=COL_IMAGE_ID, how="left"
    )


class Statistic(ABC):
    """A metric as a grouped lazy reduction of a :class:`DetectionTable`.

    Subclasses implement :meth:`_by_group`; callers use :meth:`by_group` (any
    grouping, lazily) or :meth:`value` (the ungrouped scalar).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The output column, and the metric's name in reports."""

    @property
    def empty_value(self) -> float:
        """The value of a group with no detections (a bootstrap draw of none)."""
        return 0.0

    @property
    def require_both_classes(self) -> bool:
        """Whether a group needs negative images too (two-class rank statistics)."""
        return False

    def by_group(
        self,
        table: DetectionTable,
        keys: str | Sequence[str] | None = None,
        *,
        weight_agg: WeightAgg = "first",
    ) -> pl.LazyFrame:
        """The statistic per group, as a lazy ``[*keys, name]`` frame.

        Args:
            table: The detection table.
            keys: Grouping column(s) on the detections or the metadata.
                ``None`` gives one row.
            weight_agg: Duplicate-weight resolution policy.
        """
        return self._by_group(table, normalize_keys(keys), weight_agg)

    @abstractmethod
    def _by_group(
        self, table: DetectionTable, keys: list[str], weight_agg: WeightAgg
    ) -> pl.LazyFrame: ...

    def support(
        self,
        table: DetectionTable,
        keys: list[str],
        *,
        weight_agg: WeightAgg = "first",
    ) -> pl.LazyFrame:
        """The groups where the statistic is defined, as ``[*keys]``.

        Detection statistics are defined where there is ground truth to find:
        a group with no weighted ground-truth mass has no recall (COCO scores
        such a class ``-1`` and leaves it out of the mean).
        """
        _, meta = _frames(table, keys)
        mass = weighted_gt_mass(meta, keys, weight_agg)
        if not keys:
            return mass.filter(pl.col("gt_mass") > 0.0).select(pl.lit(0).alias(_ONE))
        return mass.filter(pl.col("gt_mass") > 0.0).select(keys)

    def value(self, table: DetectionTable, *, weight_agg: WeightAgg = "first") -> float:
        """The ungrouped statistic as a Python float (one collect).

        A table with no row for the statistic (no detections) gives
        :attr:`empty_value`; a null (an operating point the curve does not
        reach) is returned as ``nan``.
        """
        out = self.by_group(table, weight_agg=weight_agg).collect(engine="streaming")
        if out.height == 0:
            return self.empty_value
        v = out[self.name].item()
        return float("nan") if v is None else float(v)


# ---------------------------------------------------------------------------
# Precision-recall family
# ---------------------------------------------------------------------------


def _expanded(
    table: DetectionTable, keys: list[str], weight_agg: WeightAgg
) -> tuple[pl.LazyFrame, list[str], bool]:
    """Per-detection ``[*keys, score, is_tp, weight, gt_mass]``.

    Returns the frame, the keys it is grouped by (a dummy one when ``keys`` is
    empty) and whether that dummy must be dropped.
    """
    det, meta = _frames(table, keys)
    det = _with_meta_keys(
        attach_resolved_weight(det, meta, weight_agg=weight_agg), meta, keys
    )
    columns = (COL_SCORE, COL_IS_TP, COL_WEIGHT)
    if keys:
        expanded = (
            det.select(*keys, *columns)
            .drop_nulls(COL_SCORE)
            .join(
                weighted_gt_mass(meta, keys, weight_agg),
                on=keys,
                how="left",
                nulls_equal=True,
            )
        )
        return expanded, keys, False
    expanded = (
        det.select(*columns)
        .drop_nulls(COL_SCORE)
        .join(weighted_gt_mass(meta, [], weight_agg), how="cross")
        .with_columns(pl.lit(0, dtype=pl.Int32).alias(_ONE))
    )
    return expanded, [_ONE], True


@dataclass(frozen=True)
class AP(Statistic):
    """Weighted average precision.

    Args:
        interpolation: ``"all_points"`` (the envelope integrated as a step
            function, Pascal VOC 2010+), ``"11_point"`` (VOC 2007) or
            ``"101_point"`` (COCO).
    """

    interpolation: APInterpolation = "all_points"

    @property
    def name(self) -> str:
        return "ap"

    def _by_group(self, table, keys, weight_agg):
        from ._metrics._precision_recall import ap_by_group, validate_interpolation

        validate_interpolation(self.interpolation)
        expanded, gkeys, drop = _expanded(table, keys, weight_agg)
        out = ap_by_group(expanded, group_col=gkeys, interpolation=self.interpolation)
        return out.drop(_ONE) if drop else out


@dataclass(frozen=True)
class Recall(Statistic):
    """Weighted recall over every detection: the highest recall reached.

    Averaged over classes and IoU thresholds, this is COCO's average recall
    (AR) at the detections kept (``max_detections``).
    """

    @property
    def name(self) -> str:
        return "recall"

    def _by_group(self, table, keys, weight_agg):
        expanded, gkeys, drop = _expanded(table, keys, weight_agg)
        out = (
            expanded.group_by(gkeys)
            .agg(
                _wtp=(pl.col(COL_IS_TP).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum(),
                _mass=pl.col("gt_mass").first(),
            )
            .filter(pl.col("_mass") > 0.0)
            .select(*gkeys, recall=pl.col("_wtp") / pl.col("_mass"))
        )
        return out.drop(_ONE) if drop else out


@dataclass(frozen=True)
class _AtScore(Statistic):
    """Weighted confusion counts of the detections at or above a score."""

    score_threshold: float

    def _counts(self, table, keys, weight_agg) -> tuple[pl.LazyFrame, list[str], bool]:
        det, meta = _frames(table, keys)
        gkeys, drop = (keys, False) if keys else ([_ONE], True)
        one = pl.lit(0, dtype=pl.Int32).alias(_ONE)
        det = _with_meta_keys(
            attach_resolved_weight(det, meta, weight_agg=weight_agg), meta, keys
        ).filter(pl.col(COL_SCORE) >= self.score_threshold)
        tp, w = pl.col(COL_IS_TP), pl.col(COL_WEIGHT)
        counts = (
            det.with_columns(one)
            .group_by(gkeys)
            .agg(
                _wtp=(tp.cast(pl.Float64) * w).sum(),
                _wfp=((~tp).cast(pl.Float64) * w).sum(),
            )
        )
        mass = weighted_gt_mass(meta, keys, weight_agg)
        groups = mass.with_columns(one) if drop else mass
        # Every group of the metadata, so a group with nothing predicted is
        # there (precision 1, recall 0) rather than absent.
        joined = groups.join(
            counts, on=gkeys, how="left", nulls_equal=True
        ).with_columns(pl.col("_wtp").fill_null(0.0), pl.col("_wfp").fill_null(0.0))
        return joined, gkeys, drop

    @staticmethod
    def _precision() -> pl.Expr:
        predicted = pl.col("_wtp") + pl.col("_wfp")
        return pl.when(predicted == 0.0).then(1.0).otherwise(pl.col("_wtp") / predicted)

    @staticmethod
    def _recall() -> pl.Expr:
        return (
            pl.when(pl.col("gt_mass") > 0.0)
            .then(pl.col("_wtp") / pl.col("gt_mass"))
            .otherwise(0.0)
        )

    def _select(self, table, keys, weight_agg, expr: pl.Expr) -> pl.LazyFrame:
        joined, gkeys, drop = self._counts(table, keys, weight_agg)
        out = joined.select(*gkeys, expr.alias(self.name))
        return out.drop(_ONE) if drop else out


@dataclass(frozen=True)
class PrecisionAt(_AtScore):
    """Weighted precision of the detections scoring at least ``score_threshold``
    (``1.0`` when none do)."""

    @property
    def name(self) -> str:
        return "precision"

    def _by_group(self, table, keys, weight_agg):
        return self._select(table, keys, weight_agg, self._precision())


@dataclass(frozen=True)
class RecallAt(_AtScore):
    """Weighted recall of the detections scoring at least ``score_threshold``."""

    @property
    def name(self) -> str:
        return "recall"

    def _by_group(self, table, keys, weight_agg):
        return self._select(table, keys, weight_agg, self._recall())


@dataclass(frozen=True)
class F1At(_AtScore):
    """Weighted F1 of the detections scoring at least ``score_threshold``."""

    @property
    def name(self) -> str:
        return "f1"

    def _by_group(self, table, keys, weight_agg):
        p, r = self._precision(), self._recall()
        f1 = pl.when((p + r) == 0.0).then(0.0).otherwise(2.0 * p * r / (p + r))
        return self._select(table, keys, weight_agg, f1)


# ---------------------------------------------------------------------------
# FROC / LROC families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FROCSensitivity(Statistic):
    """FROC sensitivity at ``fp_per_image`` false positives per image.

    Null where the curve does not reach that rate and ``extrapolate="none"``.
    """

    fp_per_image: float
    extrapolate: Extrapolate = "none"

    @property
    def name(self) -> str:
        return "sensitivity"

    def _by_group(self, table, keys, weight_agg):
        return _froc_sensitivities(
            table, keys, weight_agg, [self.fp_per_image], self.extrapolate
        ).select(*keys, "sensitivity")


@dataclass(frozen=True)
class CPM(Statistic):
    """Competition performance metric: the mean FROC sensitivity over
    ``fp_rates`` (LUNA16 / ANODE09; by default the seven rates 1/8 … 8).

    Null where the curve does not reach a rate and ``extrapolate="none"``;
    LUNA16 reads the curve with ``extrapolate="flat"``.
    """

    fp_rates: tuple[float, ...] = FROC_RATES
    extrapolate: Extrapolate = "none"

    @property
    def name(self) -> str:
        return "cpm"

    def _by_group(self, table, keys, weight_agg):
        sens = _froc_sensitivities(
            table, keys, weight_agg, list(self.fp_rates), self.extrapolate
        )
        # A null sensitivity (off the curve) makes the mean null: an operating
        # point that was not observed is not averaged in as anything.
        mean = (
            pl.when(pl.col("sensitivity").is_null().any())
            .then(None)
            .otherwise(ordered_mean(pl.col("sensitivity")))
        )
        if keys:
            return sens.group_by(keys).agg(mean.alias("cpm"))
        return sens.select(mean.alias("cpm"))


def _froc_sensitivities(
    table: DetectionTable,
    keys: list[str],
    weight_agg: WeightAgg,
    rates: list[float],
    extrapolate: Extrapolate,
) -> pl.LazyFrame:
    from ._metrics._froc import froc_sensitivities_by_group

    return froc_sensitivities_by_group(
        table, rates, group_by=keys, weight_agg=weight_agg, extrapolate=extrapolate
    )


@dataclass(frozen=True)
class FROCAUC(Statistic):
    """FROC AUC (see :func:`~polars_cv.metrics.froc_auc` for the arguments)."""

    fp_range: tuple[float, float] | None = None
    method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal"
    correction: CorrectionMethod = "normalize"
    level: Literal["detection", "image"] = "detection"
    extrapolate: Extrapolate = "none"

    @property
    def name(self) -> str:
        return "auc"

    @property
    def empty_value(self) -> float:
        return 0.5 if self.method == "mann_whitney" else 0.0

    @property
    def require_both_classes(self) -> bool:
        return self.method == "mann_whitney"

    def _by_group(self, table, keys, weight_agg):
        from ._metrics._froc import froc_auc

        _frames(table, keys)
        return froc_auc(
            table,
            method=self.method,
            fp_range=self.fp_range,
            correction=self.correction,
            level=self.level,
            group_by=keys or None,
            weight_agg=weight_agg,
            extrapolate=self.extrapolate,
        )


@dataclass(frozen=True)
class LROCAUC(Statistic):
    """LROC AUC (see :func:`~polars_cv.metrics.lroc_auc` for the arguments)."""

    variant: Literal["best_tp", "top_scoring"] = "best_tp"
    method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal"
    fpf_range: tuple[float, float] | None = None
    correction: CorrectionMethod = "normalize"
    level: Literal["detection", "image"] = "image"
    extrapolate: Extrapolate = "none"

    @property
    def name(self) -> str:
        return "auc"

    @property
    def empty_value(self) -> float:
        return 0.5 if self.method == "mann_whitney" else 0.0

    @property
    def require_both_classes(self) -> bool:
        return self.method == "mann_whitney"

    def _by_group(self, table, keys, weight_agg):
        from ._metrics._lroc import lroc_auc

        _frames(table, keys)
        return lroc_auc(
            table,
            variant=self.variant,
            method=self.method,
            fpf_range=self.fpf_range,
            correction=self.correction,
            level=self.level,
            group_by=keys or None,
            weight_agg=weight_agg,
            extrapolate=self.extrapolate,
        )


@dataclass(frozen=True)
class LROCSensitivity(Statistic):
    """LROC sensitivity at a false-positive fraction ``fpf``."""

    fpf: float
    variant: Literal["best_tp", "top_scoring"] = "best_tp"
    extrapolate: Extrapolate = "none"

    @property
    def name(self) -> str:
        return "sensitivity"

    def _by_group(self, table, keys, weight_agg):
        from ._metrics._lroc import lroc_sensitivity_at_fpf

        return lroc_sensitivity_at_fpf(
            table,
            self.fpf,
            variant=self.variant,
            group_by=keys,
            weight_agg=weight_agg,
            extrapolate=self.extrapolate,
        ).select(*keys, "sensitivity")


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanOver(Statistic):
    """The mean of ``statistic`` over facets — mAP is ``MeanOver(AP())``.

    The inner statistic is computed per ``[*keys, *over]`` and averaged within
    ``keys``, over the facets where it is defined (:meth:`Statistic.support`):
    a facet with ground truth but no detections counts as the inner
    statistic's empty value (AP 0). A facet with no ground truth is handled by
    ``undefined``:

    * ``"exclude"`` leaves it out of the mean, as COCO does (an absent class
      is not a class the model failed on);
    * ``"zero"`` averages it in as ``0`` (what
      :func:`~polars_cv.metrics.mean_average_precision` has always done).

    Args:
        statistic: The statistic to average.
        over: The facet columns; ``class_id`` and ``iou_threshold`` by default
            (whichever the table has).
        undefined: The policy for facets without ground truth (required: the
            two conventions give different numbers).
        label: The output column; ``"mean_<inner name>"`` by default.
    """

    statistic: Statistic
    undefined: Literal["exclude", "zero"]
    over: tuple[str, ...] = (COL_CLASS_ID, "iou_threshold")
    label: str | None = field(default=None)

    @property
    def name(self) -> str:
        return self.label or f"mean_{self.statistic.name}"

    @property
    def empty_value(self) -> float:
        return self.statistic.empty_value

    def facets(self, table: DetectionTable) -> list[str]:
        """The facet columns this table carries."""
        names = set(table.meta_columns())
        return [c for c in self.over if c in names]

    def per_facet(
        self,
        table: DetectionTable,
        keys: str | Sequence[str] | None = None,
        *,
        weight_agg: WeightAgg = "first",
    ) -> pl.LazyFrame:
        """The inner statistic per ``[*keys, *facets]`` over the averaging grid.

        One row per facet the mean averages over: ``undefined="exclude"`` keeps
        only facets with ground truth; ``"zero"`` keeps every facet of the
        metadata, scoring those without ground truth ``0``.
        """
        keys = normalize_keys(keys)
        facets = self.facets(table)
        inner_keys = [*keys, *[f for f in facets if f not in keys]]
        _, meta = _frames(table, inner_keys)
        inner = self.statistic.by_group(table, inner_keys, weight_agg=weight_agg)
        defined = self.statistic.support(table, inner_keys, weight_agg=weight_agg)
        name = self.statistic.name
        if self.undefined == "exclude":
            grid = defined.with_columns(pl.lit(True).alias("_defined"))
        else:
            grid = (
                meta.select(inner_keys)
                .unique()
                .join(
                    defined.with_columns(pl.lit(True).alias("_defined")),
                    on=inner_keys,
                    how="left",
                    nulls_equal=True,
                )
            )
        return (
            grid.join(inner, on=inner_keys, how="left", nulls_equal=True)
            .with_columns(
                pl.when(pl.col("_defined").fill_null(False))
                .then(pl.col(name).fill_null(self.statistic.empty_value))
                .otherwise(0.0)
                .alias(name)
            )
            .select(*inner_keys, name)
            .sort(inner_keys, nulls_last=True)
        )

    def _by_group(self, table, keys, weight_agg):
        if not keys and not self.facets(table):
            # No facets: the mean of one value is the value.
            return self.statistic.by_group(table, weight_agg=weight_agg).rename(
                {self.statistic.name: self.name}
            )
        per = self.per_facet(table, keys, weight_agg=weight_agg)
        mean = ordered_mean(pl.col(self.statistic.name)).alias(self.name)
        if keys:
            return per.group_by(keys).agg(mean)
        return per.select(mean)


def mean_ap(
    interpolation: APInterpolation = "all_points",
    *,
    undefined: Literal["exclude", "zero"] = "exclude",
) -> MeanOver:
    """mAP: :class:`AP` averaged over classes and IoU thresholds."""
    return MeanOver(AP(interpolation), undefined=undefined, label="map")
