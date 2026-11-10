"""The detection report: a matched table and every number read from it.

A :class:`DetectionReport` holds the :class:`DetectionTable` and reads every
number it shows from the same :class:`Statistic` objects :func:`bootstrap_ci`
takes, so each reported number has a confidence interval through one engine
(:meth:`DetectionReport.ci`), and every lower-level metric remains one method
away. It does not match: ``_cv._evaluate`` (``evaluate_detections``,
``evaluate_heatmaps``) builds one from images and geometry, and a table from
any other matcher can be wrapped the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import polars as pl

from ._bootstrap import WeightScheme, bootstrap_ci
from ._inputs import GT_ROW, PRED_ROW
from ._statistics import (
    AP,
    CPM,
    FROCSensitivity,
    MeanOver,
    Recall,
    Statistic,
)
from ._types import COL_CLASS_ID, COL_IMAGE_ID, COL_N_GTS, DetectionTable
from ._weights import weighted_gt_mass

if TYPE_CHECKING:
    from ._auc import Extrapolate
    from ._metrics import ConfusionResult, PrecisionRecallResult
    from ._metrics._precision_recall import APInterpolation

#: COCO's IoU thresholds, 0.50:0.05:0.95.
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.5 + 0.05 * i, 2) for i in range(10)
)

_THR = "iou_threshold"


def _materialized(table: DetectionTable) -> DetectionTable:
    """The table with both frames collected once (one run of the matching):
    a report reads it many times — the summary, the tables, every interval."""
    det, meta = table.collect()
    return replace(table, _detections=det.lazy(), _image_meta=meta.lazy())


@dataclass(frozen=True)
class ReportMetric:
    """A reported number: the statistic and the table view it reads."""

    statistic: Statistic
    table: DetectionTable


@dataclass(frozen=True, repr=False)
class DetectionReport:
    """A detection evaluation: summary, per-class and per-threshold tables.

    Every number comes from a :class:`Statistic` on :attr:`table` (or one
    threshold's view of it); :meth:`ci` bootstraps the same statistics.

    Attributes:
        table: The matched :class:`DetectionTable` (a sweep when several IoU
            thresholds were matched).
        interpolation: The AP interpolation of every AP in the report.
        fp_rates: The FROC operating points summarised (sensitivity at each,
            and their mean, the CPM), or ``None`` for no FROC summary.
        extrapolate: How FROC curves are read past their last point.
    """

    table: DetectionTable
    interpolation: APInterpolation = "all_points"
    fp_rates: tuple[float, ...] | None = None
    extrapolate: Extrapolate = "none"
    _objects: pl.LazyFrame | None = field(default=None)
    _predictions: pl.LazyFrame | None = field(default=None)
    _ground_truth: pl.LazyFrame | None = field(default=None)

    # -- the thresholds -------------------------------------------------------

    @property
    def iou_thresholds(self) -> tuple[float, ...]:
        """The IoU thresholds matched (empty when unknown)."""
        return self.table.iou_thresholds

    def at(self, iou_threshold: float | None = None) -> DetectionTable:
        """The table at one IoU threshold (default: the lowest matched)."""
        if iou_threshold is None:
            if not self.iou_thresholds:
                return self.table
            iou_threshold = min(self.iou_thresholds)
        return self.table.at_iou_threshold(iou_threshold)

    # -- what is reported ----------------------------------------------------

    @cached_property
    def metrics(self) -> dict[str, ReportMetric]:
        """Every summary number: its name → statistic and table view."""
        ap = AP(self.interpolation)
        by_class = (COL_CLASS_ID,)
        out = {
            "map": ReportMetric(
                MeanOver(ap, undefined="exclude", label="map"), self.table
            ),
        }
        ts = self.iou_thresholds
        if len(ts) > 1:
            for t in (0.5, 0.75):
                if t in ts:
                    name = f"map_{round(t * 100)}"
                    out[name] = ReportMetric(
                        MeanOver(ap, undefined="exclude", over=by_class, label=name),
                        self.at(t),
                    )
        out["mar"] = ReportMetric(
            MeanOver(Recall(), undefined="exclude", label="mar"), self.table
        )
        if self.fp_rates:
            at = self.at()
            for r in self.fp_rates:
                name = f"sensitivity@{r:g}"
                out[name] = ReportMetric(
                    MeanOver(
                        FROCSensitivity(r, self.extrapolate),
                        undefined="exclude",
                        over=by_class,
                        label=name,
                    ),
                    at,
                )
            out["cpm"] = ReportMetric(
                MeanOver(
                    CPM(self.fp_rates, self.extrapolate),
                    undefined="exclude",
                    over=by_class,
                    label="cpm",
                ),
                at,
            )
        return out

    @cached_property
    def summary(self) -> pl.DataFrame:
        """``[metric, value]``: mAP (COCO: classes without ground truth left
        out), mAP at 0.5 / 0.75 when swept, mean recall (``mar``) and, with
        ``fp_rates``, FROC sensitivities and their mean (``cpm``)."""
        frames = [
            m.statistic.by_group(m.table).select(
                pl.lit(name).alias("metric"),
                pl.col(m.statistic.name).cast(pl.Float64).alias("value"),
            )
            for name, m in self.metrics.items()
        ]
        values = pl.collect_all(frames)
        rows = [
            df if df.height else pl.DataFrame({"metric": [name], "value": [None]})
            for name, df in zip(self.metrics, values, strict=True)
        ]
        return pl.concat(rows, how="vertical_relaxed")

    def value(self, metric: str) -> float | None:
        """One summary number by name (e.g. ``"map"``)."""
        return self.summary.filter(pl.col("metric") == metric).item(0, "value")

    @cached_property
    def per_threshold(self) -> pl.DataFrame:
        """``[iou_threshold, class_id, ap, recall, n_gts, n_preds]``, one row
        per matched threshold and class. ``ap``/``recall`` are null for a class
        without ground truth (undefined, as COCO's ``-1``)."""
        ts = self.iou_thresholds or (None,)
        frames = [self._class_rows(t) for t in ts]
        return pl.concat(pl.collect_all(frames)).sort(_THR, COL_CLASS_ID)

    def _class_rows(self, t: float | None) -> pl.LazyFrame:
        view = self.at(t)
        det, meta = view.frames([COL_CLASS_ID])
        keys = [COL_CLASS_ID]
        grid = (
            meta.group_by(keys)
            .agg(pl.col(COL_N_GTS).sum().alias("n_gts"))
            .join(weighted_gt_mass(meta, keys), on=keys, how="left", nulls_equal=True)
            .join(
                det.group_by(keys).agg(pl.len().alias("n_preds")),
                on=keys,
                how="left",
                nulls_equal=True,
            )
            .join(
                AP(self.interpolation).by_group(view, keys),
                on=keys,
                how="left",
                nulls_equal=True,
            )
            .join(Recall().by_group(view, keys), on=keys, how="left", nulls_equal=True)
        )
        defined = pl.col("gt_mass") > 0.0
        return grid.select(
            pl.lit(t, dtype=pl.Float64).alias(_THR),
            COL_CLASS_ID,
            pl.when(defined).then(pl.col("ap").fill_null(0.0)).alias("ap"),
            pl.when(defined).then(pl.col("recall").fill_null(0.0)).alias("recall"),
            pl.col("n_gts").cast(pl.Int64),
            pl.col("n_preds").fill_null(0).cast(pl.Int64),
        )

    @cached_property
    def per_class(self) -> pl.DataFrame:
        """``[class_id, ap, ap_50, ap_75, recall, n_gts, n_preds]``: AP and
        recall averaged over the matched thresholds (``ap_50``/``ap_75`` when
        those were matched)."""
        pt = self.per_threshold
        at = lambda t: (  # noqa: E731
            pl.col("ap").filter(pl.col(_THR) == t).first().alias(f"ap_{round(t * 100)}")
        )
        extra = [
            at(t)
            for t in (0.5, 0.75)
            if t in self.iou_thresholds and len(self.iou_thresholds) > 1
        ]
        return (
            pt.group_by(COL_CLASS_ID)
            .agg(
                pl.col("ap").mean(),
                *extra,
                pl.col("recall").mean(),
                pl.col("n_gts").first(),
                pl.col("n_preds").first(),
            )
            .sort(COL_CLASS_ID)
        )

    # -- intervals -----------------------------------------------------------

    def ci(
        self,
        metric: str | Sequence[str] | None = None,
        *,
        by: str | list[str] | None = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
        sample_col: str | None = None,
        strata: str | list[str] | None = None,
        weight_rtol: float | None = None,
        weight_scheme: WeightScheme = "reestimate",
    ) -> pl.DataFrame:
        """Percentile bootstrap intervals for summary numbers.

        Each is :func:`bootstrap_ci` of the statistic behind the number:
        images are resampled (stratified by whether they hold ground truth),
        and the classes and IoU thresholds a number averages over are drawn
        together, so an mAP interval reflects one resample per replicate.

        Args:
            metric: A summary name, several, or ``None`` for all of them.
            by: Optional metadata column(s) to report per group (e.g.
                ``"group_id"``); each group is resampled within itself.
            n_bootstrap: Replicates.
            confidence: Confidence level in ``(0, 1)``.
            seed: Optional seed (``None`` is a fixed default: reproducible).
            sample_col: Optional entity column (e.g. a patient id) to resample
                instead of images.
            strata: Optional metadata column(s) crossed into the weight cells.
            weight_rtol: Relative tolerance within which weights share a cell.
            weight_scheme: What the weights are, and so how the draw treats
                them: ``"reestimate"``, ``"stratified"`` or ``"fixed"`` (see
                :func:`~polars_cv.metrics.froc_auc_ci_lazy`).

        Returns:
            ``[*by, metric, value, ci_lower, ci_upper]``.
        """
        names = (
            list(self.metrics)
            if metric is None
            else [metric]
            if isinstance(metric, str)
            else list(metric)
        )
        unknown = [n for n in names if n not in self.metrics]
        if unknown:
            msg = f"unknown metric(s) {unknown}; this report has {list(self.metrics)}"
            raise ValueError(msg)
        frames = []
        for name in names:
            m = self.metrics[name]
            frames.append(
                bootstrap_ci(
                    m.table,
                    m.statistic,
                    group_by=by,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=seed,
                    sample_col=sample_col,
                    strata=strata,
                    weight_rtol=weight_rtol,
                    weight_scheme=weight_scheme,
                )
                .rename({m.statistic.name: "value"})
                .with_columns(pl.lit(name).alias("metric"))
            )
        keys = [by] if isinstance(by, str) else list(by or [])
        out = pl.concat(pl.collect_all(frames), how="vertical_relaxed")
        return out.select(*keys, "metric", "value", "ci_lower", "ci_upper")

    # -- the lower layers ----------------------------------------------------

    def pr_curve(
        self, class_id: str | None = None, iou_threshold: float | None = None
    ) -> PrecisionRecallResult:
        """The precision-recall curve of one class at one threshold."""
        from ._metrics import precision_recall_curve

        return precision_recall_curve(self.at(iou_threshold), class_id=class_id)

    def froc(
        self,
        iou_threshold: float | None = None,
        *,
        group_by: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """The FROC curve at one threshold (default: the lowest matched)."""
        from ._metrics import froc_curve_lazy

        return froc_curve_lazy(self.at(iou_threshold), group_by=group_by).collect()

    def confusion(
        self,
        score_threshold: float,
        *,
        class_id: str | None = None,
        iou_threshold: float | None = None,
    ) -> ConfusionResult:
        """TP/FP/FN of the detections scoring at least ``score_threshold``."""
        from ._metrics import confusion_at_threshold

        return confusion_at_threshold(
            self.at(iou_threshold), score_threshold, class_id=class_id
        )

    def matches(self, iou_threshold: float | None = None) -> pl.DataFrame:
        """Every prediction with its outcome at one threshold, for error analysis.

        For a report from :func:`evaluate_detections`, the prediction input's
        rows with ``is_tp``, ``iou`` and ``matched_gt_row`` (the matched
        ground truth's row in the ground-truth input). Otherwise the
        detections table.
        """
        det = self.at(iou_threshold).detections
        if self._objects is None or self._predictions is None:
            return det.collect()
        keys = [COL_IMAGE_ID, COL_CLASS_ID]
        located = det.join(self._objects, on=keys, how="left", nulls_equal=True).select(
            pl.col(PRED_ROW).list.get(pl.col("det_idx")).alias("_row"),
            "is_tp",
            "iou",
            pl.col(GT_ROW)
            .list.get(pl.col("gt_idx"), null_on_oob=True)
            .alias("matched_gt_row"),
        )
        return (
            self._predictions.with_row_index("_row")
            .join(located, on="_row", how="left", nulls_equal=True)
            .drop("_row")
            .collect()
        )

    def __repr__(self) -> str:
        width = max(len(n) for n in self.metrics)
        lines = [
            f"DetectionReport ({len(self.iou_thresholds) or 1} IoU threshold(s), "
            f"{self.interpolation} AP)"
        ]
        for name, value in self.summary.iter_rows():
            shown = "n/a" if value is None else f"{value:.4f}"
            lines.append(f"  {name:<{width}}  {shown}")
        return "\n".join(lines)
