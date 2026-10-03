"""One call from predictions and ground truth to a report.

:func:`evaluate_detections` (object tables: boxes, polygons, instance masks)
and :func:`evaluate_heatmaps` (dense predictions against masks) match with the
existing entry points and return a :class:`DetectionReport`. A report holds
the :class:`DetectionTable` and reads every number it shows from the same
:class:`Statistic` objects :func:`bootstrap_ci` takes, so each reported number
has a confidence interval through one engine (:meth:`DetectionReport.ci`),
and every lower-level metric remains one method away.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from ._bootstrap import bootstrap_ci
from ._inputs import GT_ROW, PRED_ROW, group_objects, match_detections
from ._statistics import (
    AP,
    CPM,
    FROC_RATES,
    FROCSensitivity,
    MeanOver,
    Recall,
    Statistic,
)
from ._types import COL_CLASS_ID, COL_IMAGE_ID, COL_N_GTS, DetectionTable
from ._weights import weighted_gt_mass

if TYPE_CHECKING:
    from ..geometry.coords import BoxFormat
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
        values = pl.collect_all(frames, engine="streaming")
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
        return pl.concat(pl.collect_all(frames, engine="streaming")).sort(
            _THR, COL_CLASS_ID
        )

    def _class_rows(self, t: float | None) -> pl.LazyFrame:
        view = self.at(t)
        det, meta = view.frames([COL_CLASS_ID])
        keys = [COL_CLASS_ID]
        grid = (
            meta.group_by(keys)
            .agg(pl.col(COL_N_GTS).sum().alias("n_gts"))
            .join(weighted_gt_mass(meta, keys), on=keys, how="left")
            .join(
                det.group_by(keys).agg(pl.len().alias("n_preds")), on=keys, how="left"
            )
            .join(AP(self.interpolation).by_group(view, keys), on=keys, how="left")
            .join(Recall().by_group(view, keys), on=keys, how="left")
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
                )
                .rename({m.statistic.name: "value"})
                .with_columns(pl.lit(name).alias("metric"))
            )
        keys = [by] if isinstance(by, str) else list(by or [])
        out = pl.concat(
            pl.collect_all(frames, engine="streaming"), how="vertical_relaxed"
        )
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

        return froc_curve_lazy(self.at(iou_threshold), group_by=group_by).collect(
            engine="streaming"
        )

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
            return det.collect(engine="streaming")
        keys = [COL_IMAGE_ID, COL_CLASS_ID]
        located = det.join(self._objects, on=keys, how="left").select(
            pl.col(PRED_ROW).list.get(pl.col("det_idx")).alias("_row"),
            "is_tp",
            "iou",
            pl.col(GT_ROW)
            .list.get(pl.col("gt_idx"), null_on_oob=True)
            .alias("matched_gt_row"),
        )
        return (
            self._predictions.with_row_index("_row")
            .join(located, on="_row", how="left")
            .drop("_row")
            .collect(engine="streaming")
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


def evaluate_detections(
    predictions: pl.DataFrame | pl.LazyFrame,
    ground_truth: pl.DataFrame | pl.LazyFrame,
    *,
    geometry: str | Sequence[str] = "bbox",
    box_format: BoxFormat | None = None,
    image_id: str = COL_IMAGE_ID,
    class_id: str | None = COL_CLASS_ID,
    score: str = "score",
    iou_thresholds: Literal["coco"] | float | Sequence[float] = "coco",
    interpolation: APInterpolation | None = None,
    max_detections: int | None | Literal["coco"] = "coco",
    images: Iterable[str] | pl.Series | pl.DataFrame | pl.LazyFrame | None = None,
    weight: str | None = None,
    group: str | None = None,
    fp_rates: Sequence[float] | None = None,
    extrapolate: Extrapolate = "none",
) -> DetectionReport:
    """Evaluate object detections (boxes, polygons or instance masks).

    The one-call form of :func:`match_detections` plus the summary metrics,
    COCO-style by default::

        report = evaluate_detections(preds, gts, box_format="xyxy")
        report.summary            # map, map_50, map_75, mar
        report.per_class          # AP per class
        report.ci("map")          # bootstrap interval
        report.matches(0.5)       # each prediction's TP/FP, for error analysis

    With the defaults (``iou_thresholds="coco"``) matching is redone at each
    IoU threshold 0.50:0.05:0.95, AP is COCO's 101-point interpolation, each
    image keeps its 100 highest-scoring predictions per class, and a class
    without ground truth is left out of the means — COCO's evaluation, except
    that it has no area ranges and no crowd/ignore regions (and ties between
    equal scores form one PR point rather than following input order).

    Args:
        predictions: One row per predicted object (``image_id``, geometry,
            ``score``, ``class_id``).
        ground_truth: One row per ground-truth object.
        geometry: The geometry column (or four coordinate columns); see
            :func:`match_detections` for the accepted forms.
        box_format: ``"xyxy"``, ``"xywh"`` or ``"cxcywh"`` for boxes given
            as four numbers.
        image_id: The image identifier column.
        class_id: The class column, or ``None`` for one class.
        score: The prediction confidence column.
        iou_thresholds: ``"coco"``, one threshold (e.g. ``0.5``, Pascal VOC)
            or several.
        interpolation: AP interpolation; ``None`` is ``"101_point"`` for
            ``"coco"`` and ``"all_points"`` otherwise.
        max_detections: Per (image, class) cap on predictions; ``"coco"`` is
            100 with ``iou_thresholds="coco"`` and no cap otherwise.
        images: The image population (see :func:`match_detections`).
        weight: A per-image weight column of the ``images`` frame.
        group: A per-image subgroup column of the ``images`` frame.
        fp_rates: FROC operating points to summarise (``FROC_RATES`` is the
            radiology set), or ``None``.
        extrapolate: How FROC curves are read past their last point.

    Returns:
        A :class:`DetectionReport`.
    """
    coco = isinstance(iou_thresholds, str)
    if coco and iou_thresholds != "coco":
        raise ValueError(
            f"iou_thresholds must be 'coco' or numbers, got {iou_thresholds!r}"
        )
    thresholds = COCO_IOU_THRESHOLDS if coco else iou_thresholds
    cap = (100 if coco else None) if max_detections == "coco" else max_detections
    interp: APInterpolation = interpolation or ("101_point" if coco else "all_points")
    table = match_detections(
        predictions,
        ground_truth,
        geometry=geometry,
        box_format=box_format,
        image_id=image_id,
        class_id=class_id,
        score=score,
        iou_thresholds=thresholds,  # type: ignore[arg-type]
        max_detections=cap,
        images=images,
        weight=weight,
        group=group,
    )
    geo = geometry if isinstance(geometry, str) else next(iter(geometry))
    objects = group_objects(
        predictions,
        ground_truth,
        geometry=geo,
        image_id=image_id,
        class_id=class_id,
        score=score,
        images=images,
        max_detections=cap,
    ).select(COL_IMAGE_ID, COL_CLASS_ID, PRED_ROW, GT_ROW)
    return DetectionReport(
        table=_materialized(table),
        interpolation=interp,
        fp_rates=tuple(fp_rates) if fp_rates else None,
        extrapolate=extrapolate,
        _objects=objects,
        _predictions=predictions.lazy(),
        _ground_truth=ground_truth.lazy(),
    )


def evaluate_heatmaps(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    heatmap: str,
    gt: str,
    image_id: str | None = None,
    class_id: str | None = None,
    iou_threshold: float | Sequence[float] = 0.5,
    fp_rates: Sequence[float] = FROC_RATES,
    extrapolate: Extrapolate = "none",
    interpolation: APInterpolation = "all_points",
    weight: str | None = None,
    group: str | None = None,
    **matcher_options: Any,
) -> DetectionReport:
    """Evaluate dense predictions (heatmaps, probability maps) against masks.

    One row per image: each heatmap is thresholded into candidate regions,
    each region scored from the heatmap, and the regions matched to the
    ground-truth mask's regions by :class:`ContourMatcher` (any of its options
    can be passed through, e.g. ``match_by="coverage"``,
    ``extraction_threshold=0.3``, ``score_reduction="mean"``). The report
    summarises FROC — sensitivity at each of ``fp_rates`` and their mean, the
    CPM — beside AP::

        report = evaluate_heatmaps(df, heatmap="prob", gt="mask")
        report.summary              # map, mar, sensitivity@…, cpm
        report.ci("cpm")

    Args:
        data: One row per image.
        heatmap: The prediction column (any format a polars-cv source reads).
        gt: The ground-truth mask (or contour) column.
        image_id: An image identifier column (default: the row index).
        class_id: A class column (one row per image and class).
        iou_threshold: The matching threshold, or several.
        fp_rates: FROC operating points (false positives per image).
        extrapolate: How the FROC curve is read past its last point
            (``"flat"`` is LUNA16's convention).
        interpolation: AP interpolation.
        weight: A per-image weight column of ``data``.
        group: A per-image subgroup column of ``data``.
        **matcher_options: :class:`ContourMatcher` options.

    Returns:
        A :class:`DetectionReport`.
    """
    from ._matching import ContourMatcher

    # Unknown options are refused by ContourMatcher's own signature.
    matcher = ContourMatcher(iou_threshold=iou_threshold, **matcher_options)
    table = matcher.match(
        data,
        pred_col=heatmap,
        gt_col=gt,
        class_col=class_id,
        image_id_col=image_id,
        weight_col=weight,
        group_col=group,
    )
    return DetectionReport(
        table=_materialized(table),
        interpolation=interpolation,
        fp_rates=tuple(fp_rates),
        extrapolate=extrapolate,
    )
