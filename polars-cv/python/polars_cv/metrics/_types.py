"""Canonical detection table and schema constants for detection metrics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from polars._typing import EngineType

# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------

COL_IMAGE_ID = "image_id"
COL_CLASS_ID = "class_id"
COL_SCORE = "score"
COL_IS_TP = "is_tp"
COL_GT_IDX = "gt_idx"
COL_IOU = "iou"
COL_DET_IDX = "det_idx"

COL_N_GTS = "n_gts"
COL_WEIGHT = "weight"
COL_GT_LABEL = "gt_label"
COL_GROUP_ID = "group_id"
#: Present on both frames of a table matched at several IoU thresholds.
COL_IOU_THRESHOLD = "iou_threshold"

DEFAULT_CLASS = "__all__"

#: The detection frame every matcher produces: names **and** dtypes.
#:
#: The dtypes used to live only inside ``_empty_detection_table()``, which
#: hand-wrote them because this table declared names alone. Nothing tied that
#: hand-written set to what a populated match returns, so the empty and
#: non-empty paths could disagree — and a caller concatenating the two would
#: have found out at the concat, not here.
#:
#: ``test_matcher_schemas_match_the_declaration`` pins this against real matcher
#: output, so this is a record of observed behaviour rather than an intention.
DETECTION_SCHEMA: dict[str, pl.DataType] = {
    COL_IMAGE_ID: pl.String,
    COL_CLASS_ID: pl.String,
    COL_SCORE: pl.Float64,
    COL_IS_TP: pl.Boolean,
    COL_GT_IDX: pl.UInt32,
    COL_IOU: pl.Float64,
    COL_DET_IDX: pl.UInt32,
}  # ty: ignore[invalid-assignment]

#: The per-image metadata frame every matcher produces. ``group_id`` is absent
#: on purpose: it is added by :meth:`DetectionTable.with_group` and by the
#: matchers' optional ``group_col``, so it is not part of the required shape.
IMAGE_META_SCHEMA: dict[str, pl.DataType] = {
    COL_IMAGE_ID: pl.String,
    COL_CLASS_ID: pl.String,
    COL_N_GTS: pl.Int64,
    COL_WEIGHT: pl.Float64,
    COL_GT_LABEL: pl.Boolean,
}  # ty: ignore[invalid-assignment]


def _validate_schema(
    schema: pl.Schema,
    required: dict[str, pl.DataType],
    label: str,
) -> None:
    """Raise ``ValueError`` if *required*'s columns are missing from *schema*.

    Only the names are checked. ``from_matched`` accepts frames a caller
    assembled, and rejecting a ``UInt64`` ``det_idx`` that Polars will happily
    compare against a ``UInt32`` would turn a working pipeline into an error
    for no benefit. The dtypes in *required* say what the matchers *produce*,
    which is a different claim and is guarded as one.
    """
    missing = set(required) - set(schema.names())
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing)}. "
            f"Present columns: {sorted(schema.names())}"
        )


def to_lazy(data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Normalize eager/lazy inputs to ``LazyFrame``."""
    return data.lazy() if isinstance(data, pl.DataFrame) else data


@dataclass(frozen=True)
class DetectionTable:
    """Canonical intermediate representation consumed by all metric functions.

    Wraps two aligned lazy frames:

    * **detections** — one row per detection with ``image_id``, ``class_id``,
      ``score``, ``is_tp``, ``gt_idx``, ``iou``, ``det_idx``.
    * **image_metadata** — one row per (image, class) with ``n_gts``,
      ``weight``, ``gt_label``, and optionally ``group_id``.

    When the same ``image_id`` (and ``class_id``, when present) appears more
    than once in ``image_metadata`` (e.g. one rendered image owned by two
    cases), FROC/LROC resolve the per-key weight to a single value so detections
    are not fan-out-multiplied. Equal weights on the duplicates are the common
    case; disagreeing weights are resolved by the metric's ``weight_agg`` policy
    (``"first"`` default, or ``"min"``/``"max"``/``"mean"``/``"sum"``) rather than
    raising — supplying consistent weights is the caller's responsibility. Prefer
    a composite key in ``image_id`` when each ownership should be a distinct
    evaluation unit.

    Use :meth:`from_matched` to construct with schema validation.
    """

    _detections: pl.LazyFrame
    _image_meta: pl.LazyFrame
    _matching_iou_threshold: float | None = None
    #: Whether ``iou`` holds real overlaps. A pre-matched table built without
    #: an IoU column has none, and :meth:`at_iou_threshold` refuses it.
    _has_iou: bool = True
    #: The IoU thresholds of a sweep (several matchings, one per threshold,
    #: told apart by an ``iou_threshold`` column on both frames); empty for a
    #: table of one matching.
    _sweep: tuple[float, ...] = field(default=())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_matched(
        cls,
        detections: pl.LazyFrame | pl.DataFrame,
        image_meta: pl.LazyFrame | pl.DataFrame,
        *,
        matching_iou_threshold: float | None = None,
        has_iou: bool = True,
    ) -> DetectionTable:
        """Construct a ``DetectionTable`` with planning-time schema validation.

        Args:
            detections: Per-detection rows. Must contain columns ``image_id``,
                ``class_id``, ``score``, ``is_tp``, ``gt_idx``, ``iou``,
                ``det_idx``.
            image_meta: Per-image metadata. Must contain ``image_id``,
                ``class_id``, ``n_gts``, ``weight``, ``gt_label``.
            matching_iou_threshold: The IoU threshold used by the matcher.
                Stored so that :meth:`at_iou_threshold` can warn when the
                caller tries to *lower* the threshold below the matching
                level (which has no effect).
            has_iou: Whether the ``iou`` column holds real overlaps; ``False``
                when the source had none (``iou`` is then null), so
                :meth:`at_iou_threshold` refuses the table.

        Returns:
            Validated ``DetectionTable`` instance.

        Raises:
            ValueError: If required columns are missing.
        """
        det_lf = to_lazy(detections)
        meta_lf = to_lazy(image_meta)

        _validate_schema(det_lf.collect_schema(), DETECTION_SCHEMA, "detections")
        _validate_schema(meta_lf.collect_schema(), IMAGE_META_SCHEMA, "image_metadata")

        return cls(
            _detections=det_lf,
            _image_meta=meta_lf,
            _matching_iou_threshold=matching_iou_threshold,
            _has_iou=has_iou,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def detections(self) -> pl.LazyFrame:
        """Per-detection lazy frame.

        Raises:
            ValueError: On a table matched at several IoU thresholds, whose
                frames hold one matching per threshold: read them with
                :meth:`frames` grouped by ``iou_threshold``, or select one
                matching with :meth:`at_iou_threshold`.
        """
        return self.frames()[0]

    @property
    def image_metadata(self) -> pl.LazyFrame:
        """Per-image metadata lazy frame (see :attr:`detections` for a sweep)."""
        return self.frames()[1]

    @property
    def iou_thresholds(self) -> tuple[float, ...]:
        """The IoU thresholds matched: several for a sweep, one for a single
        matching at a known threshold, none when unknown (pre-matched, or a
        per-row threshold)."""
        if self._sweep:
            return self._sweep
        t = self._matching_iou_threshold
        return () if t is None else (t,)

    def frames(self, keys: Sequence[str] = ()) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """``(detections, image_metadata)`` for an evaluation grouped by ``keys``.

        The frames every metric reads. A sweep holds one matching per IoU
        threshold, so an evaluation that does not keep the thresholds apart
        would count each ground truth once per threshold: it is refused here,
        once, for every metric.

        Raises:
            ValueError: A sweep read without ``iou_threshold`` among ``keys``.
        """
        if self._sweep and COL_IOU_THRESHOLD not in keys:
            msg = (
                f"this DetectionTable was matched at {len(self._sweep)} IoU "
                f"thresholds {list(self._sweep)}; pooling them would count each "
                "ground truth once per threshold. Group by 'iou_threshold' "
                "(group_by= / by_group keys), average over it (MeanOver, e.g. "
                "mean_ap()), or select one matching with "
                ".at_iou_threshold(t)."
            )
            raise ValueError(msg)
        return self._detections, self._image_meta

    def _all_rows(self) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """Both frames unchecked, for transforms that keep every matching
        apart (filters, resampling); metrics read :meth:`frames`."""
        return self._detections, self._image_meta

    @classmethod
    def stack(cls, tables: dict[float, DetectionTable]) -> DetectionTable:
        """One sweep table from single matchings of the same data, keyed by
        their IoU threshold. A single entry is returned as it is."""
        if not tables:
            raise ValueError("stack needs at least one iou_threshold")
        if len(tables) == 1:
            return next(iter(tables.values()))
        ts = sorted(tables)
        tag = lambda t: pl.lit(float(t), dtype=pl.Float64).alias(COL_IOU_THRESHOLD)  # noqa: E731
        det = pl.concat(
            [tables[t]._detections.with_columns(tag(t)) for t in ts], how="vertical"
        )
        meta = pl.concat(
            [tables[t]._image_meta.with_columns(tag(t)) for t in ts],
            how="vertical",
        )
        first = tables[ts[0]]
        return cls(
            _detections=det,
            _image_meta=meta,
            _matching_iou_threshold=ts[0],
            _has_iou=first._has_iou,
            _sweep=tuple(float(t) for t in ts),
        )

    def meta_columns(self) -> list[str]:
        """The column names of ``image_metadata`` (resolves the schema only)."""
        return list(self._image_meta.collect_schema().names())

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    def with_group(self, group_col: str) -> DetectionTable:
        """Return a copy with ``group_id`` set from an existing metadata column.

        Args:
            group_col: Column in ``image_metadata`` to use as the group.

        Returns:
            New ``DetectionTable`` with ``group_id`` populated.
        """
        new_meta = self._image_meta.with_columns(
            pl.col(group_col).cast(pl.String).alias(COL_GROUP_ID)
        )
        return replace(self, _image_meta=new_meta)

    def filter_class(self, class_id: str) -> DetectionTable:
        """Return a copy filtered to a single class.

        Args:
            class_id: The class to retain.

        Returns:
            Filtered ``DetectionTable``.
        """
        return replace(
            self,
            _detections=self._detections.filter(pl.col(COL_CLASS_ID) == class_id),
            _image_meta=self._image_meta.filter(pl.col(COL_CLASS_ID) == class_id),
        )

    def filter_images(
        self, images: pl.Expr | Iterable[str] | pl.Series
    ) -> DetectionTable:
        """Return a copy restricted to a subset of images.

        For stratified evaluation: the same metrics over the images of one
        device, scan type or size bucket. The stored matcher settings (the
        matching IoU threshold, whether ``iou`` holds overlaps) are kept, so
        the subset is evaluated exactly as the whole table would be.

        Args:
            images: Either the ids of the images to keep (an iterable of
                ``image_id`` values), or a predicate over ``image_metadata``
                (e.g. ``pl.col("group_id") == "device_a"``). Detections follow
                the ``(image_id, class_id)`` metadata rows that are kept.

        Returns:
            Filtered ``DetectionTable``.

        Raises:
            TypeError: For a bare string, which would otherwise be read as
                a sequence of one-character ids.
        """
        if isinstance(images, str):
            msg = (
                f"filter_images takes a list of image ids or a predicate, got "
                f"the string {images!r}: pass [{images!r}] for one image"
            )
            raise TypeError(msg)
        if isinstance(images, pl.Expr):
            predicate = images
        else:
            ids = pl.Series(list(images), dtype=pl.String)
            predicate = pl.col(COL_IMAGE_ID).is_in(ids.implode())
        meta = self._image_meta.filter(predicate)
        on = [COL_IMAGE_ID, COL_CLASS_ID, *self._sweep_key()]
        keys = meta.select(on).unique()
        return replace(
            self,
            _detections=self._detections.join(keys, on=on, how="semi"),
            _image_meta=meta,
        )

    def _sweep_key(self) -> list[str]:
        """``["iou_threshold"]`` on a sweep, else nothing: the key that keeps a
        sweep's matchings apart in per-image work."""
        return [COL_IOU_THRESHOLD] if self._sweep else []

    def class_ids(self) -> list[str]:
        """Return distinct class IDs present in the detections.

        Note: triggers a partial collect on the metadata frame.
        """
        return (
            self._image_meta.select(pl.col(COL_CLASS_ID).unique())
            .collect()
            .get_column(COL_CLASS_ID)
            .to_list()
        )

    def to_per_image(self) -> pl.LazyFrame:
        """One row per image (per ``(image_id, class_id)``) with its detections summarised.

        The image metadata (``gt_label``, ``weight``, ``n_gts`` …) plus:

        - ``max_score``: the highest detection score (null without detections);
        - ``top_is_tp``: whether the highest-scoring detection is a true
          positive. Tied top scores go to the earliest detection (lowest
          ``det_idx``), the order the matcher ranked them in;
        - ``best_tp_score``: the highest true-positive score (null without one).

        LROC's per-image commitment (``best_tp`` / ``top_scoring``) reads these.
        Every aggregate is native to the streaming engine; no per-image list is
        built.
        """
        from ._grouped_scan import IsFirst, grouped_scan

        keys = [COL_IMAGE_ID, COL_CLASS_ID, *self._sweep_key()]
        det = self._detections.filter(pl.col(COL_SCORE).is_not_null())
        stats = det.group_by(keys).agg(
            max_score=pl.col(COL_SCORE).max(),
            best_tp_score=pl.when(pl.col(COL_IS_TP)).then(pl.col(COL_SCORE)).max(),
        )
        top = (
            grouped_scan(
                det.select(*keys, COL_SCORE, COL_DET_IDX, COL_IS_TP),
                keys,
                by=[COL_SCORE, COL_DET_IDX],
                descending=[True, False],
                _top=IsFirst(),
            )
            .filter(pl.col("_top"))
            .select(*keys, top_is_tp=pl.col(COL_IS_TP))
        )
        return (
            self._image_meta.join(stats, on=keys, how="left")
            .join(top, on=keys, how="left")
            .with_columns(pl.col("top_is_tp").fill_null(False))
        )

    # ------------------------------------------------------------------
    # Re-threshold helper
    # ------------------------------------------------------------------

    def at_iou_threshold(self, iou_threshold: float) -> DetectionTable:
        """Return the table evaluated at one IoU threshold.

        On a sweep (a table matched at several thresholds) a matched threshold
        is selected exactly: its own matching, as the matcher produced it at
        that threshold. Any other threshold — and every threshold of a single
        matching — re-thresholds a matching instead: the stored ``iou`` is
        compared against *iou_threshold* to set ``is_tp``, without matching
        again (on a sweep, the matching at the highest threshold below it).
        Re-thresholding is not re-matching: a detection that lost its ground
        truth at the matching threshold to a higher-scoring one is not given
        it back, which COCO's per-threshold matching would do.

        .. warning::

            Re-thresholding only works reliably when *raising* the threshold
            above the original matching IoU. Lowering it has no effect because
            detections that were unmatched at the original threshold have no
            stored ``gt_idx``/``iou`` to re-evaluate.

        Args:
            iou_threshold: New IoU threshold to apply.

        Returns:
            ``DetectionTable`` with updated ``is_tp``.

        Raises:
            ValueError: If the table has no IoU values (a pre-matched table
                built without ``iou_col``).
        """
        if self._sweep:
            matched = [t for t in self._sweep if t <= iou_threshold]
            base = max(matched) if matched else min(self._sweep)
            one = self._slice(base)
            return one if base == iou_threshold else one.at_iou_threshold(iou_threshold)
        if not self._has_iou:
            raise ValueError(
                "this DetectionTable has no IoU values (it was pre-matched "
                "without iou_col), so it cannot be re-thresholded by IoU. Pass "
                "iou_col to PreMatchedAdapter.match, or evaluate at the "
                "matching the TP flags already encode."
            )
        if (
            self._matching_iou_threshold is not None
            and iou_threshold < self._matching_iou_threshold
        ):
            warnings.warn(
                f"Lowering IoU threshold to {iou_threshold} below the matching "
                f"threshold of {self._matching_iou_threshold} has no effect — "
                f"detections unmatched at the original threshold cannot be "
                f"retroactively matched. Re-run the matcher at the lower "
                f"threshold instead.",
                UserWarning,
                stacklevel=2,
            )

        new_det = self._detections.with_columns(
            pl.when(
                pl.col(COL_GT_IDX).is_not_null() & (pl.col(COL_IOU) >= iou_threshold)
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias(COL_IS_TP)
        )
        return replace(self, _detections=new_det)

    def _slice(self, iou_threshold: float) -> DetectionTable:
        """A sweep's own matching at one of its thresholds."""
        at = pl.col(COL_IOU_THRESHOLD) == iou_threshold
        return replace(
            self,
            _detections=self._detections.filter(at).drop(COL_IOU_THRESHOLD),
            _image_meta=self._image_meta.filter(at).drop(COL_IOU_THRESHOLD),
            _matching_iou_threshold=iou_threshold,
            _sweep=(),
        )

    # ------------------------------------------------------------------
    # Collect helper
    # ------------------------------------------------------------------

    def collect(self, engine: EngineType = "auto") -> tuple[pl.DataFrame, pl.DataFrame]:
        """Materialize both frames in one pass.

        Uses ``pl.collect_all`` so that a shared upstream subplan (e.g. the
        matcher's cached contour-extraction/correspond graph, feeding both the
        detections and the image metadata) executes once across both frames
        rather than once per frame. Common-subplan elimination is what makes the
        decode-once, collect-once path hold end to end.

        Args:
            engine: Polars' engine. ``"auto"`` (the default) is polars' own
                choice: the streaming engine, unless
                ``pl.Config.set_engine_affinity`` says otherwise.

        Returns:
            Tuple of ``(detections_df, image_meta_df)``.
        """
        det_df, meta_df = pl.collect_all(
            [self._detections, self._image_meta], engine=engine
        )
        return det_df, meta_df


def ensure_columns_exist(
    columns: Sequence[str],
    required: Sequence[str],
) -> None:
    """Validate that all required columns exist.

    Args:
        columns: Available column names.
        required: Column names that must be present.

    Raises:
        ValueError: If any required column is absent.
    """
    col_set = set(columns)
    for name in required:
        if name not in col_set:
            raise ValueError(f"Required column `{name}` not found.")
