"""Canonical detection table and schema constants for detection metrics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

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

DEFAULT_CLASS = "__all__"

DETECTION_COLUMNS: set[str] = {
    COL_IMAGE_ID,
    COL_CLASS_ID,
    COL_SCORE,
    COL_IS_TP,
    COL_GT_IDX,
    COL_IOU,
    COL_DET_IDX,
}

IMAGE_META_REQUIRED: set[str] = {
    COL_IMAGE_ID,
    COL_CLASS_ID,
    COL_N_GTS,
    COL_WEIGHT,
    COL_GT_LABEL,
}


def _validate_schema(
    schema: pl.Schema,
    required: set[str],
    label: str,
) -> None:
    """Raise ``ValueError`` if *required* columns are missing from *schema*."""
    missing = required - set(schema.names())
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
    cases), FROC weight lookups dedupe by that key so detections are not
    fan-out-multiplied. Equal weights on the duplicates are fine; conflicting
    weights raise ``ValueError`` because the numerator would pick an arbitrary
    row while denominators sum every row. Prefer a composite key in
    ``image_id`` when each ownership should be a distinct evaluation unit.

    Use :meth:`from_matched` to construct with schema validation.
    """

    _detections: pl.LazyFrame
    _image_meta: pl.LazyFrame
    _matching_iou_threshold: float | None = None

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

        Returns:
            Validated ``DetectionTable`` instance.

        Raises:
            ValueError: If required columns are missing.
        """
        det_lf = to_lazy(detections)
        meta_lf = to_lazy(image_meta)

        _validate_schema(det_lf.collect_schema(), DETECTION_COLUMNS, "detections")
        _validate_schema(
            meta_lf.collect_schema(), IMAGE_META_REQUIRED, "image_metadata"
        )

        return cls(
            _detections=det_lf,
            _image_meta=meta_lf,
            _matching_iou_threshold=matching_iou_threshold,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def detections(self) -> pl.LazyFrame:
        """Per-detection lazy frame."""
        return self._detections

    @property
    def image_metadata(self) -> pl.LazyFrame:
        """Per-image metadata lazy frame."""
        return self._image_meta

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
        return DetectionTable(
            _detections=self._detections,
            _image_meta=new_meta,
            _matching_iou_threshold=self._matching_iou_threshold,
        )

    def filter_class(self, class_id: str) -> DetectionTable:
        """Return a copy filtered to a single class.

        Args:
            class_id: The class to retain.

        Returns:
            Filtered ``DetectionTable``.
        """
        return DetectionTable(
            _detections=self._detections.filter(pl.col(COL_CLASS_ID) == class_id),
            _image_meta=self._image_meta.filter(pl.col(COL_CLASS_ID) == class_id),
            _matching_iou_threshold=self._matching_iou_threshold,
        )

    def class_ids(self) -> list[str]:
        """Return distinct class IDs present in the detections.

        Note: triggers a partial collect on the metadata frame.
        """
        return (
            self._image_meta.select(pl.col(COL_CLASS_ID).unique())
            .collect(engine="streaming")
            .get_column(COL_CLASS_ID)
            .to_list()
        )

    def to_per_image(self) -> pl.LazyFrame:
        """Aggregate detections to one row per image with top-scoring detection.

        Produces one row per ``(image_id, class_id)`` with:

        - ``detections``: list of detection structs sorted by score descending
        - compatibility columns ``max_score`` and ``top_is_tp`` from the
          highest-scoring detection
        - metadata columns ``gt_label``, ``weight``, ``n_gts``

        LROC consumes ``detections`` for image-level summarization (best
        localized detection), rather than relying only on the top-scoring
        detection.
        """
        top_det = self._detections.group_by(COL_IMAGE_ID, COL_CLASS_ID).agg(
            detections=pl.struct(
                [COL_SCORE, COL_IS_TP, COL_GT_IDX, COL_IOU, COL_DET_IDX]
            ).sort_by(COL_SCORE, descending=True),
            max_score=pl.col(COL_SCORE).max(),
            top_is_tp=pl.col(COL_IS_TP).sort_by(COL_SCORE, descending=True).first(),
        )
        return self._image_meta.join(
            top_det, on=[COL_IMAGE_ID, COL_CLASS_ID], how="left"
        ).with_columns(
            pl.col("top_is_tp").fill_null(False),
            pl.col("max_score"),
        )

    # ------------------------------------------------------------------
    # Re-threshold helper
    # ------------------------------------------------------------------

    def at_iou_threshold(self, iou_threshold: float) -> DetectionTable:
        """Return a copy with ``is_tp`` recomputed at a different IoU threshold.

        The stored ``iou`` column is compared against *iou_threshold* to set
        ``is_tp`` without re-running the matching step.

        .. warning::

            Re-thresholding only works reliably when *raising* the threshold
            above the original matching IoU. Lowering it has no effect because
            detections that were unmatched at the original threshold have no
            stored ``gt_idx``/``iou`` to re-evaluate.

        Args:
            iou_threshold: New IoU threshold to apply.

        Returns:
            ``DetectionTable`` with updated ``is_tp``.
        """
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
        return DetectionTable(
            _detections=new_det,
            _image_meta=self._image_meta,
            _matching_iou_threshold=self._matching_iou_threshold,
        )

    # ------------------------------------------------------------------
    # Collect helper
    # ------------------------------------------------------------------

    def collect(self, engine: str = "streaming") -> tuple[pl.DataFrame, pl.DataFrame]:
        """Materialize both frames.

        Returns:
            Tuple of ``(detections_df, image_meta_df)``.
        """
        return (
            self._detections.collect(engine=engine),
            self._image_meta.collect(engine=engine),
        )

    # ------------------------------------------------------------------
    # Image IDs and strata (for bootstrap)
    # ------------------------------------------------------------------

    def image_ids_and_strata(
        self,
    ) -> tuple[list[str], dict[str, str] | None]:
        """Extract image IDs and optional stratification mapping.

        Returns:
            Tuple of ``(image_ids, strata_dict | None)``.
        """
        meta_df = (
            self._image_meta.select(COL_IMAGE_ID, COL_GT_LABEL)
            .unique()
            .collect(engine="streaming")
        )
        image_ids = [str(v) for v in meta_df[COL_IMAGE_ID].to_list()]
        strata = {
            str(iid): str(lbl)
            for iid, lbl in zip(
                meta_df[COL_IMAGE_ID].to_list(),
                meta_df[COL_GT_LABEL].to_list(),
                strict=True,
            )
        }
        return image_ids, strata


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
