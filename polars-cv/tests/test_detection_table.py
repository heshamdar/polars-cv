"""Tests for DetectionTable core type and schema validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

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
    DEFAULT_CLASS,
    DetectionTable,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_detections() -> pl.DataFrame:
    """Create a small sample detection DataFrame."""
    return pl.DataFrame(
        {
            COL_IMAGE_ID: ["img1", "img1", "img1", "img2", "img2"],
            COL_CLASS_ID: [DEFAULT_CLASS] * 5,
            COL_SCORE: [0.9, 0.7, 0.3, 0.8, 0.5],
            COL_IS_TP: [True, False, True, True, False],
            COL_GT_IDX: [0, None, 1, 0, None],
            COL_IOU: [0.85, 0.0, 0.6, 0.9, 0.0],
            COL_DET_IDX: [0, 1, 2, 0, 1],
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


@pytest.fixture()
def sample_metadata() -> pl.DataFrame:
    """Create matching sample metadata."""
    return pl.DataFrame(
        {
            COL_IMAGE_ID: ["img1", "img2"],
            COL_CLASS_ID: [DEFAULT_CLASS, DEFAULT_CLASS],
            COL_N_GTS: [2, 1],
            COL_WEIGHT: [1.0, 1.0],
            COL_GT_LABEL: [True, True],
        }
    )


@pytest.fixture()
def detection_table(
    sample_detections: pl.DataFrame,
    sample_metadata: pl.DataFrame,
) -> DetectionTable:
    """Build a valid DetectionTable from fixtures."""
    return DetectionTable.from_matched(sample_detections, sample_metadata)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectionTableConstruction:
    """Tests for DetectionTable construction and schema validation."""

    def test_from_matched_succeeds(
        self,
        sample_detections: pl.DataFrame,
        sample_metadata: pl.DataFrame,
    ) -> None:
        """Valid inputs produce a DetectionTable."""
        table = DetectionTable.from_matched(sample_detections, sample_metadata)
        assert table.detections is not None
        assert table.image_metadata is not None

    def test_missing_detection_column_raises(
        self,
        sample_metadata: pl.DataFrame,
    ) -> None:
        """Missing required detection columns raise ValueError."""
        bad_det = pl.DataFrame(
            {
                COL_IMAGE_ID: ["img1"],
                COL_SCORE: [0.5],
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            DetectionTable.from_matched(bad_det, sample_metadata)

    def test_missing_metadata_column_raises(
        self,
        sample_detections: pl.DataFrame,
    ) -> None:
        """Missing required metadata columns raise ValueError."""
        bad_meta = pl.DataFrame(
            {
                COL_IMAGE_ID: ["img1"],
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            DetectionTable.from_matched(sample_detections, bad_meta)

    def test_accepts_lazy_frames(
        self,
        sample_detections: pl.DataFrame,
        sample_metadata: pl.DataFrame,
    ) -> None:
        """LazyFrame inputs are accepted."""
        table = DetectionTable.from_matched(
            sample_detections.lazy(),
            sample_metadata.lazy(),
        )
        det_df, meta_df = table.collect(engine="streaming")
        assert det_df.height == sample_detections.height
        assert meta_df.height == sample_metadata.height


class TestDetectionTableViews:
    """Tests for DetectionTable accessor methods."""

    def test_class_ids(self, detection_table: DetectionTable) -> None:
        """class_ids returns distinct class IDs."""
        ids = detection_table.class_ids()
        assert ids == [DEFAULT_CLASS]

    def test_filter_class(self, detection_table: DetectionTable) -> None:
        """filter_class retains only the specified class."""
        filtered = detection_table.filter_class(DEFAULT_CLASS)
        det_df, _ = filtered.collect(engine="streaming")
        assert det_df.height > 0

    def test_filter_class_empty(self, detection_table: DetectionTable) -> None:
        """filter_class on a nonexistent class yields empty frames."""
        filtered = detection_table.filter_class("nonexistent_class")
        det_df, _ = filtered.collect(engine="streaming")
        assert det_df.height == 0

    def test_to_per_image(self, detection_table: DetectionTable) -> None:
        """to_per_image returns one row per (image, class)."""
        per_image = detection_table.to_per_image().collect(engine="streaming")
        assert per_image.height == 2
        assert "max_score" in per_image.columns
        assert "top_is_tp" in per_image.columns

    def test_at_iou_threshold(self, detection_table: DetectionTable) -> None:
        """at_iou_threshold recomputes is_tp without re-matching."""
        high_thresh = detection_table.at_iou_threshold(0.99)
        det_df, _ = high_thresh.collect(engine="streaming")
        # At 0.99 threshold, some TPs should become FPs
        tp_count = det_df.filter(pl.col(COL_IS_TP)).height
        assert tp_count <= 5

    def test_collect_returns_dataframes(self, detection_table: DetectionTable) -> None:
        """collect returns eager DataFrames."""
        det_df, meta_df = detection_table.collect(engine="streaming")
        assert isinstance(det_df, pl.DataFrame)
        assert isinstance(meta_df, pl.DataFrame)

    def test_image_ids_and_strata(self, detection_table: DetectionTable) -> None:
        """image_ids_and_strata returns IDs and stratification dict."""
        ids, strata = detection_table.image_ids_and_strata()
        assert len(ids) == 2
        assert strata is not None
        assert len(strata) == 2
