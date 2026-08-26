"""Tests for bounding-box matching Rust primitives and BBoxNamespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


BBOX_SCHEMA = pl.Struct(
    [
        pl.Field("x", pl.Float64),
        pl.Field("y", pl.Float64),
        pl.Field("width", pl.Float64),
        pl.Field("height", pl.Float64),
    ]
)


def _make_bbox_df() -> pl.DataFrame:
    """Create a DataFrame with prediction and GT bbox columns."""
    return pl.DataFrame(
        {
            "pred_bboxes": [
                [
                    {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
                    {"x": 20.0, "y": 20.0, "width": 5.0, "height": 5.0},
                ],
                [
                    {"x": 50.0, "y": 50.0, "width": 10.0, "height": 10.0},
                ],
            ],
            "gt_bboxes": [
                [
                    {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
                ],
                [
                    {"x": 55.0, "y": 55.0, "width": 8.0, "height": 8.0},
                ],
            ],
            "pred_scores": [
                [0.9, 0.5],
                [0.8],
            ],
        }
    ).cast(
        {
            "pred_bboxes": pl.List(BBOX_SCHEMA),
            "gt_bboxes": pl.List(BBOX_SCHEMA),
        }
    )


@plugin_required
class TestBBoxPairwiseIou:
    """Tests for bbox.pairwise_iou plugin function."""

    def test_pairwise_iou_identical(self) -> None:
        """Identical boxes should have IoU = 1.0."""
        import polars_cv  # noqa: F401

        df = pl.DataFrame(
            {
                "a": [[{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}]],
                "b": [[{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}]],
            }
        ).cast({"a": pl.List(BBOX_SCHEMA), "b": pl.List(BBOX_SCHEMA)})

        result = df.with_columns(iou=pl.col("a").bbox.pairwise_iou(pl.col("b")))
        iou_matrix = result["iou"].to_list()[0]
        assert abs(iou_matrix[0][0] - 1.0) < 0.01

    def test_pairwise_iou_no_overlap(self) -> None:
        """Non-overlapping boxes should have IoU = 0."""
        import polars_cv  # noqa: F401

        df = pl.DataFrame(
            {
                "a": [[{"x": 0.0, "y": 0.0, "width": 5.0, "height": 5.0}]],
                "b": [[{"x": 20.0, "y": 20.0, "width": 5.0, "height": 5.0}]],
            }
        ).cast({"a": pl.List(BBOX_SCHEMA), "b": pl.List(BBOX_SCHEMA)})

        result = df.with_columns(iou=pl.col("a").bbox.pairwise_iou(pl.col("b")))
        iou_matrix = result["iou"].to_list()[0]
        assert iou_matrix[0][0] < 0.01
