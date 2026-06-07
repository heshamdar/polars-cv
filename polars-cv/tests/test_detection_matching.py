"""
Integration tests for detection-matching contour primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.geometry import CONTOUR_SET_SCHEMA, MATCH_RESULT_SCHEMA
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


def _square(x: float, y: float, size: float) -> dict[str, object]:
    """
    Create a square contour dictionary.

    Args:
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        size: Side length.

    Returns:
        A contour dictionary matching CONTOUR_SCHEMA.
    """
    return {
        "exterior": [
            {"x": x, "y": y},
            {"x": x + size, "y": y},
            {"x": x + size, "y": y + size},
            {"x": x, "y": y + size},
        ],
        "holes": [],
        "is_closed": True,
    }


@plugin_required
class TestPairwiseIoUPrimitive:
    """Tests for `contour.pairwise_iou`."""

    def test_pairwise_iou_matrix_shape(self) -> None:
        """It returns an N x M matrix aligned to contour-set inputs."""
        df = pl.DataFrame(
            {
                "preds": [[_square(0.0, 0.0, 10.0), _square(20.0, 20.0, 10.0)]],
                "gts": [[_square(0.0, 0.0, 10.0), _square(50.0, 50.0, 10.0)]],
            },
            schema={"preds": CONTOUR_SET_SCHEMA, "gts": CONTOUR_SET_SCHEMA},
        )
        out = df.with_columns(
            iou_matrix=pl.col("preds").contour.pairwise_iou(pl.col("gts"))
        )
        matrix = out["iou_matrix"][0]
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert matrix[0][0] == pytest.approx(1.0)
        assert matrix[0][1] == pytest.approx(0.0)

    def test_pairwise_iou_empty_predictions(self) -> None:
        """Empty predictions yield an empty outer list."""
        df = pl.DataFrame(
            {"preds": [[]], "gts": [[_square(0.0, 0.0, 10.0)]]},
            schema={"preds": CONTOUR_SET_SCHEMA, "gts": CONTOUR_SET_SCHEMA},
        )
        out = df.with_columns(
            iou_matrix=pl.col("preds").contour.pairwise_iou(pl.col("gts"))
        )
        assert len(out["iou_matrix"][0]) == 0


@plugin_required
class TestMatchDetectionsPrimitive:
    """Tests for `contour.match_detections`."""

    def test_match_detections_count_invariants(self) -> None:
        """The output counts satisfy TP/FP/FN invariants."""
        df = pl.DataFrame(
            {
                "preds": [[_square(0.0, 0.0, 10.0), _square(20.0, 20.0, 10.0)]],
                "gts": [[_square(0.0, 0.0, 10.0)]],
                "scores": [[0.9, 0.1]],
            },
            schema={
                "preds": CONTOUR_SET_SCHEMA,
                "gts": CONTOUR_SET_SCHEMA,
                "scores": pl.List(pl.Float64),
            },
        )

        out = df.with_columns(
            match=pl.col("preds").contour.match_detections(
                pl.col("gts"), threshold=0.5, scores=pl.col("scores")
            )
        )
        assert out["match"].dtype == MATCH_RESULT_SCHEMA
        match = out["match"][0]

        assert match["n_preds"] == 2
        assert match["n_gts"] == 1
        assert match["n_tp"] == 1
        assert match["n_fp"] == 1
        assert match["n_fn"] == 0
        assert match["n_tp"] + match["n_fp"] == match["n_preds"]
        assert match["n_tp"] + match["n_fn"] == match["n_gts"]

    def test_match_detections_empty_predictions_all_fn(self) -> None:
        """Empty predictions produce only false negatives."""
        df = pl.DataFrame(
            {"preds": [[]], "gts": [[_square(0.0, 0.0, 10.0)]]},
            schema={"preds": CONTOUR_SET_SCHEMA, "gts": CONTOUR_SET_SCHEMA},
        )
        out = df.with_columns(
            match=pl.col("preds").contour.match_detections(pl.col("gts"), threshold=0.5)
        )
        match = out["match"][0]
        assert match["n_preds"] == 0
        assert match["n_gts"] == 1
        assert match["n_tp"] == 0
        assert match["n_fp"] == 0
        assert match["n_fn"] == 1

    def test_match_detections_score_length_mismatch_errors(self) -> None:
        """Mismatched score length raises a descriptive compute error."""
        df = pl.DataFrame(
            {
                "preds": [[_square(0.0, 0.0, 10.0), _square(10.0, 10.0, 10.0)]],
                "gts": [[_square(0.0, 0.0, 10.0)]],
                "scores": [[0.8]],
            },
            schema={
                "preds": CONTOUR_SET_SCHEMA,
                "gts": CONTOUR_SET_SCHEMA,
                "scores": pl.List(pl.Float64),
            },
        )
        with pytest.raises(Exception, match="scores length"):
            df.with_columns(
                match=pl.col("preds").contour.match_detections(
                    pl.col("gts"), scores=pl.col("scores")
                )
            ).collect(engine="streaming")


@plugin_required
class TestLabelReducePrimitive:
    """Tests for `contour.label_reduce`."""

    def test_label_reduce_bbox_reductions(self) -> None:
        """BBox mode supports max/mean/sum reductions."""
        contour = _square(0.0, 0.0, 2.0)
        df = pl.DataFrame(
            {
                "preds": [[contour]],
                "heatmap": [[[1.0, 2.0], [3.0, 4.0]]],
            },
            schema={
                "preds": CONTOUR_SET_SCHEMA,
                "heatmap": pl.List(pl.List(pl.Float64)),
            },
        )
        out = df.with_columns(
            s_max=pl.col("preds").contour.label_reduce(
                pl.col("heatmap"), reduction="max", region_mode="bbox"
            ),
            s_mean=pl.col("preds").contour.label_reduce(
                pl.col("heatmap"), reduction="mean", region_mode="bbox"
            ),
            s_sum=pl.col("preds").contour.label_reduce(
                pl.col("heatmap"), reduction="sum", region_mode="bbox"
            ),
        )
        assert out["s_max"][0][0] == pytest.approx(4.0)
        assert out["s_mean"][0][0] == pytest.approx(2.5)
        assert out["s_sum"][0][0] == pytest.approx(10.0)

    def test_buffer_space_label_reduce_matches_contour_namespace(self) -> None:
        """Buffer-space and contour-space label reduction produce identical scores."""
        contour = _square(0.0, 0.0, 2.0)
        df = pl.DataFrame(
            {
                "preds": [[contour]],
                "image": [[[1.0, 2.0], [3.0, 4.0]]],
            },
            schema={
                "preds": CONTOUR_SET_SCHEMA,
                "image": pl.List(pl.List(pl.Float64)),
            },
        )
        score_pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .label_reduce(contours=pl.col("preds"), reduction="max", region_mode="bbox")
        )
        out = df.with_columns(
            contour_scores=pl.col("preds").contour.label_reduce(
                image=pl.col("image"), reduction="max", region_mode="bbox"
            ),
            buffer_scores=pl.col("image").cv.pipe(score_pipe).sink("native"),
        )
        assert out["contour_scores"][0].to_list() == out["buffer_scores"][0].to_list()

    def test_label_reduce_accepts_array_input(self) -> None:
        """Contour label_reduce accepts fixed-size array image input."""
        contour = _square(0.0, 0.0, 2.0)
        df = pl.DataFrame(
            {
                "preds": [[contour]],
                "image": [[[1.0, 2.0], [3.0, 4.0]]],
            },
            schema={
                "preds": CONTOUR_SET_SCHEMA,
                "image": pl.List(pl.List(pl.Float64)),
            },
        ).cast({"image": pl.Array(pl.Array(pl.Float64, 2), 2)})
        out = df.with_columns(
            score=pl.col("preds").contour.label_reduce(
                image=pl.col("image"), reduction="max", region_mode="bbox"
            )
        )
        assert out["score"][0][0] == pytest.approx(4.0)


@plugin_required
class TestPolarsNativeComposition:
    """Tests pure-Polars composition for curve points."""

    def test_polars_only_threshold_sweep(self) -> None:
        """Detection outputs can be composed into threshold metrics with Polars only."""
        df = pl.DataFrame(
            {
                "image_id": [1, 2],
                "preds": [
                    [_square(0.0, 0.0, 10.0), _square(50.0, 50.0, 10.0)],
                    [_square(0.0, 0.0, 10.0)],
                ],
                "gts": [[_square(0.0, 0.0, 10.0)], [_square(30.0, 30.0, 10.0)]],
                "scores": [[0.95, 0.25], [0.85]],
            },
            schema={
                "image_id": pl.Int64,
                "preds": CONTOUR_SET_SCHEMA,
                "gts": CONTOUR_SET_SCHEMA,
                "scores": pl.List(pl.Float64),
            },
        )

        matched = df.with_columns(
            match=pl.col("preds").contour.match_detections(
                pl.col("gts"), threshold=0.5, scores=pl.col("scores")
            )
        )

        per_det = matched.select(
            "image_id",
            pl.col("match").struct.field("gt_idx"),
            pl.col("scores"),
        ).explode("gt_idx", "scores")

        thresholds = pl.DataFrame({"threshold": [0.2, 0.8]})
        curve = (
            per_det.join(thresholds, how="cross")
            .with_columns(
                keep=pl.col("scores") >= pl.col("threshold"),
                is_tp=pl.col("gt_idx").is_not_null(),
            )
            .filter(pl.col("keep"))
            .group_by("threshold")
            .agg(
                tp=pl.col("is_tp").sum().cast(pl.Int64),
                fp=(~pl.col("is_tp")).sum().cast(pl.Int64),
            )
            .sort("threshold")
        )

        assert curve.height == 2
        assert set(curve.columns) == {"threshold", "tp", "fp"}
