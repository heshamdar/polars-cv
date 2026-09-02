"""D: the ContourMatcher accepts a pre-decoded LazyPipelineExpr.

Passing a pre-decoded ``LazyPipelineExpr`` for ``pred_col`` / ``gt_col`` must
produce the same detections as the equivalent column input, and — because the
matcher now extends the caller's decoded node instead of decoding its own —
lets a segmentation graph and the contour extraction share one decode (CSE),
sunk together in one multi-output plan.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.metrics import ContourMatcher
from tests.conftest import plugin_required


def _dataset() -> pl.DataFrame:
    a = np.zeros((16, 16))
    a[3:9, 3:9] = 0.9
    b = np.zeros((16, 16))
    b[2:7, 8:13] = 0.8
    return pl.DataFrame(
        {
            "image_id": ["a", "b"],
            "pred": [a.tolist(), b.tolist()],
            "gt": [a.tolist(), b.tolist()],
        },
        schema={
            "image_id": pl.String,
            "pred": pl.List(pl.List(pl.Float64)),
            "gt": pl.List(pl.List(pl.Float64)),
        },
    )


def _decoded(col: str):
    """A pre-decoded LazyPipelineExpr equivalent to decoding the column."""
    return pl.col(col).cv.pipe(Pipeline().source("auto"))


@plugin_required
class TestPreDecodedMatcher:
    @pytest.mark.parametrize("auto_resize", [True, False])
    def test_predecoded_pred_matches_column(self, auto_resize: bool) -> None:
        data = _dataset()
        col = ContourMatcher(auto_resize=auto_resize).match(
            data, pred_col="pred", gt_col="gt", image_id_col="image_id"
        )
        expr = ContourMatcher(auto_resize=auto_resize).match(
            data, pred_col=_decoded("pred"), gt_col="gt", image_id_col="image_id"
        )
        c_det, c_meta = col.collect()
        e_det, e_meta = expr.collect()
        assert e_det.sort("image_id", "score").equals(c_det.sort("image_id", "score"))
        assert e_meta.sort("image_id").equals(c_meta.sort("image_id"))

    def test_predecoded_both_operands(self) -> None:
        data = _dataset()
        col = ContourMatcher(auto_resize=False).match(
            data, pred_col="pred", gt_col="gt", image_id_col="image_id"
        )
        expr = ContourMatcher(auto_resize=False).match(
            data,
            pred_col=_decoded("pred"),
            gt_col=_decoded("gt"),
            image_id_col="image_id",
        )
        c_det, _ = col.collect()
        e_det, _ = expr.collect()
        assert e_det.sort("image_id", "score").equals(c_det.sort("image_id", "score"))

    def test_caller_processing_composes_into_the_match(self) -> None:
        """A pre-decoded expr that already carries caller ops feeds the matcher.

        The matcher appends its threshold/extract onto whatever pipeline the
        caller built, so a decoded-and-processed node (here a no-op resize
        standing in for a segmentation stage) composes end to end.
        """
        data = _dataset()
        processed = pl.col("pred").cv.pipe(
            Pipeline().source("auto").resize(height=16, width=16)
        )
        table = ContourMatcher(auto_resize=False).match(
            data, pred_col=processed, gt_col="gt", image_id_col="image_id"
        )
        det, meta = table.collect()
        assert det.height >= 1
        assert meta.height == 2
