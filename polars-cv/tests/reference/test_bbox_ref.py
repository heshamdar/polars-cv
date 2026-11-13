"""``.bbox.nms`` against OpenCV's ``cv2.dnn.NMSBoxes``.

Both visit boxes by decreasing score and keep one whose IoU with every kept box
is at most the threshold; OpenCV's ``Rect2d`` IoU is the exact area ratio, as
polars-cv's is. Scores are continuous random draws, so no two are equal and
the visit order is the same in both (polars-cv breaks ties by position;
OpenCV's order on ties is not part of this comparison).
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
import pytest

from polars_cv import BBOX_SCHEMA
from tests.conftest import plugin_required


def _boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` boxes as ``[x, y, w, h]`` rows, clustered so many overlap."""
    centres = rng.uniform(0, 60, size=(n, 2))
    sizes = rng.uniform(5, 30, size=(n, 2))
    return np.hstack([centres - sizes / 2, sizes])


@plugin_required
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 1.0])
@pytest.mark.parity(
    ops=["bbox.nms"],
    against="OpenCV `cv2.dnn.NMSBoxes` (Rect2d), random clustered boxes",
    tolerance="exact (kept indices, in order)",
)
def test_nms_matches_opencv(seed: int, threshold: float) -> None:
    rng = np.random.default_rng(seed)
    rows = [rng.integers(0, 40) for _ in range(5)]
    boxes = [_boxes(rng, n) for n in rows]
    # Strictly positive: OpenCV drops a score at or below its score_threshold.
    scores = [rng.uniform(0.01, 1, size=n) for n in rows]
    frame = pl.DataFrame(
        {
            "boxes": [
                [dict(zip(("x", "y", "width", "height"), b, strict=True)) for b in row]
                for row in boxes
            ],
            "scores": [s.tolist() for s in scores],
        },
        schema={"boxes": pl.List(BBOX_SCHEMA), "scores": pl.List(pl.Float64)},
    )
    kept = frame.select(
        pl.col("boxes").bbox.nms(pl.col("scores"), iou_threshold=threshold)
    )["boxes"].to_list()
    for got, row, row_scores in zip(kept, boxes, scores, strict=True):
        want = cv2.dnn.NMSBoxes(
            [tuple(map(float, b)) for b in row],
            row_scores.tolist(),
            score_threshold=0.0,
            nms_threshold=threshold,
        )
        assert got == [int(i) for i in np.asarray(want).ravel()]
    # Not vacuous: at 0.5 and below, the clustered boxes overlap enough that
    # some go (at 0.7 a draw may have no pair that close).
    if threshold <= 0.5:
        assert sum(map(len, kept)) < sum(rows)
