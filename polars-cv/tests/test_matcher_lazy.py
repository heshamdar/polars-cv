"""B: the contour/bbox matchers stay lazy — one collect at the caller's boundary.

``match()`` must not eagerly materialize: it returns a ``DetectionTable`` of
uncollected frames whose shared contour-extraction/correspond subplan is cached
so the two derived frames (detections, image_metadata) execute it once under a
single ``collect``.
"""

from __future__ import annotations

import polars as pl

from polars_cv.metrics import ContourMatcher
from polars_cv.metrics._types import DETECTION_SCHEMA, IMAGE_META_SCHEMA
from tests.conftest import plugin_required


def _dataset() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "image_id": ["a", "b"],
            "pred_heatmap": [
                [
                    [1.0 if 2 <= x < 6 and 2 <= y < 6 else 0.0 for x in range(16)]
                    for y in range(16)
                ],
                [[0.0 for _ in range(16)] for _ in range(16)],
            ],
            "gt_mask": [
                [
                    [1.0 if 2 <= x < 6 and 2 <= y < 6 else 0.0 for x in range(16)]
                    for y in range(16)
                ],
                [[0.0 for _ in range(16)] for _ in range(16)],
            ],
            "sample_weight": [1.0, 1.2],
        },
        schema={
            "image_id": pl.String,
            "pred_heatmap": pl.List(pl.List(pl.Float64)),
            "gt_mask": pl.List(pl.List(pl.Float64)),
            "sample_weight": pl.Float64,
        },
    )


def _match(data: pl.DataFrame):
    return ContourMatcher(iou_threshold=0.5).match(
        data.lazy(),
        pred_col="pred_heatmap",
        gt_col="gt_mask",
        image_id_col="image_id",
        weight_col="sample_weight",
    )


@plugin_required
class TestMatcherLazy:
    def test_match_does_not_eagerly_collect(self, monkeypatch) -> None:
        """Building a match plan must not call LazyFrame.collect."""
        calls = {"n": 0}
        orig = pl.LazyFrame.collect

        def spy(self, *args, **kwargs):
            calls["n"] += 1
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", spy)
        table = _match(_dataset())
        assert calls["n"] == 0
        assert isinstance(table.detections, pl.LazyFrame)
        assert isinstance(table.image_metadata, pl.LazyFrame)

    def test_collect_is_single_pass(self, monkeypatch) -> None:
        """DetectionTable.collect runs both frames through one collect_all.

        Collecting both frames in a single query is what lets common-subplan
        elimination execute the shared, cached contour-extraction/correspond
        graph once instead of once per frame (decode-once).
        """
        import polars as pl_mod

        calls = {"n": 0, "n_frames": 0}
        orig = pl_mod.collect_all

        def spy(lazy_frames, *args, **kwargs):
            frames = list(lazy_frames)
            calls["n"] += 1
            calls["n_frames"] = len(frames)
            return orig(frames, *args, **kwargs)

        monkeypatch.setattr(pl_mod, "collect_all", spy)
        table = _match(_dataset())
        det, meta = table.collect()
        assert calls["n"] == 1
        assert calls["n_frames"] == 2

    def test_values_are_correct(self) -> None:
        """The lazy split still produces a well-formed detection table."""
        table = _match(_dataset())
        det, meta = table.collect()
        assert set(DETECTION_SCHEMA).issubset(det.columns)
        assert set(IMAGE_META_SCHEMA).issubset(meta.columns)
        # image 'a' has an overlapping pred+gt → at least one TP detection
        assert det.height >= 1
        assert bool(det["is_tp"].any())

    def test_empty_input_yields_empty_table(self) -> None:
        """An empty input frame flows through lazily to an empty table."""
        empty = _dataset().clear()
        table = _match(empty)
        det, meta = table.collect()
        assert det.height == 0
        assert meta.height == 0
        assert set(DETECTION_SCHEMA).issubset(det.columns)
        assert set(IMAGE_META_SCHEMA).issubset(meta.columns)
