"""Tests for detection matchers (PreMatchedAdapter and protocol compliance)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_cv.metrics import DetectionTable, PreMatchedAdapter, precision_recall_curve
from polars_cv.metrics._matching._protocol import Matcher
from polars_cv.metrics._types import COL_CLASS_ID, COL_IMAGE_ID, COL_N_GTS

if TYPE_CHECKING:
    pass


class TestPreMatchedAdapter:
    """Tests for the pre-matched pass-through adapter."""

    def test_protocol_compliance(self) -> None:
        """PreMatchedAdapter satisfies the Matcher protocol."""
        assert isinstance(PreMatchedAdapter(), Matcher)

    def test_basic_matching(self) -> None:
        """PreMatchedAdapter wraps flat data into a DetectionTable."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1", "img1", "img2", "img2"],
                "conf": [0.9, 0.7, 0.3, 0.8, 0.5],
                "tp": [True, False, True, True, False],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
        )

        assert isinstance(table, DetectionTable)
        det_df, meta_df = table.collect(engine="streaming")
        assert det_df.height == 5
        assert meta_df.height == 2

    def test_explicit_n_gts(self) -> None:
        """Custom n_gts_col overrides the automatic sum of TPs."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1"],
                "conf": [0.9, 0.7],
                "tp": [True, False],
                "num_gts": [5, 5],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
            n_gts_col="num_gts",
        )
        _, meta_df = table.collect(engine="streaming")
        n_gts_val = meta_df.filter(pl.col(COL_IMAGE_ID) == "img1")[COL_N_GTS].item()
        assert n_gts_val == 5

    def test_with_class_col(self) -> None:
        """Class column is respected."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1", "img1"],
                "conf": [0.9, 0.7, 0.5],
                "tp": [True, False, True],
                "cls": ["cat", "cat", "dog"],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
            class_col="cls",
        )
        det_df, _ = table.collect(engine="streaming")
        classes = det_df[COL_CLASS_ID].unique().to_list()
        assert "cat" in classes
        assert "dog" in classes

    def test_feeds_into_metric(self) -> None:
        """PreMatchedAdapter output can be consumed by precision_recall_curve."""
        data = pl.DataFrame(
            {
                "image": ["img1", "img1", "img1", "img2"],
                "conf": [0.9, 0.7, 0.3, 0.8],
                "tp": [True, False, True, True],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(
            data,
            pred_col="conf",
            gt_col="tp",
            image_id_col="image",
        )
        result = precision_recall_curve(table)
        assert result.curve.height > 0

    def test_auto_image_id(self) -> None:
        """When no image_id_col is given, row index is used."""
        data = pl.DataFrame(
            {
                "conf": [0.9, 0.7],
                "tp": [True, False],
            }
        )
        adapter = PreMatchedAdapter()
        table = adapter.match(data, pred_col="conf", gt_col="tp")
        det_df, _ = table.collect(engine="streaming")
        assert det_df.height == 2
