"""C: vectorized, seed-reproducible bootstrap for FROC / LROC AUC.

bootstrap_froc_auc / bootstrap_lroc_auc run every replicate in one lazy plan
(froc_auc / lroc_auc keyed by ``bootstrap_id``) and derive their samples from a
seeded generator, so the confidence interval is reproducible for a given seed —
fixing the non-determinism of the sequential path.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.metrics import DetectionTable, froc_auc, lroc_auc
from polars_cv.metrics._bootstrap import bootstrap_froc_auc, bootstrap_lroc_auc
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
)


def _table(images: list[tuple[str, float, bool]]) -> DetectionTable:
    """One detection per image: (image_id, score, is_tp). gt_label = has a GT."""
    det = pl.DataFrame(
        {
            COL_IMAGE_ID: [i for i, _s, _t in images],
            COL_CLASS_ID: ["__all__"] * len(images),
            COL_SCORE: [s for _i, s, _t in images],
            COL_IS_TP: [t for _i, _s, t in images],
            COL_GT_IDX: [0 if t else None for _i, _s, t in images],
            COL_IOU: [0.7 if t else 0.0 for _i, _s, t in images],
            COL_DET_IDX: list(range(len(images))),
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
    meta = pl.DataFrame(
        {
            COL_IMAGE_ID: [i for i, _s, _t in images],
            COL_CLASS_ID: ["__all__"] * len(images),
            COL_N_GTS: [1 if t else 0 for _i, _s, t in images],
            COL_WEIGHT: [1.0] * len(images),
            COL_GT_LABEL: [t for _i, _s, t in images],
        },
        schema={
            COL_IMAGE_ID: pl.String,
            COL_CLASS_ID: pl.String,
            COL_N_GTS: pl.Int64,
            COL_WEIGHT: pl.Float64,
            COL_GT_LABEL: pl.Boolean,
        },
    )
    return DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)


def _mixed() -> DetectionTable:
    return _table(
        [
            ("a", 0.9, True),
            ("b", 0.8, True),
            ("c", 0.7, False),
            ("d", 0.6, True),
            ("e", 0.5, False),
            ("f", 0.4, True),
        ]
    )


class TestBootstrapFrocReproducible:
    def test_same_seed_is_bit_identical(self) -> None:
        table = _mixed()
        r1 = bootstrap_froc_auc(table, n_bootstrap=200, seed=123)
        r2 = bootstrap_froc_auc(table, n_bootstrap=200, seed=123)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper
        assert r1.distribution == r2.distribution

    def test_distribution_length_and_point_in_range(self) -> None:
        table = _mixed()
        r = bootstrap_froc_auc(table, n_bootstrap=200, seed=1)
        assert len(r.distribution) == 200
        assert r.ci_lower <= r.point_estimate <= r.ci_upper
        assert r.point_estimate == pytest.approx(froc_auc(table).collect().item())

    def test_identical_images_give_constant_distribution(self) -> None:
        # Every image identical → any resample yields the same normalized curve,
        # so every replicate's AUC equals the point estimate.
        table = _table([(chr(ord("a") + k), 0.5, True) for k in range(8)])
        point = froc_auc(table).collect().item()
        r = bootstrap_froc_auc(table, n_bootstrap=64, seed=7)
        assert r.ci_lower == pytest.approx(point, abs=1e-9)
        assert r.ci_upper == pytest.approx(point, abs=1e-9)


class TestBootstrapLrocReproducible:
    def test_same_seed_is_bit_identical(self) -> None:
        table = _mixed()
        r1 = bootstrap_lroc_auc(table, n_bootstrap=200, seed=42)
        r2 = bootstrap_lroc_auc(table, n_bootstrap=200, seed=42)
        assert r1.distribution == r2.distribution

    def test_point_matches_lroc_auc(self) -> None:
        table = _mixed()
        r = bootstrap_lroc_auc(table, n_bootstrap=100, seed=3)
        assert r.point_estimate == pytest.approx(lroc_auc(table).collect().item())
        assert len(r.distribution) == 100
