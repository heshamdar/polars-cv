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

    @pytest.mark.parametrize("level", ["detection", "image"])
    def test_mann_whitney_levels(self, level: str) -> None:
        # Both MW levels are reachable through the bootstrap (image-level was
        # previously not wired for FROC).
        table = _mixed()
        r = bootstrap_froc_auc(
            table, n_bootstrap=50, seed=2, method="mann_whitney", level=level
        )
        assert len(r.distribution) == 50
        assert r.ci_lower <= r.ci_upper
        assert r.point_estimate == pytest.approx(
            froc_auc(table, method="mann_whitney", level=level).collect().item()
        )

    def test_mann_whitney_bootstrap_inherits_weights(self) -> None:
        """Non-unit metadata weights flow through the MW bootstrap point estimate.

        The bootstrap calls ``froc_auc(method='mann_whitney')``, which is now
        weighted, so (a) the point matches the weighted AUC and (b) the weighted
        AUC differs from the uniform-weight one (the weighting is really applied).
        """
        images = [
            ("a", 0.9, True),
            ("b", 0.8, True),
            ("c", 0.7, False),
            ("d", 0.6, True),
            ("e", 0.5, False),
            ("f", 0.4, True),
        ]
        weights = [3.0, 1.0, 3.0, 1.0, 1.0, 2.0]
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
                COL_WEIGHT: weights,
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
        weighted = DetectionTable.from_matched(det, meta, matching_iou_threshold=0.5)
        uniform = DetectionTable.from_matched(
            det,
            meta.with_columns(pl.lit(1.0).alias(COL_WEIGHT)),
            matching_iou_threshold=0.5,
        )

        w_auc = froc_auc(weighted, method="mann_whitney").collect().item()
        u_auc = froc_auc(uniform, method="mann_whitney").collect().item()
        assert w_auc != pytest.approx(u_auc)  # weighting really changes the value

        r = bootstrap_froc_auc(weighted, n_bootstrap=40, seed=5, method="mann_whitney")
        assert r.point_estimate == pytest.approx(w_auc)


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


class TestEntityLevelBootstrap:
    """sample_col resamples at the entity level and stays seed-reproducible."""

    def _table_with_cases(self) -> DetectionTable:
        # two images per case; resampling by case draws both images together
        images = [
            ("a", 0.9, True),
            ("b", 0.8, True),
            ("c", 0.7, False),
            ("d", 0.6, True),
            ("e", 0.5, False),
            ("f", 0.4, True),
        ]
        table = _table(images)
        meta = table.image_metadata.with_columns(
            case_id=pl.Series("case_id", ["c1", "c1", "c2", "c2", "c3", "c3"]).cast(
                pl.String
            )
        )
        return DetectionTable.from_matched(
            table.detections, meta, matching_iou_threshold=0.5
        )

    def test_entity_bootstrap_reproducible(self) -> None:
        table = self._table_with_cases()
        r1 = bootstrap_froc_auc(table, n_bootstrap=100, seed=5, sample_col="case_id")
        r2 = bootstrap_froc_auc(table, n_bootstrap=100, seed=5, sample_col="case_id")
        assert r1.distribution == r2.distribution
        assert len(r1.distribution) == 100
        assert r1.ci_lower <= r1.point_estimate <= r1.ci_upper
