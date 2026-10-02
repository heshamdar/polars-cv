"""Filtering extracted contours by size: relative ``min_area`` and ``largest``."""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline

from .conftest import plugin_required


def _png(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _blobs(size: int) -> bytes:
    """A ``size`` x ``size`` mask with a 10x10, a 4x4 and a 2x2 blob, apart."""
    mask = np.zeros((size, size), np.uint8)
    mask[2:12, 2:12] = 255  # area 100
    mask[20:24, 20:24] = 255  # area 16
    mask[30:32, 2:4] = 255  # area 4
    return _png(mask)


def _areas(df: pl.DataFrame, pipe: Pipeline) -> list[list[float]]:
    out = df.select(pl.col("img").cv.pipe(pipe).sink("native").contour.area())
    return [sorted(row) for row in out.to_series().to_list()]


_BASE = Pipeline().source("image_bytes").grayscale().threshold(128)


@plugin_required
class TestMinAreaFraction:
    """``min_area_fraction`` is a fraction of the image's own H x W, resolved
    per image as it runs — no prior read of the size."""

    def test_the_threshold_follows_each_images_size(self) -> None:
        # 15 / 1600 px ~ 0.0094 keeps the 16 px blob in a 40x40 image, but
        # 15 / 6400 px is not: the same fraction of an 80x80 image is 60 px.
        df = pl.DataFrame({"img": [_blobs(40), _blobs(80)]})
        got = _areas(df, _BASE.extract_contours(min_area_fraction=15 / 1600))
        assert got == [[16.0, 100.0], [100.0]]

    def test_both_thresholds_apply(self) -> None:
        df = pl.DataFrame({"img": [_blobs(40)]})
        pipe = _BASE.extract_contours(min_area=50.0, min_area_fraction=1 / 1600)
        assert _areas(df, pipe) == [[100.0]]

    def test_a_per_row_fraction(self) -> None:
        df = pl.DataFrame({"img": [_blobs(40), _blobs(40)], "f": [1 / 1600, 50 / 1600]})
        pipe = _BASE.extract_contours(min_area_fraction=pl.col("f"))
        assert _areas(df, pipe) == [[4.0, 16.0, 100.0], [100.0]]

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_a_fraction_outside_zero_one_is_refused(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="min_area_fraction"):
            _BASE.extract_contours(min_area_fraction=fraction)

    def test_a_per_row_fraction_outside_zero_one_is_refused(self) -> None:
        df = pl.DataFrame({"img": [_blobs(40)], "f": [2.0]})
        pipe = _BASE.extract_contours(min_area_fraction=pl.col("f"))
        with pytest.raises(pl.exceptions.ComputeError, match="min_area_fraction"):
            df.select(pl.col("img").cv.pipe(pipe).sink("native"))
