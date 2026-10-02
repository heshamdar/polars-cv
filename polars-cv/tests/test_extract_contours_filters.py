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


def _square(x0: float, size: float) -> dict:
    x0, size = float(x0), float(size)
    ring = [
        {"x": x0, "y": 0.0},
        {"x": x0 + size, "y": 0.0},
        {"x": x0 + size, "y": size},
        {"x": x0, "y": size},
    ]
    return {"exterior": ring, "holes": [], "is_closed": True}


@plugin_required
class TestLargest:
    """Keep the k largest contours of a set, by area, as a set."""

    def test_in_the_pipeline(self) -> None:
        df = pl.DataFrame({"img": [_blobs(40)]})
        assert _areas(df, _BASE.extract_contours().largest()) == [[100.0]]
        assert _areas(df, _BASE.extract_contours().largest(k=2)) == [[16.0, 100.0]]

    def test_largest_first_and_ties_in_input_order(self) -> None:
        from polars_cv import CONTOUR_SET_SCHEMA

        df = pl.DataFrame(
            {"c": [[_square(0, 1), _square(10, 3), _square(20, 2), _square(30, 3)]]},
            schema={"c": CONTOUR_SET_SCHEMA},
        )
        out = df.select(pl.col("c").contour.largest(k=3)).item()
        assert [c["exterior"][0]["x"] for c in out] == [10.0, 30.0, 20.0]

    def test_on_a_column_of_sets_and_of_single_contours(self) -> None:
        from polars_cv import CONTOUR_SCHEMA, CONTOUR_SET_SCHEMA

        sets = pl.DataFrame(
            {"c": [[_square(0, 1), _square(10, 3)], [], None]},
            schema={"c": CONTOUR_SET_SCHEMA},
        )
        got = sets.select(pl.col("c").contour.largest().contour.area()).to_series()
        assert got.to_list() == [[9.0], [], None]
        # A lone contour is a set of one, so the result is always a set.
        single = pl.DataFrame({"c": [_square(0, 2)]}, schema={"c": CONTOUR_SCHEMA})
        out = single.select(pl.col("c").contour.largest())
        assert out.schema["c"] == pl.List(CONTOUR_SCHEMA)

    def test_a_per_row_k(self) -> None:
        df = pl.DataFrame({"img": [_blobs(40), _blobs(40)], "k": [1, 3]})
        pipe = _BASE.extract_contours().largest(k=pl.col("k"))
        assert _areas(df, pipe) == [[100.0], [4.0, 16.0, 100.0]]

    def test_k_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="k"):
            _BASE.extract_contours().largest(k=0)
