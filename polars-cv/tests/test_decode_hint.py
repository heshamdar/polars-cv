"""
Tests for the explicit decode-scale assertion (`source(decode_max_size=...)`).

JPEG sources honor the assertion via IDCT scaling (1/8, 1/4, 1/2): the
decoder skips work and memory while guaranteeing the long side never drops
below min(decode_max_size, original). Other formats ignore the assertion and
decode at full size. The assertion is explicit opt-in because a scaled
decode followed by a resize is not bit-identical to a full decode followed
by the same resize.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _jpeg(width: int, height: int, quality: int = 92) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(7)
    # Smooth gradient + mild noise: realistic JPEG content, stable under
    # compression.
    yy, xx = np.mgrid[0:height, 0:width]
    base = (xx * 255 / max(width - 1, 1) + yy * 64 / max(height - 1, 1)) % 255
    arr = np.stack([base, base * 0.7, base * 0.4], axis=-1)
    arr = (arr + rng.normal(0, 2, arr.shape)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _gray_jpeg(width: int, height: int) -> bytes:
    from PIL import Image

    arr = np.linspace(0, 255, width * height).reshape(height, width).astype(np.uint8)
    img = Image.fromarray(arr, "L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _png(width: int, height: int) -> bytes:
    from PIL import Image

    arr = np.full((height, width, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDecodeMaxSizeValidation:
    def test_rejected_for_non_image_sources(self) -> None:
        with pytest.raises(ValueError, match="decode_max_size only applies"):
            Pipeline().source("blob", decode_max_size=100)

    def test_rejected_for_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            Pipeline().source("image_bytes", decode_max_size=0)

    def test_serialized_into_spec(self) -> None:
        import json

        pipe = Pipeline().source("image_bytes", decode_max_size=256)
        spec = json.loads(pipe._to_json())
        assert spec["source"]["decode_max_size"] == 256


@plugin_required
class TestJpegScaledDecode:
    def test_decodes_at_reduced_scale(self) -> None:
        # 800x600 with a 100px assertion: 1/8 scale -> 100x75 (long side
        # stays >= 100).
        df = pl.DataFrame({"img": [_jpeg(800, 600)]})
        pipe = Pipeline().source("image_bytes", decode_max_size=100)
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(out["out"][0])
        assert arr.shape == (75, 100, 3)
        assert arr.dtype == np.uint8

    def test_long_side_never_below_assertion(self) -> None:
        # 1/8 would give 62x50 (< 100 on the long side); the decoder must
        # pick 1/4 -> 125x100.
        df = pl.DataFrame({"img": [_jpeg(500, 400)]})
        pipe = Pipeline().source("image_bytes", decode_max_size=100)
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(out["out"][0])
        assert max(arr.shape[0], arr.shape[1]) >= 100
        assert arr.shape == (100, 125, 3)

    def test_assertion_larger_than_image_is_full_decode(self) -> None:
        df = pl.DataFrame({"img": [_jpeg(80, 60)]})
        pipe = Pipeline().source("image_bytes", decode_max_size=1024)
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert numpy_from_struct(out["out"][0]).shape == (60, 80, 3)

    def test_grayscale_jpeg_shape_matches_full_decode(self) -> None:
        jpg = _gray_jpeg(400, 320)
        df = pl.DataFrame({"img": [jpg]})
        full = df.with_columns(
            out=pl.col("img").cv.pipe(Pipeline().source("image_bytes")).sink("numpy")
        )
        scaled = df.with_columns(
            out=pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes", decode_max_size=50))
            .sink("numpy")
        )
        assert numpy_from_struct(full["out"][0]).shape == (320, 400, 1)
        assert numpy_from_struct(scaled["out"][0]).shape == (40, 50, 1)

    def test_png_ignores_assertion(self) -> None:
        df = pl.DataFrame({"img": [_png(200, 150)]})
        pipe = Pipeline().source("image_bytes", decode_max_size=50)
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert numpy_from_struct(out["out"][0]).shape == (150, 200, 3)

    def test_thumbnail_pipeline_close_to_full_decode(self) -> None:
        # The canonical use: scaled decode + resize must be visually
        # equivalent (not bit-identical) to full decode + resize.
        jpg = _jpeg(800, 600)
        df = pl.DataFrame({"img": [jpg]})

        def thumb(**source_kwargs):
            pipe = (
                Pipeline()
                .source("image_bytes", **source_kwargs)
                .resize(height=48, width=64)
            )
            out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            return numpy_from_struct(out["out"][0]).astype(np.float64)

        full = thumb()
        fast = thumb(decode_max_size=64)
        assert fast.shape == full.shape
        mean_abs_diff = np.abs(full - fast).mean()
        assert mean_abs_diff < 4.0, f"thumbnails diverged: {mean_abs_diff=}"

    def test_dtype_assertion_still_applies(self) -> None:
        # source(dtype=...) casting composes with the scaled decode.
        df = pl.DataFrame({"img": [_jpeg(800, 600)]})
        pipe = Pipeline().source("image_bytes", dtype="f32", decode_max_size=100)
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(out["out"][0])
        assert arr.dtype == np.float32
        assert arr.shape == (75, 100, 3)

    def test_file_path_source_honors_assertion(self, tmp_path) -> None:
        p = tmp_path / "big.jpg"
        p.write_bytes(_jpeg(800, 600))
        df = pl.DataFrame({"paths": [str(p)]})
        pipe = Pipeline().source("file_path", decode_max_size=100)
        out = df.with_columns(out=pl.col("paths").cv.pipe(pipe).sink("numpy"))
        assert numpy_from_struct(out["out"][0]).shape == (75, 100, 3)
