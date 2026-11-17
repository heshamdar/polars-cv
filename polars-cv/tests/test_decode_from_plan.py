"""``source(decode_from_plan=...)``: a JPEG decode shaped by the node's own
leading steps.

``"off"`` (the default) decodes exactly as without the keyword.
``"approximate"`` lets a JPEG decode skip work its node's leading static steps
make unnecessary, at a small documented cost in fidelity:

- a leading ``resize`` (or ``grayscale`` then ``resize``) decodes with IDCT
  scaling at the smallest of 1/8, 1/4, 1/2 that still covers the target on
  both sides; the resize still runs;
- a leading ``grayscale`` decodes only the luma (Y) channel of a YCbCr JPEG,
  scaled too when a following ``resize`` allows (one decode does both).

Every other encoding, and a JPEG the shortcuts cannot reduce, decodes
bit-identically to ``"off"``. The tests that compare ``"approximate"`` with
``"off"`` also assert that the outputs *differ*: a shortcut that never ran
would otherwise pass every closeness bound.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

ORIENTATION = 0x0112


def _photo(width: int, height: int, seed: int = 3) -> np.ndarray:
    """Photo-like RGB pixels: smooth gradients, texture and sensor noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    u, v = xx / width, yy / height
    r = 150 + 80 * np.sin(2.1 * np.pi * u) * np.cos(1.3 * np.pi * v)
    g = 120 + 70 * np.cos(1.7 * np.pi * (u + v))
    b = 100 + 60 * np.sin(3.0 * np.pi * v + 0.5)
    texture = 18 * np.sin(xx / 3.7) * np.sin(yy / 5.3) + 10 * np.sin(
        (xx + 2 * yy) / 11.0
    )
    arr = np.stack([r + texture, g + texture, b + 0.5 * texture], axis=-1)
    arr += rng.normal(0, 4, arr.shape)
    return arr.clip(0, 255).astype(np.uint8)


def _jpeg(
    width: int,
    height: int,
    *,
    mode: str = "RGB",
    tag: int | None = None,
    progressive: bool = False,
) -> bytes:
    img = Image.fromarray(_photo(width, height), "RGB").convert(mode)
    kwargs: dict = {"format": "JPEG", "quality": 90, "progressive": progressive}
    if tag is not None:
        exif = Image.Exif()
        exif[ORIENTATION] = tag
        kwargs["exif"] = exif
    buf = io.BytesIO()
    img.save(buf, **kwargs)
    return buf.getvalue()


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(_photo(width, height), "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _turbo_scaled(data: bytes, mode: str, size: tuple[int, int]) -> bytes:
    """libjpeg-turbo's own IDCT-scaled decode of ``data`` (Pillow's
    ``draft``), as a PNG: what the scaled decode reproduces."""
    img = Image.open(io.BytesIO(data))
    img.draft(mode, size)
    img = img.convert(mode)
    assert img.size == size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run(data: bytes, build, **source) -> np.ndarray:
    pipe = build(Pipeline().source("image_bytes", **source))
    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    out = df.select(out=pl.col("img").cv.pipe(pipe).sink("tensor"))
    return numpy_from_struct(out["out"][0])


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _resize(p: Pipeline) -> Pipeline:
    return p.resize(height=224, width=224)


def _gray(keepdims: bool):
    return lambda p: p.grayscale(keepdims=keepdims)


def _gray_resize(p: Pipeline) -> Pipeline:
    return p.grayscale().resize(height=224, width=224)


class TestBuild:
    def test_rejected_beside_decode_max_size(self) -> None:
        """Two answers to one question: how much of the JPEG to decode."""
        with pytest.raises(ValueError, match="decode_from_plan.*decode_max_size"):
            Pipeline().source(
                "image_bytes", decode_from_plan="approximate", decode_max_size=256
            )

    def test_off_beside_decode_max_size_is_accepted(self) -> None:
        Pipeline().source("image_bytes", decode_from_plan="off", decode_max_size=256)

    def test_unknown_spelling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="decode_from_plan"):
            Pipeline().source("image_bytes", decode_from_plan="exact")

    def test_schema_is_unchanged(self) -> None:
        def schema(**source) -> pl.Schema:
            pipe = _gray_resize(Pipeline().source("image_bytes", **source))
            lf = pl.LazyFrame({"img": [b""]}, schema={"img": pl.Binary})
            return lf.select(
                out=pl.col("img").cv.pipe(pipe).sink("tensor")
            ).collect_schema()

        assert schema(decode_from_plan="approximate") == schema()


@plugin_required
class TestOffAndExactCases:
    @pytest.mark.parametrize("encode", [_jpeg, _png])
    @pytest.mark.parametrize("build", [_resize, _gray(True), _gray_resize])
    def test_off_is_the_default(self, encode, build) -> None:
        data = encode(640, 480)
        assert np.array_equal(
            _run(data, build, decode_from_plan="off"), _run(data, build)
        )

    @pytest.mark.parametrize(
        "build", [_resize, _gray(True), _gray(False), _gray_resize]
    )
    def test_a_png_decodes_exactly(self, build) -> None:
        data = _png(1600, 1200)
        assert np.array_equal(
            _run(data, build, decode_from_plan="approximate"), _run(data, build)
        )

    def test_a_jpeg_with_no_reduction_to_make_decodes_exactly(self) -> None:
        """300 rows cannot halve and still cover 224."""
        data = _jpeg(400, 300)
        assert np.array_equal(
            _run(data, _resize, decode_from_plan="approximate"), _run(data, _resize)
        )

    @pytest.mark.parametrize("keepdims", [True, False])
    def test_a_gray_jpeg_decodes_exactly(self, keepdims: bool) -> None:
        """No chroma to skip: an L-mode JPEG takes the normal path."""
        data = _jpeg(800, 600, mode="L")
        build = _gray(keepdims)
        assert np.array_equal(
            _run(data, build, decode_from_plan="approximate"), _run(data, build)
        )

    def test_a_per_row_resize_takes_no_shortcut(self) -> None:
        data = _jpeg(1600, 1200)
        df = pl.DataFrame({"img": [data], "h": [224], "w": [224]})

        def run(**source) -> np.ndarray:
            pipe = (
                Pipeline()
                .source("image_bytes", **source)
                .resize(height=pl.col("h"), width=pl.col("w"))
            )
            out = df.select(out=pl.col("img").cv.pipe(pipe).sink("tensor"))
            return numpy_from_struct(out["out"][0])

        assert np.array_equal(run(decode_from_plan="approximate"), run())

    def test_a_declared_dtype_takes_no_luma_shortcut(self) -> None:
        """A cast before grayscale would make its luma fractional, which a u8
        Y channel cannot match: the luma shortcut is u8-only."""
        data = _jpeg(800, 600)
        build = _gray(True)
        assert np.array_equal(
            _run(data, build, decode_from_plan="approximate", dtype="f32"),
            _run(data, build, dtype="f32"),
        )


@plugin_required
class TestApproximate:
    def test_scaled_decode_before_resize(self) -> None:
        """1600x1200 -> 224x224 decodes at 1/4 (400x300).

        Measured against "off": PSNR 49.3 dB, mean abs diff 0.62, max 4.
        """
        data = _jpeg(1600, 1200)
        off = _run(data, _resize)
        approx = _run(data, _resize, decode_from_plan="approximate")
        assert approx.shape == off.shape == (224, 224, 3)
        assert approx.dtype == off.dtype
        assert not np.array_equal(approx, off), "the scaled decode did not run"
        assert _psnr(approx, off) >= 30
        assert np.abs(approx.astype(int) - off.astype(int)).mean() <= 3

    @pytest.mark.parametrize("keepdims", [True, False])
    def test_luma_decode_before_grayscale(self, keepdims: bool) -> None:
        """The Y channel against the luma of the RGB decode.

        Measured on 800x600: 1.8% of pixels differ, all by 1 but one pixel
        (of 480,000) by 2, where clamping the RGB decode moved its luma.
        """
        data = _jpeg(800, 600)
        build = _gray(keepdims)
        off = _run(data, build)
        approx = _run(data, build, decode_from_plan="approximate")
        assert approx.shape == off.shape
        assert approx.dtype == off.dtype
        diff = np.abs(approx.astype(int) - off.astype(int))
        assert diff.any(), "the luma decode did not run"
        assert diff.max() <= 2
        assert (diff > 1).mean() <= 1e-4
        assert (diff > 0).mean() <= 0.05

    def test_luma_decode_when_the_resize_cannot_scale(self) -> None:
        """grayscale -> resize on a JPEG no IDCT scale reduces: the luma
        decode still applies (measured: 2.6% of pixels differ, max by 2)."""
        data = _jpeg(400, 300)
        off = _run(data, _gray_resize)
        approx = _run(data, _gray_resize, decode_from_plan="approximate")
        assert approx.shape == off.shape == (224, 224, 1)
        diff = np.abs(approx.astype(int) - off.astype(int))
        assert diff.any(), "the luma decode did not run"
        assert diff.max() <= 2

    def test_grayscale_then_resize_composes(self) -> None:
        """A 1/4-scale luma decode, then the grayscale and resize as written.

        Measured against "off": PSNR 50.2 dB, mean abs diff 0.55, max 3.
        """
        data = _jpeg(1600, 1200)
        off = _run(data, _gray_resize)
        approx = _run(data, _gray_resize, decode_from_plan="approximate")
        assert approx.shape == off.shape == (224, 224, 1)
        assert not np.array_equal(approx, off)
        assert _psnr(approx, off) >= 30
        assert np.abs(approx.astype(int) - off.astype(int)).mean() <= 3

    def test_grayscale_then_resize_decodes_scaled_luma(self) -> None:
        """Luma and scale in one decode: the result is libjpeg-turbo's 1/4
        luma decode, resized (measured: identical). A scaled RGB decode
        converted to gray afterwards differs from it (by up to 2 before
        resize).
        """
        data = _jpeg(1600, 1200)
        approx = _run(data, _gray_resize, decode_from_plan="approximate")
        reference = _run(_turbo_scaled(data, "L", (400, 300)), _resize)
        assert approx.shape == reference.shape == (224, 224, 1)
        assert np.array_equal(approx, reference)

    def test_scaled_decode_of_a_progressive_jpeg(self) -> None:
        """A progressive file scales as a baseline one does: libjpeg-turbo's
        1/4 decode resized, to within zune's colour conversion (measured
        max 1), and close to "off" (PSNR 49.3 dB).
        """
        data = _jpeg(1600, 1200, progressive=True)
        off = _run(data, _resize)
        approx = _run(data, _resize, decode_from_plan="approximate")
        assert approx.shape == off.shape == (224, 224, 3)
        assert not np.array_equal(approx, off), "the scaled decode did not run"
        assert _psnr(approx, off) >= 30
        assert np.abs(approx.astype(int) - off.astype(int)).mean() <= 3
        reference = _run(_turbo_scaled(data, "RGB", (400, 300)), _resize)
        assert np.abs(approx.astype(int) - reference.astype(int)).max() <= 2

    @pytest.mark.parametrize("tag", [3, 6])
    @pytest.mark.parametrize("build", [_resize, _gray(False), _gray_resize])
    def test_orientation_turns_the_shortcut_decode(self, tag: int, build) -> None:
        """A tag that transposes swaps which stored side covers which target."""
        data = _jpeg(1600, 900, tag=tag)
        off = _run(data, build, orientation="exif")
        approx = _run(data, build, orientation="exif", decode_from_plan="approximate")
        assert approx.shape == off.shape
        assert not np.array_equal(approx, off)
        assert _psnr(approx, off) >= 30

    def test_auto_and_file_path_take_it(self, tmp_path) -> None:
        data = _jpeg(1600, 1200)
        path = tmp_path / "a.jpg"
        path.write_bytes(data)
        off = _run(data, _resize)
        for column, source in [
            (pl.Series("c", [data], pl.Binary), {"format": "auto"}),
            (
                pl.Series("c", [str(path)]),
                {"format": "file_path", "allowed_roots": [str(tmp_path)]},
            ),
        ]:
            fmt = source.pop("format")
            pipe = _resize(
                Pipeline().source(fmt, decode_from_plan="approximate", **source)
            )
            out = pl.DataFrame([column]).select(
                out=pl.col("c").cv.pipe(pipe).sink("tensor")
            )
            got = numpy_from_struct(out["out"][0])
            assert got.shape == off.shape
            assert not np.array_equal(got, off), fmt
            assert _psnr(got, off) >= 30
