"""EXIF orientation: ``source(orientation="exif")`` and ``image_info()`` (CR-137).

A camera stores the sensor's pixels and tags how to turn them for display.
The default, ``orientation="stored"``, decodes the pixels as stored (what PIL's
``Image.open`` gives, and what polars-cv always did). ``orientation="exif"``
applies the tag, as ``PIL.ImageOps.exif_transpose`` does. ``image_info()``
reports the tag without decoding, so the displayed size is computable.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from PIL import Image, ImageOps

from polars_cv import Pipeline, numpy_from_struct

from .conftest import plugin_required

pytestmark = plugin_required

ORIENTATION = 0x0112
TAGS = list(range(1, 9))
H, W = 16, 24


def _pixels() -> np.ndarray:
    """An image no flip or transpose maps to itself: every quadrant differs."""
    yy, xx = np.mgrid[0:H, 0:W]
    return np.stack([xx * 10, yy * 15, (xx + 2 * yy) * 5], axis=-1).astype(np.uint8)


def _encode(fmt: str, tag: int | None, exif: bytes | None = None) -> bytes:
    img = Image.fromarray(_pixels(), "RGB")
    kwargs: dict[str, Any] = {}
    if fmt == "TIFF":
        if tag is not None:
            kwargs["tiffinfo"] = {ORIENTATION: tag}
    elif exif is not None:
        kwargs["exif"] = exif
    elif tag is not None:
        data = Image.Exif()
        data[ORIENTATION] = tag
        kwargs["exif"] = data
    if fmt == "WEBP":
        kwargs["lossless"] = True
    if fmt == "JPEG":
        kwargs["quality"] = 95
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _decode(data: bytes, **source: Any) -> np.ndarray:
    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    pipe = Pipeline().source("image_bytes", **source)
    return numpy_from_struct(
        df.select(o=pl.col("img").cv.pipe(pipe).sink("tensor"))["o"][0]
    )


def _displayed(stored: np.ndarray, tag: int | None) -> np.ndarray:
    """PIL's ``exif_transpose`` of exactly these pixels under *tag*."""
    img = Image.fromarray(stored)
    if tag is not None:
        img.getexif()[ORIENTATION] = tag
    return np.asarray(ImageOps.exif_transpose(img))


FORMATS = ["JPEG", "PNG", "WEBP", "TIFF"]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tag", TAGS)
def test_exif_applies_the_tag_as_pil_does(fmt: str, tag: int) -> None:
    data = _encode(fmt, tag)
    stored = _decode(data)
    assert stored.shape == (H, W, 3), "the default decodes the stored layout"
    got = _decode(data, orientation="exif")
    want = _displayed(stored, tag)
    assert got.shape == want.shape
    assert np.array_equal(got, want)
    # And the whole of PIL's own read agrees on the displayed size.
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    assert got.shape[:2] == (pil.height, pil.width)
    if tag != 1:
        assert not np.array_equal(got, stored) or got.shape != stored.shape


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tag", [None, *TAGS])
def test_stored_is_the_default_and_ignores_the_tag(fmt: str, tag: int | None) -> None:
    data = _encode(fmt, tag)
    untagged = _decode(_encode(fmt, None))
    assert np.array_equal(_decode(data), untagged)
    assert np.array_equal(_decode(data, orientation="stored"), untagged)


@pytest.mark.parametrize("fmt", FORMATS)
def test_exif_without_a_tag_changes_nothing(fmt: str) -> None:
    data = _encode(fmt, None)
    assert np.array_equal(_decode(data, orientation="exif"), _decode(data))


@pytest.mark.parametrize(
    "exif",
    [b"Exif\x00\x00garbage", b"Exif\x00\x00II*\x00\xff\xff\xff\xff", b"Exif\x00\x00"],
    ids=["garbage", "bad-offset", "empty"],
)
def test_a_malformed_exif_block_reads_as_no_tag(exif: bytes) -> None:
    """A decode that succeeds today must not fail because EXIF was asked for."""
    data = _encode("JPEG", None, exif=exif)
    assert np.array_equal(_decode(data, orientation="exif"), _decode(data))
    assert _info(data)["orientation"] is None


@pytest.mark.parametrize("tag", [0, 9, 300])
def test_a_tag_outside_one_to_eight_reads_as_no_tag(tag: int) -> None:
    data = _encode("JPEG", tag)
    assert np.array_equal(_decode(data, orientation="exif"), _decode(data))
    assert _info(data)["orientation"] is None


@pytest.mark.parametrize("tag", [3, 6, 8])
def test_orientation_follows_a_scaled_decode(tag: int) -> None:
    """``decode_max_size`` picks the scale from the stored size; the tag then
    turns what was decoded."""
    img = Image.fromarray(np.resize(_pixels(), (400, 800, 3)), "RGB")
    exif = Image.Exif()
    exif[ORIENTATION] = tag
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    data = buf.getvalue()
    scaled = _decode(data, decode_max_size=100)
    assert scaled.shape == (50, 100, 3)
    got = _decode(data, decode_max_size=100, orientation="exif")
    assert np.array_equal(got, _displayed(scaled, tag))


def test_auto_and_file_path_take_it(tmp_path: Path) -> None:
    data = _encode("JPEG", 6)
    want = _decode(data, orientation="exif")
    assert want.shape == (W, H, 3)

    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    auto = Pipeline().source(orientation="exif")
    got = df.select(o=pl.col("img").cv.pipe(auto).sink("tensor"))["o"][0]
    assert np.array_equal(numpy_from_struct(got), want)

    path = tmp_path / "a.jpg"
    path.write_bytes(data)
    paths = pl.DataFrame({"p": [str(path)]})
    fp = Pipeline().source(
        "file_path", orientation="exif", allowed_roots=[str(tmp_path)]
    )
    got = paths.select(o=pl.col("p").cv.pipe(fp).sink("tensor"))["o"][0]
    assert np.array_equal(numpy_from_struct(got), want)


def test_it_applies_to_image_sources_only() -> None:
    with pytest.raises(ValueError, match="'orientation' does not apply"):
        Pipeline().source("blob", orientation="exif")
    with pytest.raises(ValueError, match="orientation"):
        Pipeline().source("image_bytes", orientation="display")
    with pytest.raises((ValueError, TypeError)):
        Pipeline().source("image_bytes", orientation=pl.col("o"))


def test_ops_run_on_the_oriented_view() -> None:
    """The oriented buffer is a view; ops read it where it lies."""
    data = _encode("PNG", 5)
    oriented = _decode(data, orientation="exif")
    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    pipe = (
        Pipeline()
        .source("image_bytes", orientation="exif")
        .crop(top=2, left=3, height=8, width=6)
        .blur(sigma=1.0)
    )
    direct = (
        Pipeline()
        .source("array")
        .crop(top=2, left=3, height=8, width=6)
        .blur(sigma=1.0)
    )
    got = df.select(o=pl.col("img").cv.pipe(pipe).sink("tensor"))["o"][0]
    ref = pl.DataFrame(
        {"a": [oriented]}, schema={"a": pl.Array(pl.UInt8, oriented.shape)}
    ).select(o=pl.col("a").cv.pipe(direct).sink("tensor"))["o"][0]
    assert np.array_equal(numpy_from_struct(got), numpy_from_struct(ref))


# image_info(): the tag without a decode.


def _info(data: bytes) -> dict[str, Any]:
    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    return df.select(i=pl.col("img").cv.image_info())["i"][0]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tag", TAGS)
def test_image_info_reports_the_tag(fmt: str, tag: int) -> None:
    info = _info(_encode(fmt, tag))
    assert info["orientation"] == tag
    # The size stays the stored one, whatever the tag.
    assert (info["width"], info["height"]) == (W, H)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_image_info_orientation_is_null_without_a_tag(fmt: str) -> None:
    assert _info(_encode(fmt, None))["orientation"] is None


def test_image_info_schema_names_the_field() -> None:
    lf = pl.LazyFrame({"img": [b""]}, schema={"img": pl.Binary})
    planned = lf.select(i=pl.col("img").cv.image_info()).collect_schema()["i"]
    assert planned == pl.Struct(
        {
            "width": pl.Int64,
            "height": pl.Int64,
            "channels": pl.Int64,
            "dtype": pl.String,
            "orientation": pl.Int64,
        }
    )
    executed = pl.DataFrame({"img": [_encode("JPEG", 6)]}).select(
        i=pl.col("img").cv.image_info()
    )
    assert executed.schema["i"] == planned


@pytest.mark.parametrize("tag", [None, 3, 6])
def test_image_info_reads_the_tag_from_a_path(tag: int | None, tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(_encode("JPEG", tag))
    df = pl.DataFrame({"p": [str(path)]})
    info = df.select(i=pl.col("p").cv.image_info(allowed_roots=[str(tmp_path)]))["i"][0]
    assert info["orientation"] == tag
    assert (info["width"], info["height"]) == (W, H)


def test_the_displayed_size_is_computable_from_image_info() -> None:
    data = _encode("JPEG", 6)
    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    info = pl.col("img").cv.image_info()
    turned = info.struct.field("orientation").is_between(5, 8).fill_null(False)
    out = df.select(
        w=pl.when(turned)
        .then(info.struct.field("height"))
        .otherwise(info.struct.field("width")),
        h=pl.when(turned)
        .then(info.struct.field("width"))
        .otherwise(info.struct.field("height")),
    )
    shown = _decode(data, orientation="exif")
    assert (out["h"][0], out["w"][0]) == shown.shape[:2]
