"""Header-only metadata from path columns, and ``.cv.image_info()``.

``.cv.width()`` and friends took only binary columns, so a path column meant
``.cv.read_bytes()`` first — the whole file read to look at its header. They
now take a path column directly: a local file is read only as far as its
header needs, through the same ``fetch`` stage (and ``allowed_roots`` sandbox)
as every other path read.
"""

from __future__ import annotations

import io

import polars as pl
import pytest
from PIL import Image

from .conftest import make_rect_png, plugin_required


def _jpeg_with_a_late_header(width: int, height: int) -> bytes:
    """A JPEG whose frame header (SOF) sits past the first 64 KiB, behind
    three large comment segments, so a fixed-size prefix read would miss it."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 200, 30)).save(buf, format="JPEG")
    plain = buf.getvalue()
    # Three 60 KB comment (COM) segments straight after start-of-image.
    payload = b"c" * 60_000
    com = b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + payload
    data = plain[:2] + com * 3 + plain[2:]
    assert data.find(b"\xff\xc0") > 64 * 1024, "the fixture's SOF must be late"
    return data


@pytest.fixture
def files(tmp_path) -> dict[str, str]:
    paths = {
        "png": tmp_path / "a.png",
        "jpeg": tmp_path / "late.jpg",
    }
    paths["png"].write_bytes(make_rect_png(height=30, width=50, channels=4))
    paths["jpeg"].write_bytes(_jpeg_with_a_late_header(70, 40))
    return {k: str(v) for k, v in paths.items()}


@plugin_required
class TestMetadataFromPaths:
    def test_paths_agree_with_bytes(self, files) -> None:
        df = pl.DataFrame({"p": [files["png"], None, files["jpeg"]]})
        raw = pl.col("p").cv.read_bytes()
        for method in ("width", "height", "channels", "image_dtype"):
            by_path = df.select(getattr(pl.col("p").cv, method)()).to_series()
            by_bytes = df.select(getattr(raw.cv, method)()).to_series()
            assert by_path.to_list() == by_bytes.to_list(), method
        assert df.select(pl.col("p").cv.width()).to_series().to_list() == [50, None, 70]

    def test_image_info_is_every_field_from_one_read(self, files) -> None:
        df = pl.DataFrame({"p": [files["png"], files["jpeg"]]})
        out = df.select(pl.col("p").cv.image_info())
        assert out.schema["p"] == pl.Struct(
            {
                "width": pl.UInt32,
                "height": pl.UInt32,
                "channels": pl.UInt32,
                "dtype": pl.String,
            }
        )
        assert out.to_series().to_list() == [
            {"width": 50, "height": 30, "channels": 4, "dtype": "uint8"},
            {"width": 70, "height": 40, "channels": 3, "dtype": "uint8"},
        ]

    def test_image_info_over_bytes(self, files) -> None:
        df = pl.DataFrame({"p": [files["png"]]})
        out = df.select(pl.col("p").cv.read_bytes().cv.image_info()).item()
        assert out == {"width": 50, "height": 30, "channels": 4, "dtype": "uint8"}

    def test_an_unreadable_path_raises_or_nulls(self, tmp_path) -> None:
        df = pl.DataFrame({"p": [str(tmp_path / "missing.png")]})
        with pytest.raises(pl.exceptions.ComputeError, match="missing.png"):
            df.select(pl.col("p").cv.width())
        assert df.select(pl.col("p").cv.width(on_error="null")).item() is None

    def test_allowed_roots_sandbox_the_read(self, files, tmp_path) -> None:
        df = pl.DataFrame({"p": [files["png"]]})
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        with pytest.raises(pl.exceptions.ComputeError, match="allowed"):
            df.select(pl.col("p").cv.width(allowed_roots=[str(elsewhere)]))
        assert (
            df.select(pl.col("p").cv.width(allowed_roots=[str(tmp_path)])).item() == 50
        )

    def test_path_options_on_a_binary_column_are_refused(self, files) -> None:
        df = pl.DataFrame({"p": [files["png"]]})
        with pytest.raises(pl.exceptions.ComputeError, match="path column"):
            df.select(pl.col("p").cv.read_bytes().cv.width(on_error="null"))
