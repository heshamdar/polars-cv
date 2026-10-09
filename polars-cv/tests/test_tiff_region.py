"""TIFF region decode: a crop after the source reads only the tiles it needs.

Tiled and strip TIFFs decode a leading crop's window chunk by chunk
(``view_buffer::interop::tiff_region``). Every case compares against the truth
the fixture wrote (``write_tiled_tiff``), with the ``roi_decode`` pass on and
off, across codecs, dtypes, channel counts, tile shapes that do not divide the
image, and windows inside one tile, across tiles, at the edges and empty.

That only the needed tiles are read is checked where a user would see it: a
file with one corrupt tile still crops cleanly anywhere away from it, while a
whole-image decode of the same file fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import OptFlags, Pipeline, numpy_from_struct
from polars_cv._optimize import PASS_NAMES
from tests.conftest import TiffFixture, plugin_required, write_tiled_tiff

if TYPE_CHECKING:
    from pathlib import Path

tifffile = pytest.importorskip("tifffile")

_ON = OptFlags.all()
_OFF = OptFlags(**{**{n: True for n in PASS_NAMES}, "roi_decode": False})

_H, _W = 70, 90

#: (top, left, height, width) — None runs to the image's far edge.
_WINDOWS: list[tuple[int, int, int | None, int | None]] = [
    (3, 5, 9, 7),  # inside the first tile
    (10, 12, 40, 60),  # across tiles
    (60, 80, 10, 10),  # the bottom-right (partial) tile
    (0, 0, _H, _W),  # the whole image
    (33, 41, None, None),  # to the far edges
    (20, 20, 0, 5),  # empty
]


def _crop_rows(
    fx: TiffFixture,
    flags: OptFlags,
    windows: list[tuple[int, int, int | None, int | None]] = _WINDOWS,
    on_error: str = "raise",
) -> list[np.ndarray | None]:
    """One row per window, each cropping the same file, as arrays."""
    data = fx.path.read_bytes()
    df = pl.DataFrame(
        {
            "b": [data] * len(windows),
            "t": [w[0] for w in windows],
            "l": [w[1] for w in windows],
            "h": [w[2] if w[2] is not None else _H - w[0] for w in windows],
            "w": [w[3] if w[3] is not None else _W - w[1] for w in windows],
        }
    )
    pipe = (
        Pipeline()
        .source("image_bytes", on_error=on_error)
        .crop(top=pl.col("t"), left=pl.col("l"), height=pl.col("h"), width=pl.col("w"))
    )
    out = df.select(o=pl.col("b").cv.pipe(pipe).sink("numpy", opt_flags=flags))["o"]
    return [None if v is None else numpy_from_struct(v) for v in out.to_list()]


def _truth(
    fx: TiffFixture, window: tuple[int, int, int | None, int | None]
) -> np.ndarray:
    t, left, h, w = window
    level = fx.levels[0]
    return level[
        t : None if h is None else t + h, left : None if w is None else left + w
    ]


_LOSSLESS = [
    (dtype, channels, compression)
    for compression in (None, "lzw", "deflate")
    for dtype, channels in (
        ("u8", 1),
        ("u8", 3),
        ("u8", 4),
        ("u16", 1),
        ("u16", 3),
        ("f32", 1),
    )
]


@plugin_required
class TestLosslessRegions:
    """Lossless TIFFs: every window is exactly the written pixels."""

    @pytest.mark.parametrize(("dtype", "channels", "compression"), _LOSSLESS)
    @pytest.mark.parametrize(
        "tile", [(16, 16), (32, 48), None], ids=["16x16", "32x48", "strips"]
    )
    def test_every_window_is_the_written_pixels(
        self,
        tmp_path: Path,
        dtype: str,
        channels: int,
        compression: str | None,
        tile: tuple[int, int] | None,
    ) -> None:
        fx = write_tiled_tiff(
            tmp_path / "a.tif",
            height=_H,
            width=_W,
            channels=channels,
            dtype=dtype,
            tile=tile,
            compression=compression,
        )
        for flags in (_ON, _OFF):
            for window, got in zip(_WINDOWS, _crop_rows(fx, flags), strict=True):
                want = _truth(fx, window)
                assert got is not None
                assert got.dtype == want.dtype, window
                np.testing.assert_array_equal(got, want, err_msg=str(window))

    def test_bigtiff(self, tmp_path: Path) -> None:
        """A BigTIFF (``II+\\0``) decodes whole and by window."""
        fx = write_tiled_tiff(
            tmp_path / "b.tif", height=_H, width=_W, bigtiff=True, compression="lzw"
        )
        for flags in (_ON, _OFF):
            for window, got in zip(_WINDOWS, _crop_rows(fx, flags), strict=True):
                np.testing.assert_array_equal(
                    got, _truth(fx, window), err_msg=str(window)
                )

    @pytest.mark.parametrize("bigtiff", [False, True], ids=["classic", "bigtiff"])
    @pytest.mark.parametrize(("dtype", "channels"), [("u8", 3), ("u16", 3), ("f32", 1)])
    def test_big_endian(
        self, tmp_path: Path, bigtiff: bool, dtype: str, channels: int
    ) -> None:
        """A big-endian (``MM``) file: its IFDs, tag values and chunk tables
        are read big-endian, and its samples byte-swapped."""
        fx = write_tiled_tiff(
            tmp_path / "m.tif",
            height=_H,
            width=_W,
            channels=channels,
            dtype=dtype,
            byteorder=">",
            bigtiff=bigtiff,
            compression="deflate",
        )
        assert fx.path.read_bytes()[:2] == b"MM"
        for flags in (_ON, _OFF):
            for window, got in zip(_WINDOWS, _crop_rows(fx, flags), strict=True):
                np.testing.assert_array_equal(
                    got, _truth(fx, window), err_msg=str(window)
                )

    def test_pyramid_level_zero(self, tmp_path: Path) -> None:
        """A pyramidal file's first IFD is the image a source decodes."""
        fx = write_tiled_tiff(tmp_path / "p.tif", height=_H, width=_W, levels=3)
        for window, got in zip(_WINDOWS, _crop_rows(fx, _ON), strict=True):
            np.testing.assert_array_equal(got, _truth(fx, window), err_msg=str(window))


@plugin_required
class TestJpegTiles:
    """JPEG-compressed tiles, in both colour layouts TIFF writers use.

    The TIFF Photometric tag decides how the JPEG's components read, as in
    libtiff: YCbCr-coded data tagged YCbCr converts to RGB; RGB-coded data
    tagged RGB is RGB already. The truth is tifffile's (libjpeg's) decode; a
    different IDCT and colour conversion may differ by a few levels.
    """

    @pytest.mark.parametrize(
        ("channels", "photometric"), [(3, "ycbcr"), (3, "rgb"), (1, "ycbcr")]
    )
    def test_windows_match_the_reference_decode(
        self, tmp_path: Path, channels: int, photometric: str
    ) -> None:
        fx = write_tiled_tiff(
            tmp_path / "j.tif",
            height=_H,
            width=_W,
            channels=channels,
            tile=(16, 32),
            compression="jpeg",
            jpeg_photometric=photometric,
        )
        on, off = _crop_rows(fx, _ON), _crop_rows(fx, _OFF)
        for window, got, whole in zip(_WINDOWS, on, off, strict=True):
            want = _truth(fx, window).astype(int)
            assert got is not None and whole is not None
            assert got.shape == want.shape, window
            diff = np.abs(got.astype(int) - want)
            if diff.size:
                assert diff.max() <= 4, (window, diff.max())
                assert diff.mean() <= 0.75, (window, diff.mean())
            # The window and the full decode run the same tile decoder.
            np.testing.assert_array_equal(got, whole, err_msg=str(window))


def _corrupt_last_tile(fx: TiffFixture) -> None:
    """Overwrite the compressed bytes of the image's last tile."""
    with tifffile.TiffFile(fx.path) as tif:
        page = tif.pages[0]
        offset, count = page.dataoffsets[-1], page.databytecounts[-1]
    data = bytearray(fx.path.read_bytes())
    data[offset : offset + count] = b"\xff" * count
    fx.path.write_bytes(bytes(data))


@plugin_required
class TestOnlyTheNeededTilesAreRead:
    """A window decodes from its own tiles; a damaged tile elsewhere is
    never read."""

    @pytest.mark.parametrize("compression", ["lzw", "deflate", "jpeg"])
    def test_a_corrupt_tile_away_from_the_window_is_not_read(
        self, tmp_path: Path, compression: str
    ) -> None:
        fx = write_tiled_tiff(
            tmp_path / "c.tif",
            height=_H,
            width=_W,
            tile=(16, 16),
            compression=compression,
        )
        _corrupt_last_tile(fx)
        clear = [(0, 0, 32, 32), (16, 16, 20, 40)]
        got = _crop_rows(fx, _ON, windows=clear)
        whole = _crop_rows(fx, _OFF, windows=clear, on_error="null")
        assert all(g is not None for g in got)
        if compression != "jpeg":
            for window, g in zip(clear, got, strict=True):
                np.testing.assert_array_equal(g, _truth(fx, window))
        assert whole == [None, None], "a whole-image decode reads the corrupt tile"

    def test_a_window_on_the_corrupt_tile_is_a_row_error(self, tmp_path: Path) -> None:
        """...and a window that does need it fails like any bad image."""
        fx = write_tiled_tiff(
            tmp_path / "c.tif", height=_H, width=_W, tile=(16, 16), compression="lzw"
        )
        _corrupt_last_tile(fx)
        got = _crop_rows(
            fx, _ON, windows=[(0, 0, 8, 8), (64, 80, 6, 10)], on_error="null"
        )
        assert got[0] is not None
        assert got[1] is None


@plugin_required
class TestUnsupportedLayouts:
    """What the tile reader does not handle fails or falls back plainly."""

    def test_an_unsupported_compression_is_a_row_error(self, tmp_path: Path) -> None:
        """Zstandard tiles (not a codec the decoder carries) are the row's
        decode error, nulled under ``on_error="null"``, either way."""
        fx = write_tiled_tiff(tmp_path / "z.tif", height=_H, width=_W)
        data = tifffile.imread(fx.path)
        tifffile.imwrite(fx.path, data, tile=(16, 16), compression="zstd")
        for flags in (_ON, _OFF):
            assert _crop_rows(fx, flags, windows=[(0, 0, 4, 4)], on_error="null") == [
                None
            ]

    def test_planar_separate_samples_are_refused_cleanly(self, tmp_path: Path) -> None:
        """One plane per sample (PlanarConfiguration 2) is not decoded: it is
        the row's decode error, naming the layout, not an engine panic."""
        fx = write_tiled_tiff(tmp_path / "s.tif", height=_H, width=_W)
        tifffile.imwrite(
            fx.path,
            np.moveaxis(fx.levels[0], -1, 0),
            tile=(16, 16),
            photometric="rgb",
            planarconfig="separate",
        )
        for flags in (_ON, _OFF):
            assert _crop_rows(fx, flags, windows=[(0, 0, 4, 4)], on_error="null") == [
                None
            ]
            with pytest.raises(pl.exceptions.ComputeError, match="planar"):
                _crop_rows(fx, flags, windows=[(0, 0, 4, 4)])


@pytest.mark.network
@plugin_required
class TestAperioSample:
    """A real Aperio SVS: JPEG tiles as a scanner wrote them."""

    def test_a_window_matches_tifffile(self, svs_sample: Path) -> None:
        """A window of level 0 matches tifffile's (libjpeg's) decode within
        JPEG tolerance, with the pass on and off alike."""
        with tifffile.TiffFile(svs_sample) as tif:
            ref = tif.pages[0].asarray()
        top, left, size = 300, 400, 256
        want = ref[top : top + size, left : left + size].astype(int)
        df = pl.DataFrame({"b": [svs_sample.read_bytes()]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=top, left=left, height=size, width=size)
        )
        for flags in (_ON, _OFF):
            out = df.select(o=pl.col("b").cv.pipe(pipe).sink("numpy", opt_flags=flags))
            got = numpy_from_struct(out["o"][0]).astype(int)
            assert got.shape == want.shape
            diff = np.abs(got - want)
            assert diff.max() <= 6 and diff.mean() <= 1.0, (diff.max(), diff.mean())
