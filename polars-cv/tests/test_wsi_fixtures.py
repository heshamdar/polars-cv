"""Self-tests for the tiled / pyramidal TIFF fixtures (``write_tiled_tiff``).

The patch and whole-slide tests compare polars-cv's TIFF decodes against the
truth these fixtures return, so the fixtures themselves are checked first: a
factory that silently wrote strips when asked for tiles, or one level when asked
for three, would make every region-decode test pass against the wrong file.
Pure Python (tifffile reads the files back); no compiled plugin needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.conftest import write_tiled_tiff

if TYPE_CHECKING:
    from pathlib import Path

tifffile = pytest.importorskip("tifffile")


class TestTiledTiffFixture:
    """``write_tiled_tiff`` writes what it is asked to and returns its truth."""

    @pytest.mark.parametrize("compression", [None, "lzw", "deflate"])
    @pytest.mark.parametrize(
        ("dtype", "channels"),
        [("u8", 1), ("u8", 3), ("u8", 4), ("u16", 1), ("u16", 3), ("f32", 1)],
    )
    def test_lossless_truth_is_what_tifffile_reads(
        self, tmp_path: Path, compression: str | None, dtype: str, channels: int
    ) -> None:
        """Lossless fixtures read back bit-exactly as the returned truth."""
        fx = write_tiled_tiff(
            tmp_path / "a.tif",
            height=70,
            width=90,
            channels=channels,
            dtype=dtype,
            tile=(32, 16),
            compression=compression,
        )
        assert fx.lossless
        (truth,) = fx.levels
        assert truth.shape == (70, 90, channels)
        with tifffile.TiffFile(fx.path) as tif:
            page = tif.pages[0]
            assert page.is_tiled
            assert (page.tilelength, page.tilewidth) == (32, 16)
            back = page.asarray()
        np.testing.assert_array_equal(back.reshape(truth.shape), truth)

    def test_pyramid_levels_are_top_level_ifds_halving_in_size(
        self, tmp_path: Path
    ) -> None:
        """Each level is its own top-level IFD, half the previous one, and the
        reduced ones are flagged as reduced-resolution (SubfileType 1)."""
        fx = write_tiled_tiff(
            tmp_path / "p.tif", height=100, width=130, levels=3, compression="lzw"
        )
        assert [lv.shape[:2] for lv in fx.levels] == [(100, 130), (50, 65), (25, 33)]
        with tifffile.TiffFile(fx.path) as tif:
            assert len(tif.pages) == 3
            assert [p.subfiletype for p in tif.pages] == [0, 1, 1]
            for page, truth in zip(tif.pages, fx.levels, strict=True):
                assert page.is_tiled
                np.testing.assert_array_equal(page.asarray(), truth)

    def test_strips_when_no_tile_is_given(self, tmp_path: Path) -> None:
        """``tile=None`` writes a strip TIFF, for the strip region-decode path."""
        fx = write_tiled_tiff(tmp_path / "s.tif", height=40, width=50, tile=None)
        with tifffile.TiffFile(fx.path) as tif:
            assert not tif.pages[0].is_tiled

    def test_bigtiff_when_asked(self, tmp_path: Path) -> None:
        """``bigtiff=True`` writes the BigTIFF header (``II+\\0``)."""
        fx = write_tiled_tiff(tmp_path / "b.tif", height=40, width=50, bigtiff=True)
        assert fx.path.read_bytes()[:4] == b"II+\x00"
        with tifffile.TiffFile(fx.path) as tif:
            assert tif.is_bigtiff

    @pytest.mark.parametrize(
        ("jpeg_photometric", "tag", "component_ids", "adobe"),
        [("ycbcr", "YCBCR", [1, 2, 3], False), ("rgb", "RGB", list(b"RGB"), True)],
    )
    def test_jpeg_tiles_in_both_photometrics(
        self,
        tmp_path: Path,
        jpeg_photometric: str,
        tag: str,
        component_ids: list[int],
        adobe: bool,
    ) -> None:
        """JPEG fixtures come YCbCr-coded and tagged YCbCr (libtiff style) or
        RGB-coded (Adobe APP14 transform 0) and tagged RGB. The tile streams
        are checked, since a decoder's colour handling keys on them; the truth
        is tifffile's own decode, and that decode is close to the source."""
        fx = write_tiled_tiff(
            tmp_path / "j.tif",
            height=64,
            width=96,
            compression="jpeg",
            jpeg_photometric=jpeg_photometric,
            levels=2,
        )
        assert not fx.lossless
        with tifffile.TiffFile(fx.path) as tif:
            for page, truth in zip(tif.pages, fx.levels, strict=True):
                assert page.compression.name == "JPEG"
                assert page.photometric.name == tag
                np.testing.assert_array_equal(page.asarray(), truth)
            first = tif.pages[0]
            tif.filehandle.seek(first.dataoffsets[0])
            stream = tif.filehandle.read(first.databytecounts[0])
        sof = stream.find(b"\xff\xc0")
        assert [stream[sof + 10 + 3 * k] for k in range(3)] == component_ids
        assert (b"Adobe" in stream) is adobe
        source = write_tiled_tiff(tmp_path / "ref.tif", height=64, width=96).levels[0]
        diff = np.abs(fx.levels[0].astype(int) - source.astype(int))
        assert diff.mean() < 3.0

    def test_content_is_seeded_and_tile_distinct(self, tmp_path: Path) -> None:
        """The same seed writes the same pixels; different tiles differ, so a
        misplaced tile cannot compare equal by accident."""
        a = write_tiled_tiff(tmp_path / "1.tif", height=64, width=64, tile=(16, 16))
        b = write_tiled_tiff(tmp_path / "2.tif", height=64, width=64, tile=(16, 16))
        np.testing.assert_array_equal(a.levels[0], b.levels[0])
        img = a.levels[0]
        tiles = {
            img[y : y + 16, x : x + 16].tobytes()
            for y in (0, 16, 32, 48)
            for x in (0, 16, 32, 48)
        }
        assert len(tiles) == 16

    def test_rejects_jpeg_on_non_u8(self, tmp_path: Path) -> None:
        """JPEG holds 8-bit samples only; asking for more is a fixture error,
        not a silently different file."""
        with pytest.raises(ValueError, match="jpeg"):
            write_tiled_tiff(
                tmp_path / "x.tif", height=8, width=8, dtype="u16", compression="jpeg"
            )


@pytest.mark.network
class TestSvsSample:
    """The real Aperio sample is what the whole-slide tests assume it is."""

    def test_is_a_jpeg_tiled_pyramid_with_interleaved_images(
        self, svs_sample: Path
    ) -> None:
        """A tiled JPEG level 0, at least one reduced level, and the
        non-pyramid IFDs (thumbnail) a level scan must skip."""
        with tifffile.TiffFile(svs_sample) as tif:
            pages = list(tif.pages)
            assert pages[0].is_tiled
            assert pages[0].compression.name == "JPEG"
            assert any(not p.is_tiled for p in pages[1:])
