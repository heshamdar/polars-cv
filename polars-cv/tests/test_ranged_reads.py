"""A crop (or a level) of a TIFF named by path reads only what it decodes.

A ``file_path`` source used to read the whole file for every row, so one
256-pixel patch of a 2 GB slide read 2 GB. For a TIFF, a crop right after the
source (or a pyramid level above 0) now opens the file and reads its IFDs and
the chunks the window overlaps; every other file is read whole, as before.

Measured where it matters, at the user-facing call: ``_lib._fetch_bytes_read``
counts every byte the plugin's path reads take from a file or a store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

import polars_cv._lib as lib
from polars_cv import OptFlags, Pipeline, numpy_from_struct
from polars_cv._optimize import PASS_NAMES
from tests.conftest import (
    TiffFixture,
    make_image_png,
    plugin_required,
    write_tiled_tiff,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

tifffile = pytest.importorskip("tifffile")

_ON = OptFlags.all()
_OFF = OptFlags(**{**{n: True for n in PASS_NAMES}, "roi_decode": False})


def _measured(run: Callable[[], pl.DataFrame]) -> tuple[pl.DataFrame, int]:
    """``run()`` and the bytes the plugin read from files and stores for it."""
    before = lib._fetch_bytes_read()
    out = run()
    return out, lib._fetch_bytes_read() - before


def _slide(tmp_path: Path, **kwargs: object) -> TiffFixture:
    """A 1024x1024 uncompressed slide in 64-pixel tiles: 3 MiB of pixels."""
    options: dict[str, object] = {"height": 1024, "width": 1024, "tile": (64, 64)}
    options.update(kwargs)
    return write_tiled_tiff(tmp_path / "slide.tif", **options)  # type: ignore[arg-type]


def _crop(df: pl.DataFrame, flags: OptFlags = _ON, **source: object) -> pl.DataFrame:
    pipe = (
        Pipeline()
        .source("file_path", **source)  # type: ignore[arg-type]
        .crop(top=pl.col("t"), left=pl.col("l"), height=64, width=64)
    )
    return df.select(o=pl.col("p").cv.pipe(pipe).sink("numpy", opt_flags=flags))


@plugin_required
class TestLocalFiles:
    def test_a_window_reads_a_small_part_of_the_file(self, tmp_path: Path) -> None:
        """Four 64x64 windows of a 3 MiB slide read under 5% of it — and
        decode exactly the written pixels."""
        fx = _slide(tmp_path)
        size = fx.path.stat().st_size
        origins = [(0, 0), (100, 200), (512, 960), (960, 0)]
        df = pl.DataFrame(
            {
                "p": [str(fx.path)] * 4,
                "t": [o[0] for o in origins],
                "l": [o[1] for o in origins],
            }
        )
        out, read = _measured(lambda: _crop(df))
        assert read < 0.05 * size, (read, size)
        for (t, left), row in zip(origins, out["o"], strict=True):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )

    def test_without_the_pass_the_whole_file_is_read(self, tmp_path: Path) -> None:
        """The measurement is real: a whole-image decode reads the file."""
        fx = _slide(tmp_path)
        df = pl.DataFrame({"p": [str(fx.path)], "t": [0], "l": [0]})
        _, read = _measured(lambda: _crop(df, _OFF))
        assert read >= fx.path.stat().st_size

    def test_a_level_reads_its_own_tiles(self, tmp_path: Path) -> None:
        """Level 2 of a 3-level pyramid (1/16 of the pixels) reads well under
        level 0's share of the file."""
        fx = _slide(tmp_path, levels=3)
        size = fx.path.stat().st_size
        df = pl.DataFrame({"p": [str(fx.path)]})
        pipe = Pipeline().source("file_path", level=2)
        out, read = _measured(
            lambda: df.select(o=pl.col("p").cv.pipe(pipe).sink("numpy"))
        )
        np.testing.assert_array_equal(numpy_from_struct(out["o"][0]), fx.levels[2])
        assert read < 0.15 * size, (read, size)

    def test_any_other_file_is_read_whole(self, tmp_path: Path) -> None:
        """A PNG has no chunks to pick: it is read whole and cropped."""
        path = tmp_path / "a.png"
        path.write_bytes(make_image_png(80, 80, seed=2))
        df = pl.DataFrame({"p": [str(path)], "t": [8], "l": [8]})
        out, read = _measured(lambda: _crop(df))
        assert read >= path.stat().st_size
        assert numpy_from_struct(out["o"][0]).shape == (64, 64, 3)

    def test_the_sandbox_still_applies(self, tmp_path: Path) -> None:
        """``allowed_roots`` refuses a path outside it on the ranged read too."""
        fx = _slide(tmp_path)
        df = pl.DataFrame({"p": [str(fx.path)], "t": [0], "l": [0]})
        elsewhere = str(tmp_path / "elsewhere")
        with pytest.raises(pl.exceptions.ComputeError, match="allowed"):
            _crop(df, allowed_roots=[elsewhere])
        assert _crop(df, allowed_roots=[elsewhere], on_error="null")["o"].to_list() == [
            None
        ]

    def test_a_missing_file_fails_as_before(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"p": [str(tmp_path / "absent.tif")], "t": [0], "l": [0]})
        for flags in (_ON, _OFF):
            with pytest.raises(pl.exceptions.ComputeError, match="absent.tif"):
                _crop(df, flags)
            assert _crop(df, flags, on_error="null")["o"].to_list() == [None]

    def test_a_layout_the_chunk_reader_leaves_falls_back_whole(
        self, tmp_path: Path
    ) -> None:
        """A TIFF layout only the ``tiff`` crate decodes (floats under the
        floating-point predictor) is read whole and cropped, with the same
        pixels as without the pass."""
        path = tmp_path / "fp.tif"
        rng = np.random.default_rng(0)
        data = rng.random((96, 96), dtype=np.float32)
        tifffile.imwrite(path, data, tile=(16, 16), compression="zlib", predictor=3)
        df = pl.DataFrame({"p": [str(path)], "t": [8], "l": [8]})
        on, read = _measured(lambda: _crop(df))
        off = _crop(df, _OFF)
        assert on.equals(off)
        np.testing.assert_array_equal(
            numpy_from_struct(on["o"][0])[..., 0], data[8:72, 8:72]
        )
        assert read >= path.stat().st_size
