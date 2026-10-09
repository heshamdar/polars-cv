"""Pyramid levels: ``source(level=k)`` decodes a pyramidal TIFF's level ``k``.

A level is IFD 0 plus each later image that is a reduced copy of it — tiled
(or flagged reduced-resolution), smaller in both axes, the same aspect — so the
thumbnail, label and macro images an Aperio SVS interleaves are never levels.
A crop after a levelled source names pixels of that level, and decodes only
its window there as at level 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import OptFlags, Pipeline, numpy_from_struct
from polars_cv._optimize import PASS_NAMES
from tests.conftest import (
    TiffFixture,
    make_image_png,
    plugin_required,
    write_tiled_tiff,
)

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("tifffile")

_ON = OptFlags.all()
_OFF = OptFlags(**{**{n: True for n in PASS_NAMES}, "roi_decode": False})


def _slide(tmp_path: Path, **kwargs: object) -> TiffFixture:
    """A 3-level pyramid with the SVS extras between and after its levels."""
    options: dict[str, object] = {
        "height": 96,
        "width": 128,
        "levels": 3,
        "tile": (16, 16),
        "compression": "lzw",
        "svs_extras": True,
    }
    options.update(kwargs)
    return write_tiled_tiff(tmp_path / "slide.tif", **options)  # type: ignore[arg-type]


def _arrays(
    df: pl.DataFrame, pipe: Pipeline, flags: OptFlags = _ON
) -> list[np.ndarray | None]:
    out = df.select(o=pl.col("b").cv.pipe(pipe).sink("numpy", opt_flags=flags))["o"]
    return [None if v is None else numpy_from_struct(v) for v in out.to_list()]


@plugin_required
class TestLevels:
    """Each level decodes to its own pixels."""

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_a_level_decodes_whole(self, tmp_path: Path, level: int) -> None:
        fx = _slide(tmp_path)
        df = pl.DataFrame({"b": [fx.path.read_bytes()]})
        (got,) = _arrays(df, Pipeline().source("image_bytes", level=level))
        np.testing.assert_array_equal(got, fx.levels[level])

    @pytest.mark.parametrize("level", [0, 1, 2])
    @pytest.mark.parametrize("flags", [_ON, _OFF], ids=["roi", "whole"])
    def test_a_window_is_in_its_levels_pixels(
        self, tmp_path: Path, level: int, flags: OptFlags
    ) -> None:
        fx = _slide(tmp_path)
        df = pl.DataFrame({"b": [fx.path.read_bytes()]})
        pipe = (
            Pipeline()
            .source("image_bytes", level=level)
            .crop(top=3, left=5, height=9, width=11)
        )
        (got,) = _arrays(df, pipe, flags)
        np.testing.assert_array_equal(got, fx.levels[level][3:12, 5:16])

    def test_the_level_may_vary_per_row(self, tmp_path: Path) -> None:
        """``level`` takes an expression: each row decodes its own level."""
        fx = _slide(tmp_path)
        df = pl.DataFrame({"b": [fx.path.read_bytes()] * 3, "k": [2, 0, 1]})
        got = _arrays(df, Pipeline().source("image_bytes", level=pl.col("k")))
        for k, arr in zip([2, 0, 1], got, strict=True):
            np.testing.assert_array_equal(arr, fx.levels[k])

    def test_file_path_and_auto_sources(self, tmp_path: Path) -> None:
        """Paths read a level the same way, through ``file_path`` and ``auto``."""
        fx = _slide(tmp_path, compression="deflate")
        df = pl.DataFrame({"b": [str(fx.path)]})
        for fmt in ("file_path", "auto"):
            (got,) = _arrays(df, Pipeline().source(fmt, level=1))
            np.testing.assert_array_equal(got, fx.levels[1])

    def test_a_non_pyramid_image_has_only_level_zero(self) -> None:
        """A PNG (or a TIFF of one image) is level 0 alone: level 0 is the
        image, any other level is the row's error."""
        df = pl.DataFrame({"b": [make_image_png(8, 8)]})
        (zero,) = _arrays(df, Pipeline().source("image_bytes", level=0))
        assert zero is not None and zero.shape == (8, 8, 3)
        pipe = Pipeline().source("image_bytes", level=1, on_error="null")
        assert _arrays(df, pipe) == [None]
        with pytest.raises(pl.exceptions.ComputeError, match="level 1"):
            _arrays(df, Pipeline().source("image_bytes", level=1))

    def test_a_level_past_the_last_is_the_rows_error(self, tmp_path: Path) -> None:
        """Asking for level 3 of a 3-level pyramid names how many there are."""
        fx = _slide(tmp_path)
        df = pl.DataFrame({"b": [fx.path.read_bytes()] * 2, "k": [1, 3]})
        with pytest.raises(pl.exceptions.ComputeError, match="3 pyramid levels"):
            _arrays(df, Pipeline().source("image_bytes", level=pl.col("k")))
        nulled = _arrays(
            df, Pipeline().source("image_bytes", level=pl.col("k"), on_error="null")
        )
        assert nulled[0] is not None and nulled[1] is None

    def test_a_null_level_follows_the_null_parameter_policy(
        self, tmp_path: Path
    ) -> None:
        fx = _slide(tmp_path)
        df = pl.DataFrame({"b": [fx.path.read_bytes()] * 2, "k": [1, None]})
        pipe = Pipeline().source("image_bytes", level=pl.col("k")).on_null_param("null")
        got = _arrays(df, pipe)
        np.testing.assert_array_equal(got[0], fx.levels[1])
        assert got[1] is None
        with pytest.raises(pl.exceptions.ComputeError):
            _arrays(df, Pipeline().source("image_bytes", level=pl.col("k")))

    def test_the_planned_schema_does_not_depend_on_the_level(self) -> None:
        """A level changes only height and width, which an image source does
        not know at plan time either way."""
        lf = pl.LazyFrame({"b": [b""], "k": [1]})
        schemas = [
            lf.select(
                pl.col("b")
                .cv.pipe(Pipeline().source("image_bytes", dtype="u8", level=lv))
                .sink("list")
            ).collect_schema()
            for lv in (0, 2, pl.col("k"))
        ]
        assert schemas[0] == schemas[1] == schemas[2]


class TestLevelArguments:
    """What the builder refuses outright."""

    def test_level_and_decode_max_size_are_exclusive(self) -> None:
        """Both pick a resolution; asking for both is ambiguous."""
        with pytest.raises(ValueError, match="level"):
            Pipeline().source("image_bytes", level=1, decode_max_size=64)

    def test_level_is_refused_where_there_is_no_image_to_level(self) -> None:
        with pytest.raises(ValueError, match="level"):
            Pipeline().source("blob", level=1)
