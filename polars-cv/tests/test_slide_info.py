"""``.cv.slide_info()``: a slide's pyramid, read from its header alone.

One struct per row: every pyramid level's size, downsample and tile size (the
levels ``source(level=)`` decodes, from the same scan), and the microns per
pixel when the file records them. Any other image is a single level with no
tiles. The metadata accessors (``.cv.width()`` and kin) read TIFF headers
through the same parser, so the TIFFs only it understands report their size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from tests.conftest import make_image_png, plugin_required, write_tiled_tiff

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("tifffile")

_LEVEL = pl.Struct(
    {
        "level": pl.UInt32,
        "width": pl.UInt32,
        "height": pl.UInt32,
        "downsample": pl.Float64,
        "tile_width": pl.UInt32,
        "tile_height": pl.UInt32,
    }
)
_SLIDE = pl.Struct(
    {"levels": pl.List(_LEVEL), "mpp_x": pl.Float64, "mpp_y": pl.Float64}
)

#: An Aperio ImageDescription, as a ScanScope writes it.
_APERIO = (
    "Aperio Image Library v10.0.51\r\n46920x33014 [0,100 46000x32914] (256x256) "
    "JPEG/RGB Q=30|AppMag = 20|MPP = 0.4990|Left = 25.691574|Top = 23.449873"
)


def _info(df: pl.DataFrame, **kwargs: object) -> list[dict | None]:
    return df.select(i=pl.col("b").cv.slide_info(**kwargs))["i"].to_list()  # type: ignore[arg-type]


@plugin_required
class TestSlideInfo:
    def test_a_pyramid_lists_its_levels_and_skips_the_rest(
        self, tmp_path: Path
    ) -> None:
        """Three levels, their tiles and downsamples; the SVS thumbnail, label
        and macro are not levels."""
        fx = write_tiled_tiff(
            tmp_path / "s.tif",
            height=96,
            width=128,
            levels=3,
            tile=(16, 32),
            svs_extras=True,
            description=_APERIO,
        )
        (info,) = _info(pl.DataFrame({"b": [fx.path.read_bytes()]}))
        assert info is not None
        assert info["levels"] == [
            {
                "level": 0,
                "width": 128,
                "height": 96,
                "downsample": 1.0,
                "tile_width": 32,
                "tile_height": 16,
            },
            {
                "level": 1,
                "width": 64,
                "height": 48,
                "downsample": 2.0,
                "tile_width": 32,
                "tile_height": 16,
            },
            {
                "level": 2,
                "width": 32,
                "height": 24,
                "downsample": 4.0,
                "tile_width": 32,
                "tile_height": 16,
            },
        ]
        assert info["mpp_x"] == pytest.approx(0.499)
        assert info["mpp_y"] == pytest.approx(0.499)

    def test_microns_per_pixel_from_resolution_in_centimetres(
        self, tmp_path: Path
    ) -> None:
        """Without an Aperio header, a resolution in pixels per cm gives it."""
        fx = write_tiled_tiff(
            tmp_path / "r.tif", height=40, width=50, resolution_cm=40000.0
        )
        (info,) = _info(pl.DataFrame({"b": [fx.path.read_bytes()]}))
        assert info is not None
        assert info["mpp_x"] == pytest.approx(0.25)
        assert info["mpp_y"] == pytest.approx(0.25)

    def test_no_recorded_scale_is_null(self, tmp_path: Path) -> None:
        fx = write_tiled_tiff(tmp_path / "n.tif", height=40, width=50, tile=None)
        (info,) = _info(pl.DataFrame({"b": [fx.path.read_bytes()]}))
        assert info is not None
        assert (info["mpp_x"], info["mpp_y"]) == (None, None)
        assert info["levels"] == [
            {
                "level": 0,
                "width": 50,
                "height": 40,
                "downsample": 1.0,
                "tile_width": None,
                "tile_height": None,
            }
        ]

    def test_any_other_image_is_one_untiled_level(self) -> None:
        (info,) = _info(pl.DataFrame({"b": [make_image_png(7, 9)]}))
        assert info == {
            "levels": [
                {
                    "level": 0,
                    "width": 9,
                    "height": 7,
                    "downsample": 1.0,
                    "tile_width": None,
                    "tile_height": None,
                }
            ],
            "mpp_x": None,
            "mpp_y": None,
        }

    def test_paths_read_the_same_as_bytes(self, tmp_path: Path) -> None:
        fx = write_tiled_tiff(
            tmp_path / "p.tif", height=96, width=128, levels=2, svs_extras=True
        )
        by_bytes = _info(pl.DataFrame({"b": [fx.path.read_bytes()]}))
        by_path = _info(pl.DataFrame({"b": [str(fx.path)]}))
        assert by_path == by_bytes

    def test_null_and_unreadable_rows(self, tmp_path: Path) -> None:
        """A null row is null; bytes that are no image are null; a missing
        path fails the query unless ``on_error="null"``."""
        assert _info(
            pl.DataFrame({"b": [None, b"not an image"]}, schema={"b": pl.Binary})
        ) == [None, None]
        missing = pl.DataFrame({"b": [str(tmp_path / "absent.tif")]})
        with pytest.raises(pl.exceptions.ComputeError):
            _info(missing)
        assert _info(missing, on_error="null") == [None]

    def test_the_planned_schema_is_the_output(self) -> None:
        lf = pl.LazyFrame({"b": [make_image_png(4, 4)]})
        q = lf.select(pl.col("b").cv.slide_info())
        assert q.collect_schema()["b"] == _SLIDE
        assert q.collect()["b"].dtype == _SLIDE

    def test_the_levels_are_the_levels_a_source_decodes(self, tmp_path: Path) -> None:
        """Each listed level decodes, through ``source(level=)``, at the size
        listed."""
        from polars_cv import Pipeline, numpy_from_struct

        fx = write_tiled_tiff(
            tmp_path / "s.tif", height=96, width=128, levels=3, svs_extras=True
        )
        df = pl.DataFrame({"b": [fx.path.read_bytes()]})
        (info,) = _info(df)
        assert info is not None
        for lv in info["levels"]:
            pipe = Pipeline().source("image_bytes", level=lv["level"])
            arr = numpy_from_struct(
                df.select(pl.col("b").cv.pipe(pipe).sink("numpy"))["b"][0]
            )
            assert arr.shape[:2] == (lv["height"], lv["width"])


@plugin_required
class TestTiffMetadata:
    """The header accessors understand the TIFFs the decoder does."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"compression": "jpeg"},
            {"bigtiff": True},
            {"compression": "jpeg", "channels": 1},
        ],
        ids=["jpeg-ycbcr", "bigtiff", "jpeg-gray"],
    )
    def test_width_height_channels_dtype(
        self, tmp_path: Path, kwargs: dict[str, object]
    ) -> None:
        fx = write_tiled_tiff(tmp_path / "m.tif", height=40, width=56, **kwargs)  # type: ignore[arg-type]
        channels = fx.levels[0].shape[2]
        out = pl.DataFrame({"b": [fx.path.read_bytes()]}).select(
            w=pl.col("b").cv.width(),
            h=pl.col("b").cv.height(),
            c=pl.col("b").cv.channels(),
            d=pl.col("b").cv.image_dtype(),
        )
        assert out.row(0) == (56, 40, channels, "uint8")
