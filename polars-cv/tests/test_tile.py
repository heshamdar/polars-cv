"""``tile``: one decoded image cut into its patches, ``[N, h, w, C]``.

The in-memory counterpart of ``patch_grid`` + ``explode`` + ``crop``: the image
is decoded once and every patch cut from it. Patch ``i`` must be exactly what
the row-per-patch recipe crops for ``patch_grid``'s cell ``i`` — the two share
one grid (``view_buffer::geometry::grid``), and this file checks that at the
user-facing entry points.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_cv as cv
from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import make_image_png, plugin_required


def _png_frame(height: int = 23, width: int = 31) -> pl.DataFrame:
    return pl.DataFrame({"b": [make_image_png(height, width, channels=3, seed=4)]})


@plugin_required
class TestTile:
    @pytest.mark.parametrize("edge", ["drop", "shift"])
    @pytest.mark.parametrize("stride", [None, (3, 5), (9, 12)])
    def test_patches_are_the_grid_recipes_crops(
        self, edge: str, stride: tuple[int, int] | None
    ) -> None:
        """Patch ``i`` equals the crop the explode recipe takes for cell ``i``."""
        df = _png_frame()
        sh, sw = stride if stride is not None else (None, None)
        pipe = (
            Pipeline()
            .source("image_bytes")
            .tile(height=6, width=8, stride_height=sh, stride_width=sw, edge=edge)
        )
        tiles = numpy_from_struct(
            df.select(pl.col("b").cv.pipe(pipe).sink("numpy"))["b"][0]
        )

        rows = (
            df.with_columns(h=23, w=31)
            .with_columns(
                cell=cv.patch_grid(
                    "h", "w", size=(6, 8), stride=stride if stride else None, edge=edge
                )
            )
            .explode("cell")
            .unnest("cell")
        )
        crop = (
            Pipeline()
            .source("image_bytes")
            .crop(top=pl.col("top"), left=pl.col("left"), height=6, width=8)
        )
        crops = rows.select(pl.col("b").cv.pipe(crop).sink("numpy"))["b"]
        assert tiles.shape == (len(crops), 6, 8, 3)
        for i, c in enumerate(crops):
            np.testing.assert_array_equal(
                tiles[i], numpy_from_struct(c), err_msg=f"patch {i}"
            )

    def test_the_planned_schema_is_the_output(self) -> None:
        lf = _png_frame().lazy()
        q = lf.select(
            pl.col("b")
            .cv.pipe(
                Pipeline().source("image_bytes", dtype="u8").tile(height=4, width=4)
            )
            .sink("list")
        )
        assert q.collect_schema()["b"] == q.collect()["b"].dtype

    def test_the_stride_may_vary_per_row(self) -> None:
        """A per-row stride changes only how many patches a row has."""
        df = pl.concat([_png_frame(16, 16)] * 2).with_columns(s=pl.Series([4, 8]))
        pipe = (
            Pipeline()
            .source("image_bytes")
            .tile(
                height=8, width=8, stride_height=pl.col("s"), stride_width=pl.col("s")
            )
        )
        out = df.select(pl.col("b").cv.pipe(pipe).sink("numpy"))["b"]
        assert [numpy_from_struct(v).shape[0] for v in out] == [9, 4]

    def test_explode_gives_a_row_per_patch(self) -> None:
        """Through the list sink, ``explode`` turns the patches into rows."""
        pipe = Pipeline().source("image_bytes", dtype="u8").tile(height=8, width=8)
        rows = (
            _png_frame(16, 24)
            .select(pl.col("b").cv.pipe(pipe).sink("list"))
            .explode("b")
        )
        assert rows.height == 6
        assert np.asarray(rows["b"][0].to_list()).shape == (8, 8, 3)


class TestTileArguments:
    def test_a_zero_size_or_stride_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Pipeline().source("image_bytes").tile(height=0, width=4)
        with pytest.raises(ValueError, match="positive"):
            Pipeline().source("image_bytes").tile(height=4, width=4, stride_width=0)

    def test_an_image_op_after_tile_is_refused_at_build(self) -> None:
        """Patches are rank 4; an op that reads ``[H, W, C]`` refuses them
        when the pipeline is built, not when a row runs."""
        with pytest.raises(ValueError):
            Pipeline().source("image_bytes").tile(height=4, width=4).grayscale()
