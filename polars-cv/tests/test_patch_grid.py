"""``polars_cv.patch_grid``: the patches that tile each row's image, as rows.

The grid arithmetic is Rust's (``view_buffer::geometry::grid``, exhaustively
swept against a reference there); these tests pin the expression around it —
its schema, nulls, per-row sizes, argument handling — and the recipe it exists
for: ``explode`` the grid, then ``crop`` each row with its fields, which must
cut exactly the pixels numpy slicing does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

import polars_cv as cv
from polars_cv import GridEdge, Pipeline, numpy_from_struct
from tests.conftest import make_image_png, plugin_required

if TYPE_CHECKING:
    pass

_CELL = pl.Struct(
    {
        "row": pl.UInt32,
        "col": pl.UInt32,
        "top": pl.UInt32,
        "left": pl.UInt32,
        "height": pl.UInt32,
        "width": pl.UInt32,
    }
)


def _grid(df: pl.DataFrame, **kwargs: object) -> list[list[dict[str, int]] | None]:
    """``patch_grid`` over ``df``'s ``h``/``w`` columns, as Python values."""
    return df.select(g=cv.patch_grid("h", "w", **kwargs))["g"].to_list()  # type: ignore[arg-type]


class TestArguments:
    """Argument handling that needs no compiled plugin."""

    @pytest.mark.parametrize("size", [(4,), (4, 4, 4), "4", 4.0, True])
    def test_size_must_be_an_int_or_a_pair(self, size: object) -> None:
        """A size is ``n`` or ``(h, w)``; anything else is refused up front."""
        with pytest.raises((TypeError, ValueError), match="size"):
            cv.patch_grid("h", "w", size=size)  # type: ignore[arg-type]

    def test_unknown_edge_is_refused_naming_the_valid_ones(self) -> None:
        """``edge`` is a :class:`GridEdge` or its spelling."""
        with pytest.raises(ValueError, match="drop"):
            cv.patch_grid("h", "w", size=4, edge="wrap")


@plugin_required
class TestPatchGrid:
    """The expression's output."""

    def test_schema_is_a_list_of_crop_shaped_cells(self) -> None:
        """One list of ``{row, col, top, left, height, width}`` per row, every
        field ``UInt32`` — the field names are ``crop``'s keywords."""
        lf = pl.LazyFrame({"h": [8], "w": [8]})
        expr = cv.patch_grid("h", "w", size=4)
        assert lf.select(g=expr).collect_schema()["g"] == pl.List(_CELL)
        assert lf.select(g=expr).collect()["g"].dtype == pl.List(_CELL)

    def test_a_small_grid_exactly(self) -> None:
        """A 6×9 image in 4×4 patches: two whole columns under ``drop``, a
        third aligned to the right edge under ``shift``; rows likewise."""
        df = pl.DataFrame({"h": [6], "w": [9]})
        (drop,) = _grid(df, size=4)
        assert drop is not None
        assert [(c["row"], c["col"], c["top"], c["left"]) for c in drop] == [
            (0, 0, 0, 0),
            (0, 1, 0, 4),
        ]
        assert {(c["height"], c["width"]) for c in drop} == {(4, 4)}
        (shift,) = _grid(df, size=4, edge=GridEdge.SHIFT)
        assert shift is not None
        assert [(c["top"], c["left"]) for c in shift] == [
            (0, 0),
            (0, 4),
            (0, 5),
            (2, 0),
            (2, 4),
            (2, 5),
        ]

    def test_rectangular_size_and_stride(self) -> None:
        """``size``/``stride`` take ``(rows, cols)``; an overlapping stride
        gives overlapping patches."""
        df = pl.DataFrame({"h": [4], "w": [10]})
        (cells,) = _grid(df, size=(4, 6), stride=(4, 2))
        assert cells is not None
        assert [(c["top"], c["left"], c["height"], c["width"]) for c in cells] == [
            (0, 0, 4, 6),
            (0, 2, 4, 6),
            (0, 4, 4, 6),
        ]

    def test_sizes_vary_per_row_and_a_small_image_has_no_patches(self) -> None:
        """Each row's grid follows its own height and width; an image smaller
        than a patch gets an empty list (not null)."""
        df = pl.DataFrame({"h": [8, 4, 3], "w": [4, 8, 8]})
        lengths = [len(g) if g is not None else None for g in _grid(df, size=4)]
        assert lengths == [2, 2, 0]

    def test_a_null_size_is_a_null_row(self) -> None:
        """A row with no height or width has no grid: null, not empty."""
        df = pl.DataFrame({"h": [8, None, 8], "w": [8, 8, None]})
        grids = _grid(df, size=4)
        assert grids[0] is not None
        assert grids[1:] == [None, None]

    def test_literal_sizes_broadcast(self) -> None:
        """A height or width given as an int applies to every row, alone or
        beside a per-row column."""
        df = pl.DataFrame({"h": [4, 8, 12]})
        both = df.with_columns(g=cv.patch_grid(8, 12, size=4))["g"]
        assert both.list.len().to_list() == [6, 6, 6]
        mixed = df.with_columns(g=cv.patch_grid("h", 12, size=4))["g"]
        assert mixed.list.len().to_list() == [3, 6, 9]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [({"size": 0}, "size"), ({"size": 4, "stride": (2, 0)}, "stride")],
    )
    def test_zero_size_or_stride_fails_the_query(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        """No grid has a zero size or stride; Rust refuses it."""
        df = pl.DataFrame({"h": [8], "w": [8]})
        with pytest.raises(pl.exceptions.ComputeError, match=message):
            _grid(df, **kwargs)

    def test_negative_extent_fails_the_query(self) -> None:
        """A negative height is not an image size."""
        df = pl.DataFrame({"h": [-1], "w": [8]})
        with pytest.raises(pl.exceptions.ComputeError, match="height"):
            _grid(df, size=4)

    def test_streaming_and_in_memory_agree(self) -> None:
        """The call is elementwise: both engines give the same rows."""
        lf = pl.LazyFrame({"h": list(range(0, 60, 3)), "w": list(range(60, 0, -3))})
        q = lf.select(g=cv.patch_grid("h", "w", size=5, stride=3, edge="shift"))
        assert q.collect(engine="streaming").equals(q.collect(engine="in-memory"))


@plugin_required
class TestExplodeAndCrop:
    """The recipe: explode the grid, crop each row with its fields."""

    @pytest.mark.parametrize("edge", ["drop", "shift"])
    def test_every_patch_is_the_numpy_slice(self, edge: str) -> None:
        """Each exploded row's crop is exactly ``img[top:top+h, left:left+w]``."""
        height, width = 23, 31
        png = make_image_png(height, width, channels=3, seed=3)
        img = (
            pl.DataFrame({"b": [png]})
            .select(pl.col("b").cv.pipe(Pipeline().source("image_bytes")).sink("numpy"))
            .pipe(lambda d: numpy_from_struct(d.row(0)[0]))
        )
        assert img.shape == (height, width, 3)

        patches = (
            pl.DataFrame({"b": [png], "h": [height], "w": [width]})
            .with_columns(
                cell=cv.patch_grid("h", "w", size=(8, 10), stride=(7, 9), edge=edge)
            )
            .explode("cell")
            .unnest("cell")
        )
        crop = (
            Pipeline()
            .source("image_bytes")
            .crop(
                top=pl.col("top"),
                left=pl.col("left"),
                height=pl.col("height"),
                width=pl.col("width"),
            )
        )
        out = patches.with_columns(p=pl.col("b").cv.pipe(crop).sink("numpy"))
        assert out.height > 0
        for row in out.iter_rows(named=True):
            t, left = row["top"], row["left"]
            want = img[t : t + 8, left : left + 10]
            np.testing.assert_array_equal(numpy_from_struct(row["p"]), want)
