"""Patch grids: cut each row's image into patches, one row per patch.

:func:`patch_grid` lists the patches that tile each row's image. The plugin
stays one-row-in, one-row-out; Polars' ``explode`` turns the patches into rows,
and each row's fields are exactly ``crop``'s keywords::

    patches = (
        df.with_columns(cell=cv.patch_grid("height", "width", size=256))
        .explode("cell")
        .unnest("cell")
    )
    pipe = Pipeline().source("file_path").crop(
        top=pl.col("top"), left=pl.col("left"), height=256, width=256
    )
    patches.with_columns(patch=pl.col("path").cv.pipe(pipe).sink("numpy"))

The grid itself is computed in Rust (``view_buffer::geometry::grid``), the one
definition of where patch ``i`` lies.
"""

from __future__ import annotations

from typing import Union

import polars as pl

from polars_cv import _plugin
from polars_cv._ops_generated import GridEdge

__all__ = ["patch_grid"]

#: A height or width: a column name, an expression, or one int for every row.
ExtentInput = Union[str, int, pl.Expr]


def _pair(value: int | tuple[int, int], name: str) -> list[int]:
    """``value`` as ``[rows, cols]``: an int applies to both axes."""
    if isinstance(value, int) and not isinstance(value, bool):
        return [value, value]
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return list(value)
    msg = f"patch_grid(): {name} must be an int or a (rows, cols) pair of ints, got {value!r}"
    raise TypeError(msg)


def _extent(value: ExtentInput) -> pl.Expr:
    """A height/width argument as an expression."""
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, str):
        return pl.col(value)
    return pl.lit(value, dtype=pl.UInt32)


def patch_grid(
    height: ExtentInput,
    width: ExtentInput,
    *,
    size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    edge: GridEdge | str = GridEdge.DROP,
) -> pl.Expr:
    """The patches that tile each row's ``height × width`` image.

    Patches are listed row-major. Every patch is whole and inside the image:
    ``edge`` decides what happens to a remainder too small for one, and an
    image smaller than a patch has an empty list. A null height or width gives
    a null row.

    Args:
        height: Image height per row: a column name, an expression (e.g.
            ``pl.col("path").cv.height()``), or one int for every row.
        width: Image width per row, likewise.
        size: Patch size, ``n`` or ``(rows, cols)``.
        stride: Distance between patch origins, ``n`` or ``(rows, cols)``;
            defaults to ``size`` (patches that touch without overlapping). A
            smaller stride overlaps patches, a larger one leaves gaps.
        edge: :class:`GridEdge` — ``"drop"`` leaves the remainder uncovered;
            ``"shift"`` adds one patch aligned to the far edge.

    Returns:
        A ``List(Struct{row, col, top, left, height, width})`` expression, all
        ``UInt32``: grid position, top-left pixel and patch size. Explode it
        for one row per patch; the fields are ``crop``'s keywords.

    Example:
        ```python
        >>> import polars as pl
        >>> import polars_cv as cv
        >>> df = pl.DataFrame({"h": [6], "w": [9]})
        >>> cells = df.select(c=cv.patch_grid("h", "w", size=4, edge="shift"))
        >>> cells.explode("c").unnest("c").select("top", "left").rows()
        [(0, 0), (0, 4), (0, 5), (2, 0), (2, 4), (2, 5)]
        ```
    """
    sizes = _pair(size, "size")
    strides = sizes if stride is None else _pair(stride, "stride")
    try:
        edge_name = GridEdge(edge).value
    except ValueError:
        valid = [e.value for e in GridEdge]
        msg = f"patch_grid(): unknown edge {edge!r} (expected one of {valid})"
        raise ValueError(msg) from None
    return _plugin.call(
        "patch_grid",
        args=[_extent(height), _extent(width)],
        kwargs={"size": sizes, "stride": strides, "edge": edge_name},
    )
