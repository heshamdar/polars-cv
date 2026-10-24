"""
Reading ``sink("numpy")`` / ``sink("ndarray")`` output back as NumPy arrays.

``numpy_from_struct`` reads one row, ``numpy_from_column`` a whole column
without copying; both build the array through :func:`_view`, the one place a
sink struct becomes memory. Re-exported from :mod:`polars_cv`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ._dtype_names import SINK_NUMPY_NAMES
from .extension_types import NUMPY_OUTPUT_SCHEMA, NdArrayType

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def numpy_from_struct(
    row: dict[str, object] | pl.Series,
    *,
    copy: bool = True,
) -> "np.ndarray":
    """
    Convert numpy sink output struct to a NumPy array.

    Reads both ``sink("numpy")`` and ``sink("ndarray")`` output: a row of either
    is the same dict, and a single-row Series may be the plain struct or the
    ``polars_cv.ndarray`` type over it.

    To read a whole column, use :func:`numpy_from_column`, which views each
    row in the column's own memory instead of copying it through a ``dict``.

    Args:
        row: Struct value from output column.
        copy: Whether to copy data (default True). If False, returns a view.
    """
    import numpy as np

    if row is None:
        # A null row of the numpy/torch/ndarray sink is a null value.
        msg = "row is null (a null input or a failed row); it has no array"
        raise ValueError(msg)

    # Extract fields from struct
    if isinstance(row, dict):
        data = row.get("data")
        dtype_str = row.get("dtype")
        shape_list = row.get("shape")
        strides_list = row.get("strides")
        offset = row.get("offset", 0)
    elif isinstance(row, pl.Series):
        # A `sink("ndarray")` column is the same struct under a type tag.
        if isinstance(row.dtype, NdArrayType):
            row = row.ext.storage()
        # Single-row Series from struct indexing
        if row.dtype == pl.Struct:
            struct_data = row.struct.unnest()
            data = struct_data["data"][0]
            dtype_str = struct_data["dtype"][0]
            shape_list = struct_data["shape"][0]
            strides_list = (
                struct_data["strides"][0] if "strides" in struct_data.columns else None
            )
            offset = struct_data["offset"][0] if "offset" in struct_data.columns else 0
        else:
            msg = f"Expected Struct Series, got {row.dtype}"
            raise ValueError(msg)
    else:
        # Assume it's a struct value that can be accessed like a dict
        try:
            data = row["data"]
            dtype_str = row["dtype"]
            shape_list = row["shape"]
            strides_list = row.get("strides") if hasattr(row, "get") else None
            offset = row.get("offset", 0) if hasattr(row, "get") else 0
        except (TypeError, KeyError) as e:
            msg = f"Cannot extract struct fields from {type(row)}: {e}"
            raise ValueError(msg) from e

    # Validate required fields
    if data is None:
        msg = "Struct field 'data' is null"
        raise ValueError(msg)
    if dtype_str is None:
        msg = "Struct field 'dtype' is null"
        raise ValueError(msg)
    if shape_list is None:
        msg = "Struct field 'shape' is null"
        raise ValueError(msg)

    # Convert shape to tuple
    if isinstance(shape_list, pl.Series):
        shape = tuple(int(x) for x in shape_list.to_list())
    else:
        shape = tuple(int(x) for x in shape_list)  # ty: ignore[not-iterable]

    # Convert strides to tuple (if present)
    strides: tuple[int, ...] | None = None
    if strides_list is not None:
        if isinstance(strides_list, pl.Series):
            strides = tuple(int(x) for x in strides_list.to_list())
        else:
            strides = tuple(int(x) for x in strides_list)  # ty: ignore[not-iterable]

    # Convert offset
    if offset is None:
        offset = 0
    else:
        offset = int(offset)  # ty: ignore[invalid-argument-type]

    if strides is None:
        # No stride metadata (older/dict callers): C-contiguous.
        itemsize = _checked_dtype(str(dtype_str)).itemsize
        c_strides: list[int] = []
        step = itemsize
        for n in reversed(shape):
            c_strides.append(step)
            step *= n
        strides = tuple(reversed(c_strides))

    # The row's bytes as a uint8 array: it owns (references) `data` and
    # names its address, and `_view` reads the row through it, checked.
    backing = np.frombuffer(_as_buffer(data), dtype=np.uint8)
    return _view(
        backing,
        backing.__array_interface__["data"][0],
        backing.nbytes,
        str(dtype_str),
        shape,
        strides,
        offset,
        copy=copy,
    )


def _checked_dtype(dtype_name: str) -> np.dtype:
    """The numpy dtype a sink struct names, refusing anything else.

    The sink writes `DType::numpy_name()` into the struct, so those names are
    exactly what can legitimately arrive. Generated from `dtype_table!` rather
    than listed here: the hand-written list this replaced admitted numpy's
    *character codes* too, which meant `"u8"` (numpy uint64) sat in the same
    set as this project's `"u8"` (uint8) — so a caller hand-building a struct
    with `dtype="u8"` got a uint64 reinterpretation of the bytes, silently.
    """
    import numpy as np

    if dtype_name not in SINK_NUMPY_NAMES:
        msg = f"Unsupported dtype '{dtype_name}'. Allowed: {sorted(SINK_NUMPY_NAMES)}"
        raise ValueError(msg)
    return np.dtype(dtype_name)


def _view(
    owner: object,
    address: int,
    nbytes: int,
    dtype_name: str,
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int,
    *,
    copy: bool,
) -> np.ndarray:
    """A numpy array over `nbytes` of memory at `address`, as a sink struct
    describes it: `shape`, byte `strides` (negative for a flip) and a byte
    `offset` to element zero.

    **The one way a sink struct becomes an array**, for
    :func:`numpy_from_struct` and :func:`numpy_from_column` alike. The array is
    built through ``__array_interface__``, which numpy does not bounds-check,
    so the byte range the strides reach is checked against the row's `nbytes`
    here: a struct describing memory outside its row (only a hand-built one
    can) is refused rather than read. The view holds `owner`, which keeps the
    memory alive, and is read-only.
    """
    import numpy as np

    dtype = _checked_dtype(dtype_name)
    shape, strides, offset = tuple(shape), tuple(strides), int(offset)
    if offset % dtype.itemsize != 0:
        msg = (
            f"Byte offset {offset} is not a multiple of itemsize {dtype.itemsize}; "
            "cannot reconstruct a typed strided view from this struct."
        )
        raise ValueError(msg)
    if len(strides) != len(shape):
        msg = f"strides {strides} and shape {shape} have different rank"
        raise ValueError(msg)
    if all(n > 0 for n in shape):
        low = offset + sum(
            min(0, (n - 1) * s) for n, s in zip(shape, strides, strict=True)
        )
        high = offset + sum(
            max(0, (n - 1) * s) for n, s in zip(shape, strides, strict=True)
        )
        if low < 0 or high + dtype.itemsize > nbytes:
            msg = (
                f"shape {list(shape)} with strides {list(strides)} at offset {offset} "
                f"reaches bytes [{low}, {high + dtype.itemsize}), outside the row's "
                f"{nbytes} bytes"
            )
            raise ValueError(msg)
    view = np.asarray(
        _RowView(
            owner,
            {
                "version": 3,
                "shape": shape,
                "strides": strides,
                "typestr": dtype.str,
                "data": (address + offset, True),
            },
        )
    )
    # `copy()` is C-ordered; `ascontiguousarray` would turn a 0-d scalar 1-d.
    return view.copy() if copy else view


class _RowView:
    """One row's bytes, described to numpy through ``__array_interface__``.

    ``np.asarray`` over this object views the address it names and keeps the
    object as the array's ``.base``, so the Arrow buffer the object holds (an
    ``ArrowBytes`` from the plugin) lives exactly as long as the array.
    """

    __slots__ = ("__array_interface__", "_owner")

    def __init__(self, owner: object, interface: dict[str, object]) -> None:
        self._owner = owner
        self.__array_interface__ = interface


def numpy_from_column(
    column: pl.Series,
    *,
    copy: bool = False,
) -> list[np.ndarray | None]:
    """
    Read every row of a ``sink("numpy")`` / ``sink("ndarray")`` column as a
    NumPy array, without copying.

    Each array is a view of the column's own Arrow memory, with the row's
    shape, dtype and byte strides (a transposed or flipped output stays a
    strided view). Nothing passes through Python ``bytes``, which is what
    reading rows as dicts and calling :func:`numpy_from_struct` costs: one copy
    of every row into ``bytes``, and another with ``copy=True``. The views are
    read-only, since Polars memory is immutable, and each keeps the memory it
    reads alive after the column is gone.

    Args:
        column: A numpy- or ndarray-sink output column.
        copy: Return owned, writable, C-contiguous arrays instead of views.

    Returns:
        One array per row, and ``None`` for a null row.

    Raises:
        TypeError: The column is not a numpy/ndarray sink column.
        ValueError: A row names a dtype the sink does not emit, or a view
            reaching outside the row's bytes (a hand-built struct; the sink
            never produces one).
    """
    from ._lib import binary_rows

    if isinstance(column.dtype, NdArrayType):
        column = column.ext.storage()
    if column.dtype != NUMPY_OUTPUT_SCHEMA:
        msg = (
            f"numpy_from_column reads a numpy/ndarray sink column "
            f"({NUMPY_OUTPUT_SCHEMA}), got {column.dtype}"
        )
        raise TypeError(msg)

    fields = column.struct.unnest()
    arrays: list[np.ndarray | None] = []
    for row, dtype_name, shape, strides, offset in zip(
        binary_rows(fields["data"]),
        fields["dtype"].to_list(),
        fields["shape"].to_list(),
        fields["strides"].to_list(),
        fields["offset"].to_list(),
        strict=True,
    ):
        if row is None:
            arrays.append(None)
            continue
        owner, address, nbytes = row
        arrays.append(
            _view(owner, address, nbytes, dtype_name, shape, strides, offset, copy=copy)
        )
    return arrays


def _as_buffer(data: object) -> bytes | bytearray | memoryview:
    """Get a buffer-protocol object from data, avoiding unnecessary copies.

    Tries to use memoryview for zero-copy access. Falls back to bytes()
    if the object doesn't support the buffer protocol.

    Args:
        data: The data object (typically bytes from a Polars Binary column).

    Returns:
        A buffer-protocol compatible object.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return data
    # Try memoryview for objects that support the buffer protocol
    try:
        return memoryview(data)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    except TypeError:  # pragma: no cover  # non-buffer input; defensive fallback
        # Fallback: copy into bytes
        return bytes(data)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
