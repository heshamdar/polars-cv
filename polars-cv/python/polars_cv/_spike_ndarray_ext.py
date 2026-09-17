"""SPIKE (throwaway): host-side Arrow extension type for ``polars_cv.ndarray``.

The second type in the design-review spike. It tags the numpy/torch sink struct
(``{data, dtype, shape, strides, offset}``) so a consumer can identify a tensor
column by its type tag instead of sniffing struct fields.

Isolated: this does NOT change the production ``.sink("numpy")`` path (its output
stays a plain, untagged struct) or ``numpy_from_struct``. The reader here unwraps
``.ext.storage()`` and delegates to the existing ``numpy_from_struct``. The Rust
copy is registered in the ``_lib`` module init (``src/ext_ndarray.rs``); host
registration is lazy, via the shared ``_spike_ext`` helper. Delete after the
migrate-or-drop decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from polars_cv import NUMPY_OUTPUT_SCHEMA, numpy_from_struct
from polars_cv._namespace import _LIB_PATH
from polars_cv._spike_ext import ensure_registered, is_extension_named, register_lazy

if TYPE_CHECKING:
    import numpy as np

NDARRAY_EXT_NAME = "polars_cv.ndarray"


class NdArray(pl.datatypes.BaseExtension):
    """The ``polars_cv.ndarray`` extension type over the numpy sink struct."""

    def __init__(self) -> None:
        super().__init__(
            name=NDARRAY_EXT_NAME, storage=NUMPY_OUTPUT_SCHEMA, metadata=None
        )

    def _string_repr(self) -> str:
        return "ndarray"


register_lazy(NDARRAY_EXT_NAME, NdArray)


def ndarray_ext(arr: "np.ndarray") -> pl.Series:
    """Build a one-row ``polars_cv.ndarray`` Series from a NumPy array.

    Stored C-contiguous (offset 0, C strides): the spike proves the *tag*
    round-trips, not the strided-view reconstruction that the production numpy
    path already covers. ``.ext.to()`` only relabels the exact storage struct.
    """
    import numpy as np

    ensure_registered()
    a = np.ascontiguousarray(arr)
    base = pl.DataFrame(
        {
            "data": [a.tobytes()],
            "dtype": [str(a.dtype)],
            "shape": [[int(d) for d in a.shape]],
            "strides": [[int(s) for s in a.strides]],
            "offset": [0],
        },
        schema={
            "data": pl.Binary,
            "dtype": pl.String,
            "shape": pl.List(pl.UInt64),
            "strides": pl.List(pl.Int64),
            "offset": pl.UInt64,
        },
    )
    return base.select(tagged=pl.struct(pl.all()).ext.to(NdArray())).to_series()


def numpy_from_ext(series: pl.Series, *, copy: bool = True) -> "np.ndarray":
    """Reconstruct a NumPy array from a ``polars_cv.ndarray`` column.

    Unwraps ``.ext.storage()`` when tagged, then delegates to the canonical
    ``numpy_from_struct`` — never reimplements it. The payoff over
    ``numpy_from_struct`` alone: the caller does not have to know the struct
    shape, only that the column carries the tensor tag.
    """
    storage = (
        series.ext.storage()
        if is_extension_named(series.dtype, NDARRAY_EXT_NAME)
        else series
    )
    return numpy_from_struct(storage, copy=copy)


def ndarray_ext_identity(expr: IntoExpr) -> pl.Expr:
    """Pass a ``polars_cv.ndarray`` column through the plugin, keeping the tag."""
    ensure_registered()
    return register_plugin_function(
        plugin_path=_LIB_PATH,
        function_name="ndarray_ext_identity",
        args=expr,
        is_elementwise=True,
    )
