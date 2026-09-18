"""SPIKE (throwaway): host-side Arrow extension type for ``polars_cv.bbox``.

Part of the design-review extension-type spike, rounding out the geometry family.
Tags the ``{x, y, width, height}`` struct so a consumer identifies a bbox by its
type tag, not by field-name inspection. Isolated: not wired into the ``.bbox``
namespace. Registration is lazy (see ``_spike_ext``). Delete after the
migrate-or-drop decision.
"""

from __future__ import annotations

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from polars_cv._namespace import _LIB_PATH
from polars_cv._spike_ext import ensure_registered, register_lazy
from polars_cv.geometry.schemas import BBOX_SCHEMA

BBOX_EXT_NAME = "polars_cv.bbox"


class BBox(pl.datatypes.BaseExtension):
    """The ``polars_cv.bbox`` extension type: ``{x, y, width, height}`` float storage."""

    def __init__(self) -> None:
        super().__init__(name=BBOX_EXT_NAME, storage=BBOX_SCHEMA, metadata=None)

    def _string_repr(self) -> str:
        return "bbox"


register_lazy(BBOX_EXT_NAME, BBox)


def bbox_ext(x: IntoExpr, y: IntoExpr, width: IntoExpr, height: IntoExpr) -> pl.Expr:
    """Construct a ``polars_cv.bbox`` column with no plugin call (metadata relabel)."""
    ensure_registered()

    def _f(v: IntoExpr) -> pl.Expr:
        expr = pl.col(v) if isinstance(v, str) else pl.lit(v)
        return expr.cast(pl.Float64)

    return pl.struct(
        _f(x).alias("x"),
        _f(y).alias("y"),
        _f(width).alias("width"),
        _f(height).alias("height"),
    ).ext.to(BBox())


def bbox_ext_identity(expr: IntoExpr) -> pl.Expr:
    """Pass a ``polars_cv.bbox`` column through the plugin, keeping the tag."""
    ensure_registered()
    return register_plugin_function(
        plugin_path=_LIB_PATH,
        function_name="bbox_ext_identity",
        args=expr,
        is_elementwise=True,
    )
