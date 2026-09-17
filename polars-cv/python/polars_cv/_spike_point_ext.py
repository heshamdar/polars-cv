"""SPIKE (throwaway): host-side Arrow extension type for ``polars_cv.point``.

Feasibility probe for the Polars-plugin design review. Registers
``polars_cv.point`` on the host copy of polars, gives a zero-copy ``.ext.to()``
constructor, and a thin wrapper over the ``point_ext_translate`` plugin op.

This is NOT part of the public API and is not exported from ``polars_cv``.
Registration is lazy (see ``_spike_ext``): the host type is recorded at import
via ``register_lazy`` and both copies of polars-core are registered on first use
by ``ensure_registered`` — so importing this module does not load the compiled
``.so``. Both registrations must agree on the name/storage or an incoming tagged
column decays to its ``{x, y}`` storage. Delete after the migrate-or-drop decision.
"""

from __future__ import annotations

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from polars_cv._namespace import _LIB_PATH
from polars_cv._spike_ext import ensure_registered, register_lazy
from polars_cv.geometry.schemas import POINT_SCHEMA

POINT_EXT_NAME = "polars_cv.point"


class PointXY(pl.datatypes.BaseExtension):
    """The ``polars_cv.point`` extension type: ``{x, y}`` float storage, no metadata."""

    def __init__(self) -> None:
        super().__init__(name=POINT_EXT_NAME, storage=POINT_SCHEMA, metadata=None)

    def _string_repr(self) -> str:
        # Shown as ``ext[point[xy]]`` in a DataFrame header.
        return "point[xy]"


register_lazy(POINT_EXT_NAME, PointXY)


def point_ext(x: IntoExpr, y: IntoExpr) -> pl.Expr:
    """Construct a ``polars_cv.point`` column with **no plugin call**.

    ``.ext.to()`` only relabels metadata; the input must already be the exact
    ``{x: Float64, y: Float64}`` storage, so we cast into it first. Contrast with
    the dict-building ``geometry.schemas.contour_from_points`` constructor.
    """
    ensure_registered()

    def _coord(v: IntoExpr) -> pl.Expr:
        expr = pl.col(v) if isinstance(v, str) else pl.lit(v)
        return expr.cast(pl.Float64)

    return pl.struct(
        _coord(x).alias("x"),
        _coord(y).alias("y"),
    ).ext.to(PointXY())


def point_ext_translate(expr: IntoExpr, dx: float, dy: float) -> pl.Expr:
    """Translate a ``polars_cv.point`` column by ``(dx, dy)``, keeping the tag."""
    ensure_registered()
    return register_plugin_function(
        plugin_path=_LIB_PATH,
        function_name="point_ext_translate",
        args=expr,
        kwargs={"dx": float(dx), "dy": float(dy)},
        is_elementwise=True,
    )
