"""SPIKE (throwaway): host-side Arrow extension type for ``polars_cv.contour``.

Part of the design-review extension-type spike, rounding out the geometry family.
Tags the ``{exterior, holes, is_closed}`` struct so a consumer identifies a
contour by its type tag instead of the structural "looks-like" matching
``parse_contour`` does today. Isolated: not wired into the ``.contour`` namespace.
Registration is lazy (see ``_ext``). Delete after the migrate-or-drop decision.
"""

from __future__ import annotations

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from polars_cv._namespace import _LIB_PATH
from polars_cv.geometry.schemas import CONTOUR_SCHEMA, contour_from_points
from tests.spike_point_ext._ext import ensure_registered, register_lazy

CONTOUR_EXT_NAME = "polars_cv.contour"


class Contour(pl.datatypes.BaseExtension):
    """The ``polars_cv.contour`` extension type over the ``{exterior, holes, is_closed}`` struct."""

    def __init__(self) -> None:
        super().__init__(name=CONTOUR_EXT_NAME, storage=CONTOUR_SCHEMA, metadata=None)

    def _string_repr(self) -> str:
        return "contour"


register_lazy(CONTOUR_EXT_NAME, Contour)


def contour_ext(
    points: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> pl.Series:
    """Build a one-row ``polars_cv.contour`` Series from ``(x, y)`` tuples.

    Reuses the existing ``contour_from_points`` builder for the dict, then relabels
    with ``.ext.to()`` (no plugin call). Contrast: today that same dict would flow
    as an anonymous struct that every consumer re-validates by shape.
    """
    ensure_registered()
    contour_dict = contour_from_points(points, holes=holes)
    base = pl.DataFrame({"c": [contour_dict]}, schema={"c": CONTOUR_SCHEMA})
    return base.select(tagged=pl.col("c").ext.to(Contour())).to_series()


def contour_ext_identity(expr: IntoExpr) -> pl.Expr:
    """Pass a ``polars_cv.contour`` column through the plugin, keeping the tag."""
    ensure_registered()
    return register_plugin_function(
        plugin_path=_LIB_PATH,
        function_name="contour_ext_identity",
        args=expr,
        is_elementwise=True,
    )
