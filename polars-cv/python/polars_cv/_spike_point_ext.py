"""SPIKE (throwaway): host-side Arrow extension type for ``polars_cv.point``.

Feasibility probe for the Polars-plugin design review. Registers
``polars_cv.point`` on the host copy of polars, gives a zero-copy ``.ext.to()``
constructor, and a thin wrapper over the ``point_ext_translate`` plugin op.

This is NOT part of the public API and is not exported from ``polars_cv``.
The Rust copy of polars-core is registered separately in the ``_lib`` module
init (``src/ext_point.rs``); both registrations must agree on the name/storage
or an incoming tagged column decays to its ``{x, y}`` storage. Delete after the
migrate-or-drop decision.
"""

from __future__ import annotations

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from polars_cv._namespace import _LIB_PATH
from polars_cv.geometry.schemas import POINT_SCHEMA

POINT_EXT_NAME = "polars_cv.point"


class PointXY(pl.datatypes.BaseExtension):
    """The ``polars_cv.point`` extension type: ``{x, y}`` float storage, no metadata."""

    def __init__(self) -> None:
        super().__init__(name=POINT_EXT_NAME, storage=POINT_SCHEMA, metadata=None)

    def _string_repr(self) -> str:
        # Shown as ``ext[point[xy]]`` in a DataFrame header.
        return "point[xy]"


def _register_host() -> None:
    """Register on the host copy of polars, idempotently.

    Module import runs this once; guard against a duplicate-name error if the
    type was already registered (e.g. by a re-import in the same interpreter).
    """
    try:
        pl.register_extension_type(POINT_EXT_NAME, PointXY)
    except Exception:  # pragma: no cover - registry rejects a duplicate name
        pass


def _register_plugin() -> None:
    """Trigger the *Rust*-side registration on the plugin's copy of polars-core.

    SPIKE FINDING: polars loads the ``.so`` as a plain dynamic library to
    resolve an expression symbol, which does NOT run the ``#[pymodule]`` init
    where ``ext_point::register()`` lives. So the plugin registry stays empty
    unless something imports the extension *module*. Without this, an incoming
    ``polars_cv.point`` column decays to a generic extension inside the plugin
    and the type downcast fails. A real migration must run this registration
    before any query using the type resolves its schema (e.g. from the package
    ``__init__``), which couples geometry import to plugin presence — a design
    tension to weigh in the migrate-or-drop decision.
    """
    try:
        import polars_cv._lib  # noqa: F401  (import for its module-init side effect)
    except ImportError:  # pragma: no cover - plugin not built
        pass


_register_host()
_register_plugin()


def point_ext(x: IntoExpr, y: IntoExpr) -> pl.Expr:
    """Construct a ``polars_cv.point`` column with **no plugin call**.

    ``.ext.to()`` only relabels metadata; the input must already be the exact
    ``{x: Float64, y: Float64}`` storage, so we cast into it first. Contrast with
    the dict-building ``geometry.schemas.contour_from_points`` constructor.
    """

    def _coord(v: IntoExpr) -> pl.Expr:
        expr = pl.col(v) if isinstance(v, str) else pl.lit(v)
        return expr.cast(pl.Float64)

    return pl.struct(
        _coord(x).alias("x"),
        _coord(y).alias("y"),
    ).ext.to(PointXY())


def point_ext_translate(expr: IntoExpr, dx: float, dy: float) -> pl.Expr:
    """Translate a ``polars_cv.point`` column by ``(dx, dy)``, keeping the tag."""
    return register_plugin_function(
        plugin_path=_LIB_PATH,
        function_name="point_ext_translate",
        args=expr,
        kwargs={"dx": float(dx), "dy": float(dy)},
        is_elementwise=True,
    )
