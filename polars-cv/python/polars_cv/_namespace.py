"""Shared base for Polars expression namespaces backed by the Rust plugin.

The ``.cv``, ``.point``, ``.contour`` and ``.bbox`` accessors all wrap the same
compiled extension and previously each re-declared ``LIB_PATH``, an identical
``__init__(self, expr)`` and the same ``register_plugin_function(...)`` call
shape. ``_PluginNamespace`` centralises that plumbing so each namespace method
collapses to a single ``self._plugin(...)`` call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from polars.plugins import register_plugin_function

# The compiled extension lives alongside this module in the ``polars_cv``
# package directory. Every namespace resolves to this same path.
_LIB_PATH = Path(__file__).parent


class _PluginNamespace:
    """Base class for ``@pl.api.register_expr_namespace`` accessors.

    Stores the wrapped expression and exposes :meth:`_plugin`, which forwards
    to :func:`polars.plugins.register_plugin_function` with the wrapped
    expression supplied as the first plugin argument.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def _plugin(
        self,
        function_name: str,
        *,
        args: list[pl.Expr] | None = None,
        kwargs: dict[str, Any] | None = None,
        is_elementwise: bool = True,
    ) -> pl.Expr:
        """Invoke a plugin function with ``self._expr`` as the first argument.

        Args:
            function_name: Name of the registered Rust plugin function.
            args: Additional expression arguments after ``self._expr``.
            kwargs: Static keyword arguments passed to the plugin.
            is_elementwise: Whether the function is elementwise.
        """
        return register_plugin_function(
            plugin_path=_LIB_PATH,
            function_name=function_name,
            args=[self._expr, *(args or [])],
            kwargs=kwargs,
            is_elementwise=is_elementwise,
        )
