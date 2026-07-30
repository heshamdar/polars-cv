"""Shared base for Polars expression namespaces backed by the Rust plugin.

The ``.cv``, ``.point``, ``.contour`` and ``.bbox`` accessors all wrap the same
compiled extension and previously each re-declared ``LIB_PATH``, an identical
``__init__(self, expr)`` and the same ``register_plugin_function(...)`` call
shape. ``_PluginNamespace`` centralises that plumbing so each namespace method
collapses to a single ``self._plugin(...)`` call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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


class _ArgBinder:
    """Builds a plugin call whose parameters may be literals or expressions.

    The geometry namespaces bypass the ``vb_graph`` graph engine, so they have
    no ``ParamValue`` machinery. Their per-row channel is instead the plugin's
    *input series*: an expression-valued parameter is appended as an extra
    argument and Rust reads it at the current row.

    Position alone cannot identify those inputs. Several of these functions
    already read *optional* data operands positionally (``point.rotate``'s
    ``origin``, ``match_detections``' ``scores``), so an appended parameter
    would be indistinguishable from an omitted operand. Every variable
    argument — data operand and dynamic parameter alike — is therefore
    registered in ``input_slots``, a ``name -> index`` map passed as a kwarg,
    and Rust looks inputs up by name rather than by position.

    Index 0 is always the namespace's own expression (``_plugin`` prepends it),
    so the first appended argument lands at index 1.
    """

    def __init__(self) -> None:
        self._args: list[pl.Expr] = []
        self._kwargs: dict[str, Any] = {}
        self._slots: dict[str, int] = {}

    def _append(self, name: str, expr: pl.Expr) -> None:
        # +1 leaves room for the namespace's own expression at index 0.
        self._slots[name] = len(self._args) + 1
        self._args.append(expr)

    def add_data(self, name: str, expr: pl.Expr | None) -> None:
        """Register a data operand (another column), skipping it when absent."""
        if expr is not None:
            self._append(name, expr)

    def add_param(
        self,
        name: str,
        value: float | int | pl.Expr | None,
        *,
        cast: Callable[[Any], Any] = float,
    ) -> None:
        """Register a parameter as either a per-row input or a scalar kwarg."""
        if value is None:
            return
        if isinstance(value, pl.Expr):
            self._append(name, value)
        else:
            self._kwargs[name] = cast(value)

    def call(
        self,
        namespace: _PluginNamespace,
        function_name: str,
        **kwargs: Any,
    ) -> pl.Expr:
        """Invoke ``function_name`` with the collected args, kwargs and slots."""
        return namespace._plugin(
            function_name,
            args=self._args,
            kwargs={**self._kwargs, **kwargs, "input_slots": self._slots},
        )
