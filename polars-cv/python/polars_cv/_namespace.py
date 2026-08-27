"""Shared base for Polars expression namespaces backed by the Rust plugin.

The ``.cv``, ``.point``, ``.contour`` and ``.bbox`` accessors all wrap the same
compiled extension and previously each re-declared ``LIB_PATH``, an identical
``__init__(self, expr)`` and the same ``register_plugin_function(...)`` call
shape. ``_PluginNamespace`` centralises that plumbing so each namespace method
collapses to a single ``self._plugin(...)`` call.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import polars as pl
from polars.plugins import register_plugin_function

from polars_cv._types import NullParamPolicy

# The compiled extension lives alongside this module in the ``polars_cv``
# package directory. Every namespace resolves to this same path.
_LIB_PATH = Path(__file__).parent

#: Accepted ``on_null(...)`` values, read from the Rust enum's Python mirror
#: rather than spelled here. ``NullParamPolicy`` is registered in
#: ``PLUGIN_REGISTRY``, so ``test_every_rust_enum_is_parity_checked`` holds the
#: mirror to what ``enum_variants("NullParamPolicy")`` reports.
_NULL_PARAM_POLICIES = tuple(p.value for p in NullParamPolicy)


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


class _GeomNullPolicy:
    """Adds ``on_null`` to the geometry accessors — and only to those.

    Deliberately **not** on :class:`_PluginNamespace`. ``.cv`` shares that base
    but routes per-row parameters through the ``vb_graph`` graph engine, where
    the policy belongs to the pipeline (``Pipeline.on_null_param``). Inheriting
    ``on_null`` onto ``.cv`` would let ``pl.col("x").cv.on_null("null")`` chain
    and read as effective while silently doing nothing, because only
    :meth:`_ArgBinder.call` reads ``_on_null``. Keeping it on a geometry-only
    mixin makes that call an ``AttributeError`` instead of a quiet no-op.
    """

    _on_null: str = "raise"

    def on_null(self, policy: str):
        """Set what a null in a per-row expression parameter means.

        These namespaces have no ``Pipeline`` object to hang a graph-level
        setting on, so the policy lives on the accessor itself and chains
        ahead of the call::

            pl.col("c").contour.on_null("null").normalize(pl.col("w"), 100)

        - ``"raise"`` (default): a null parameter fails the expression.
        - ``"null"``: rows whose parameter is null yield null, matching how a
          null input geometry is already handled.

        For a **fallback value** instead, fill the null in the expression
        itself — ``pl.col("w").fill_null(1.0)``.

        The ``.cv`` namespace deliberately has no ``on_null``; its equivalent
        is ``Pipeline.on_null_param``.

        Args:
            policy: One of ``"raise"``, ``"null"``.

        Returns:
            A copy of this namespace with the policy applied. The original is
            unchanged, matching ``Pipeline``'s immutable-builder convention.
        """
        if policy not in _NULL_PARAM_POLICIES:
            msg = f"on_null must be one of {_NULL_PARAM_POLICIES}, got '{policy}'"
            raise ValueError(msg)
        new = copy.copy(self)
        new._on_null = policy
        return new


class _ArgBinder:
    """Builds a plugin call whose parameters may be literals or expressions.

    The geometry namespaces bypass the ``vb_graph`` graph engine, so they have
    no ``ParamValue`` machinery. Their per-row channel is instead the plugin's
    *input series*: an expression-valued parameter is appended as an extra
    argument and Rust reads it at the current row.

    Position alone cannot identify those inputs. Several of these functions
    already read *optional* data operands positionally (``point.rotate``'s
    ``origin``, ``correspond``' ``order``), so an appended parameter
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
            kwargs={
                **self._kwargs,
                **kwargs,
                "input_slots": self._slots,
                # Injected centrally so no geometry method has to declare it;
                # Rust reads it in `GeomParams::new`.
                "on_null": namespace._on_null,
            },
        )
