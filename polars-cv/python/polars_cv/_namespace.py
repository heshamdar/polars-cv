"""Shared base for Polars expression namespaces backed by the Rust plugin.

The ``.cv``, ``.point``, ``.contour`` and ``.bbox`` accessors all wrap the same
compiled extension and previously each re-declared ``LIB_PATH``, an identical
``__init__(self, expr)`` and the same ``register_plugin_function(...)`` call
shape. ``_PluginNamespace`` centralises that plumbing so each namespace method
collapses to a single ``self._plugin(...)`` call, which goes through
:func:`polars_cv._plugin.call` like every other way into the plugin.
"""

from __future__ import annotations

import copy
import json
from enum import Enum
from typing import Any, TypeVar

import polars as pl

from polars_cv import _plugin
from polars_cv._types import NullParamPolicy, _to_python

#: Accepted ``on_null(...)`` values, read from ``NullParamPolicy`` — a class
#: generated from the Rust enum (``PLUGIN_REGISTRY`` → ``enum_catalog.json``)
#: rather than spelled here.
_NULL_PARAM_POLICIES = tuple(p.value for p in NullParamPolicy)


class _PluginNamespace:
    """Base class for ``@pl.api.register_expr_namespace`` accessors.

    Stores the wrapped expression and exposes :meth:`_plugin`, which forwards
    to :func:`polars_cv._plugin.call` with the wrapped expression supplied as
    the first plugin argument.
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
        return _plugin.call(
            function_name,
            args=[self._expr, *(args or [])],
            kwargs=kwargs,
            is_elementwise=is_elementwise,
        )


#: The concrete namespace type, so `on_null` chains keep their accessor methods.
_Policy = TypeVar("_Policy", bound="_GeomNamespace")


class _GeomNamespace(_PluginNamespace):
    """The geometry accessors' base: their one plugin call, and ``on_null``.

    Every ``.contour``/``.point``/``.bbox`` method is generated
    (``_ops_generated``) as one :meth:`_call` of its Rust definition
    (``src/geom_fns.rs``), with that definition's signature, defaults and
    docstring.

    ``on_null`` is here, not on ``_PluginNamespace``.

    ``.cv`` shares that base but routes per-row parameters through the
    ``vb_graph`` graph engine, where the policy belongs to the pipeline
    (``Pipeline.on_null_param``). Inheriting ``on_null`` onto ``.cv`` would let
    ``pl.col("x").cv.on_null("null")`` chain and read as effective while
    silently doing nothing, because only :meth:`_call` reads ``_on_null``.
    Keeping it on the geometry base makes that call an ``AttributeError``
    instead of a quiet no-op.
    """

    _on_null: str = "raise"

    def on_null(self: _Policy, policy: str) -> _Policy:
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

    def _call(self, function: str, values: dict[str, Any]) -> pl.Expr:
        """Call the plugin function ``function`` with its arguments ``values``.

        The per-row wire form of every typed op: an argument is its literal
        value, or — for a ``pl.Expr`` — ``{"$slot": n}``, where ``n`` is the
        position of the plugin input the expression is appended as (a
        parameter read per row, or a data operand). Each expression's position
        is written into its own field, so no input is identified by name or by
        an assumed position. Index 0 is the namespace's own expression. An
        absent optional operand (``None``) is left out. The literals are
        checked against the function's Rust definition here, as the
        expression is built (``_lib.check_geom_call``); raises ``ValueError``.
        """
        args: list[pl.Expr] = []
        fields: dict[str, Any] = {}
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, pl.Expr):
                args.append(value)
                fields[name] = {"$slot": len(args)}
            elif isinstance(value, Enum):
                fields[name] = value.value
            else:
                fields[name] = _to_python(value)
        # Refused now, where it was written, by the definition itself — not
        # when the query runs.
        from polars_cv._lib import check_geom_call

        check_geom_call(function, json.dumps(fields))
        return self._plugin(
            function,
            args=args,
            kwargs={"args": fields, "on_null": self._on_null},
        )
