"""The one way polars-cv reaches its compiled plugin.

Every expression that runs Rust — the ``vb_graph`` pipeline engine and every
``.cv`` / ``.point`` / ``.contour`` / ``.bbox`` accessor — is built by
:func:`call`. ``test_only_the_plugin_module_registers_plugin_functions`` fails
on any other ``register_plugin_function`` reference in the package, so the three
things this module does cannot be skipped:

1. **It pins polars to the exact extension file Python imports.** Handed a
   directory, polars loads the first ``.so`` that ``iterdir()`` returns; beside
   a stale second build (``_lib.abi3.so`` next to ``_lib.cpython-311-*.so``)
   that is an arbitrary one of the two, and it need not be the one whose FFI
   the planner just read. ``plugin_path`` resolves the file the import system
   would load, without executing it.

2. **It hands the plugin storage, never an extension type.** Each argument is
   wrapped in ``.ext.storage()``, which is a no-op on a plain column and a
   zero-copy relabel on a tagged one. So a tagged column is accepted wherever
   its plain struct is, and no Rust function has to know extension types exist
   on the way *in* — they only ever build them on the way out (e.g.
   ``sink("ndarray")``). A tag adds no acceptance of its own: the storage is
   validated by the same parser that validates an untagged column.

3. **It declares every plugin function deterministic**, which is what makes
   polars' common-subexpression elimination (polars >= 2.0) merge equal plugin
   calls in one query: the same pipeline written twice decodes and runs once.
   Every function is a pure function of its inputs; no op draws randomness (a
   per-row random affine reads its matrices from columns). A path read sees
   storage as it is during the query, as polars' own scans do. Polars compares
   calls by their kwargs, so the graph wire carries nothing build-specific
   (``PipelineGraph._to_dict`` names nodes by position).
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Any, Sequence

import polars as pl
from polars.plugins import register_plugin_function


@lru_cache(maxsize=1)
def plugin_path() -> str:
    """The compiled extension file ``import polars_cv._lib`` would load."""
    spec = importlib.util.find_spec("polars_cv._lib")
    if spec is None or spec.origin is None:
        msg = (
            "polars-cv's compiled extension (polars_cv._lib) is not built; "
            "run `maturin develop` in polars-cv/"
        )
        raise ImportError(msg)
    return spec.origin


def call(
    function_name: str,
    *,
    args: Sequence[pl.Expr],
    kwargs: dict[str, Any] | None = None,
    is_elementwise: bool = True,
) -> pl.Expr:
    """Build an expression that runs the plugin function *function_name*.

    Args:
        function_name: Name of the ``#[polars_expr]`` function in the plugin.
        args: Expression inputs, in the order the Rust function reads them.
            Each reaches the plugin as its storage (see the module docstring).
        kwargs: Static keyword arguments, serialized to the plugin.
        is_elementwise: Whether the function is elementwise.
    """
    inputs: list[pl.Expr] = []
    for arg in args:
        if not isinstance(arg, pl.Expr):
            msg = (
                f"plugin arguments must be pl.Expr, got {type(arg).__name__} "
                f"for {function_name!r}"
            )
            raise TypeError(msg)
        inputs.append(arg.ext.storage())
    return register_plugin_function(
        plugin_path=plugin_path(),
        function_name=function_name,
        args=inputs,
        kwargs=kwargs,
        is_elementwise=is_elementwise,
        is_deterministic=True,
    )
