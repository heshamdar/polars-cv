"""SPIKE (throwaway): shared lazy first-use registration for extension types.

Feasibility probe for the Polars-plugin design review. The extension types the
spike defines (``polars_cv.point`` / ``.contour`` / ``.bbox`` / ``.ndarray``) must be registered on
*two* copies of polars-core: the host (via ``pl.register_extension_type``) and
the plugin (via the Rust ``register_extension_type`` that runs in the ``_lib``
``#[pymodule]`` init).

The plugin-side registration only fires when Python imports the compiled
extension module (``import polars_cv._lib``) — polars loads the ``.so`` as a
plain dynamic library to resolve an expression symbol and never runs the module
init. Registering *eagerly* at import would force any import of a geometry
module to load the compiled ``.so``, breaking the "geometry imports with no
compiled extension" invariant.

So registration is **lazy**: each type module records itself via
``register_lazy`` at import (pure Python, no ``.so`` load), and the first
operation that actually needs a type calls ``ensure_registered`` — which
triggers the single ``_lib`` import and then runs the host registrations. Import stays
cheap and plugin-free; the ``.so`` is pulled in only on first real use.

Delete after the migrate-or-drop decision.
"""

from __future__ import annotations

import importlib

import polars as pl

# (ext_name, host BaseExtension subclass) recorded at type-module import.
_PENDING: list[tuple[str, type[pl.datatypes.BaseExtension]]] = []
# Names whose host registration has already run, so a type imported after the
# first ensure_registered() still gets picked up on the next call.
_REGISTERED_NAMES: set[str] = set()


def register_lazy(ext_name: str, host_cls: type[pl.datatypes.BaseExtension]) -> None:
    """Record a spike extension type to register on first use (no ``.so`` load)."""
    _PENDING.append((ext_name, host_cls))


def ensure_registered() -> None:
    """Register any not-yet-registered spike types on plugin + host, idempotently.

    Cheap to call on every entry point: returns early once nothing is pending.
    Every failure raises rather than degrading to a half-registered state — a
    host-only registration would let a tagged column be built and then decay to
    its storage at the plugin boundary, surfacing as a confusing error far from
    the cause. A name is recorded as registered only after its registration
    succeeded, so a failed attempt is retried rather than silently skipped.
    """
    pending = [(name, cls) for name, cls in _PENDING if name not in _REGISTERED_NAMES]
    if not pending:
        return

    # Plugin side first: its registration is a module-init side effect of the
    # single ``_lib`` import (later imports are sys.modules no-ops).
    try:
        lib = importlib.import_module("polars_cv._lib")
    except ImportError as e:
        msg = (
            "the spike extension types need the compiled plugin "
            "(`maturin develop --features pyo3-extension,spike-ext-types`)"
        )
        raise ImportError(msg) from e
    if not getattr(lib, "__spike_ext_types__", False):
        msg = (
            "the compiled plugin was built without the `spike-ext-types` feature, "
            "so its copy of polars-core has no spike extension types; rebuild with "
            "`maturin develop --features pyo3-extension,spike-ext-types`"
        )
        raise RuntimeError(msg)

    # Host side. No duplicate guard to swallow: ``_REGISTERED_NAMES`` already
    # keeps this from registering a name twice, so any error here is real.
    for name, cls in pending:
        pl.register_extension_type(name, cls)
        _REGISTERED_NAMES.add(name)


def is_extension_named(dtype: object, ext_name: str) -> bool:
    """True iff ``dtype`` is an extension type carrying ``ext_name``.

    Tolerant of which class the reconstructed dtype presents as (the built-in
    ``Extension`` vs. a registered subclass) — an unstable-API detail; the
    load-bearing check is the extension name.
    """
    getter = getattr(dtype, "ext_name", None)
    try:
        return callable(getter) and getter() == ext_name
    except Exception:  # noqa: BLE001
        return False
