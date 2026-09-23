"""SPIKE (throwaway): shared lazy first-use registration for extension types.

Feasibility probe for the Polars-plugin design review. The extension types the
spike defines (``polars_cv.point``, ``polars_cv.ndarray``) must be registered on
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
operation that actually needs a type calls ``ensure_registered`` — which runs
the host registrations and triggers the single ``_lib`` import. Import stays
cheap and plugin-free; the ``.so`` is pulled in only on first real use.

Delete after the migrate-or-drop decision.
"""

from __future__ import annotations

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
    """Register any not-yet-registered spike types on host + plugin, idempotently.

    Cheap to call on every entry point: returns early once nothing is pending.
    """
    pending = [(name, cls) for name, cls in _PENDING if name not in _REGISTERED_NAMES]
    if not pending:
        return

    for name, cls in pending:
        try:
            pl.register_extension_type(name, cls)
        except Exception:  # noqa: BLE001 - registry rejects a duplicate name
            pass
        _REGISTERED_NAMES.add(name)

    # The lazy trigger for the Rust-side registration (module-init side effect).
    # One import; later ones are sys.modules no-ops. Absent when the plugin is
    # unbuilt — host-only registration still allows pure-Python round-trips.
    try:
        import polars_cv._lib  # noqa: F401
    except ImportError:
        pass


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
