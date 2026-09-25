"""Type stub for the compiled Rust extension ``polars_cv._lib``.

The extension is a native module with no Python source, so a static type checker
sees nothing without this stub. Only the FFI the Python planner *imports by name*
is declared here — the plugin functions themselves (``vb_graph``,
``read_file_bytes``, the geometry entry points) are registered through
``register_plugin_function`` by path, never imported, so they need no signature.

Signatures mirror the call sites in ``pipeline.py`` / ``lazy.py`` / ``__init__.py``;
the ``op_*`` inputs are JSON-serialised op specs (``json.dumps(spec.to_dict())``).
"""

from __future__ import annotations

from typing import Any

__version__: str
__source_hash__: str

class PlanState:
    """The planner's state at one op boundary (``src/plan.rs``'s ``State``).

    Built only by Rust (``plan_source``/``plan_step``/``plan_assert``); frozen.
    """

    DIM_NAMES: tuple[str, str, str]
    def __init__(self) -> None:
        """The state of a pipeline with no source yet."""

    @property
    def domain(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def ndim(self) -> int | None: ...
    @property
    def dims(self) -> tuple[int | None, int | None, int | None]: ...
    @property
    def asserted(self) -> tuple[bool, bool, bool]: ...
    @property
    def declared(self) -> bool: ...
    def _wire(self) -> str: ...

def node_pass(
    pass_name: str,
    ops: list[str],
    states: list[PlanState],
    assertions: list[int],
) -> list[int] | None:
    """Run a node-scope logical pass; the node's new op order, or ``None``."""

def pass_catalog() -> str:
    """Return the optimisation-pass catalogue as JSON (see ``tests/golden/pass_catalog.json``)."""

def plan_step(
    op_json: str, state: PlanState, other: PlanState | None = None
) -> PlanState:
    """The planned state after appending ``op_json``; ``other`` is a binary
    op's other operand's state."""

def plan_assert(
    state: PlanState, assertion_json: str, after_op: str | None = None
) -> PlanState:
    """The planned state after a shape declaration; ``ValueError`` if it contradicts."""

def rotation_matrix_2d(
    angle_deg: float,
    cx: float,
    cy: float,
    scale: float,
) -> Any:
    """The affine parameters a ``rotate`` executes as, for a known input shape."""

def op_catalog() -> str:
    """Return the typed op catalogue as JSON (see ``tests/golden/op_catalog.json``)."""

def io_catalog() -> str:
    """Return the source/sink catalogue as JSON (see ``tests/golden/io_catalog.json``)."""

def enum_catalog() -> str:
    """Return the registered-enum catalogue as JSON (see ``tests/golden/enum_catalog.json``)."""

def plan_sink(
    sink_json: str,
    state: PlanState,
    source_json: str | None = None,
    alias: str | None = None,
) -> None:
    """Validate a serialized sink against its typed format and the output's
    planned state; raise ``ValueError``."""

def plan_source(source_json: str) -> PlanState:
    """Validate a serialized source and return its planned state."""
