"""Type stub for the compiled Rust extension ``polars_cv._lib``.

The extension is a native module with no Python source, so a static type checker
sees nothing without this stub. Only the FFI the Python planner *imports by name*
is declared here — the plugin functions themselves (``vb_graph``,
``read_file_bytes``, the geometry entry points) are registered through
``register_plugin_function`` by path, never imported, so they need no signature.

Signatures mirror the call sites in ``pipeline.py`` / ``lazy.py`` / ``_graph.py``;
``*_json`` inputs are wire JSON, expression parameters as ``{"$slot": i}`` over
the pipeline's own expression table.
"""

from __future__ import annotations

from typing import Any

__version__: str
__source_hash__: str

class PlanState:
    """The planner's state at one op boundary (``src/plan.rs``'s ``State``).

    Built only by Rust (a :class:`Plan`'s steps); frozen.
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
    def _wire(self) -> str: ...

class Plan:
    """A pipeline's source, typed ops and the state at every op boundary
    (``src/plan.rs``). Immutable: every method returns a new plan, every state
    of which Rust planned."""

    def __init__(self) -> None:
        """The plan of a pipeline with no source and no ops."""

    @staticmethod
    def continuing(start: PlanState) -> Plan: ...
    def with_source(
        self, source_json: str, refs: dict[str, PlanState] | None = None
    ) -> Plan: ...
    def rebased(self, source_json: str, start: PlanState) -> Plan: ...
    def push(self, op_json: str, refs: dict[str, PlanState] | None = None) -> Plan: ...
    def select(self, positions: list[int], start: int | None = None) -> Plan: ...
    def run_pass(self, pass_name: str) -> Plan | None: ...
    def state_at(self, position: int) -> PlanState: ...
    @property
    def state(self) -> PlanState: ...
    @property
    def has_source(self) -> bool: ...
    @property
    def source_format(self) -> str | None: ...
    def __len__(self) -> int: ...
    def ops_json(self) -> list[str]: ...
    def source_json(self) -> str | None: ...
    def to_spec(self, slot_map: list[int]) -> str: ...

def pass_catalog() -> str:
    """Return the optimisation-pass catalogue as JSON (see ``tests/golden/pass_catalog.json``)."""

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

def check_graph(graph_json: str) -> None:
    """Compile and plan a graph and check every output's sink, as the plugin
    will; raise ``ValueError``."""
