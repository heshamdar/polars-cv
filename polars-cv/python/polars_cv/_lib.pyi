"""Type stub for the compiled Rust extension ``polars_cv._lib``.

The extension is a native module with no Python source, so a static type checker
sees nothing without this stub. Only the FFI the Python planner *imports by name*
is declared here — the plugin functions themselves (``vb_graph``,
``read_file_bytes``, the geometry entry points) are registered through
``register_plugin_function`` by path, never imported, so they need no signature.

Signatures mirror the call sites in ``pipeline.py`` / ``lazy.py`` / ``_graph.py``;
``*_json`` inputs are wire JSON, expression parameters as ``{"$slot": i}`` over
the pipeline's own expression table.

**This stub is checked against the built extension**, in both directions, by
``tests/test_lib_stub_parity.py`` (the comparison is ``tests/_stub_parity.py``):
every name, parameter name/kind/order/default and member kind here must match
``polars_cv._lib``, and every name ``_lib`` exports must be declared here or be
in that test's ``UNSTUBBED`` table with a reason. So a ``#[pyfunction]`` change
in Rust fails that test until this file follows it; there is no other check, so
edit the two together. Types are not compared (they are not recoverable from the
extension), only the shape of each signature.
"""

from __future__ import annotations

import polars as pl

__version__: str
__source_hash__: str
__debug_assertions__: bool
__build_profile__: str
__allocator__: str

class PlanState:
    """The planner's state at one op boundary (``src/plan.rs``'s ``State``).

    Built only by Rust (a :class:`Plan`'s steps); frozen.
    """

    def __init__(self) -> None:
        """The state of a pipeline with no source yet."""

    @property
    def domain(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def ndim(self) -> int | None: ...
    @property
    def dims(self) -> tuple[int | None, ...]:
        """One size per dimension when the rank is known (``None`` unknown);
        over an unknown rank, the sizes declared for the leading ones."""
    def _wire(self) -> str: ...

class Plan:
    """A pipeline's source, typed ops and the state at every op boundary
    (``src/plan.rs``). Immutable: every method returns a new plan, every state
    of which Rust planned."""

    def __init__(self) -> None:
        """The plan of a pipeline with no source and no ops."""

    @staticmethod
    def continuing(start: PlanState) -> Plan: ...
    def with_source(self, source_json: str) -> Plan: ...
    def rebased(self, start: PlanState) -> Plan: ...
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
    def to_spec(self, slot_map: list[int], node_map: dict[str, str]) -> str: ...
    def operands(self) -> list[str]: ...
    def to_node(
        self, slot_map: list[int], node_map: dict[str, str], reads: int | str
    ) -> str: ...

def pass_catalog() -> str:
    """Return the optimisation-pass catalogue as JSON (see ``tests/golden/pass_catalog.json``)."""

def op_catalog() -> str:
    """Return the typed op catalogue as JSON (see ``tests/golden/op_catalog.json``)."""

def io_catalog() -> str:
    """Return the source/sink catalogue as JSON (see ``tests/golden/io_catalog.json``)."""

def enum_catalog() -> str:
    """Return the registered-enum catalogue as JSON (see ``tests/golden/enum_catalog.json``)."""

def _this_repr(tree_json: str) -> str:
    """A ``this`` expression's text from its wire tree (Rust's ``Display``)."""

def _check_cloud_access(
    path: str, options: dict[str, str] | None = None
) -> dict[str, str | None]:
    """Resolve and exercise the credential a read of ``path`` would use:
    ``{backend, credential_source, status, error}``; see
    ``polars_cv.check_cloud_access``."""

def check_graph(graph_json: str) -> None:
    """Compile and plan a graph and check every output's sink, as the plugin
    will; raise ``ValueError``."""

def check_geom_call(function: str, args_json: str) -> None:
    """Parse a geometry accessor call's arguments against its function's
    definition; raise ``ValueError``."""

def geom_catalog() -> str:
    """Return the geometry accessor catalogue as JSON (see
    ``tests/golden/geom_catalog.json``)."""

class ArrowBytes:
    """An Arrow buffer of a column, kept alive for the arrays viewing it."""

def binary_rows(series: pl.Series) -> list[tuple[ArrowBytes, int, int] | None]:
    """Each row of a Binary column as ``(owner, address, length)``; no copies."""

def _last_fetch_peak_resident() -> int:
    """The most fetched remote bodies the most recent fetching call held at
    once; racy under concurrent calls, meant for tests."""

def _take_max_split_rows() -> int:
    """The most rows one plugin call ran since this was last called (which
    resets it); meant for tests."""

def set_row_parallelism(enabled: bool) -> bool:
    """Turn row splitting over the plugin's pool on or off process-wide;
    returns the previous setting."""

def row_parallelism() -> bool:
    """Whether plugin calls split their rows over the plugin's pool."""

def _last_split_workers() -> int:
    """Workers (calling thread + pool helpers) that ran the most recent split
    call's rows; racy under concurrent calls, meant for tests."""
