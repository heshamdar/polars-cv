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

def op_schema(
    spec_json: str,
    domain: str,
    dtype: str,
    ndim: int | None,
) -> tuple[str, str, int | None]:
    """Resolve one op's full schema effect: ``(domain, dtype, ndim)``."""

def op_contract(spec_json: str) -> dict[str, Any]:
    """Return the full contract for a single serialized op spec."""

def op_output_channels(spec_json: str, input_channels: int | None) -> int | None:
    """Plan-time output channel count for a single op."""

def op_infer_shape(spec_json: str, dims: list[int | None]) -> list[int | None] | None:
    """Plan-time output shape for a single-buffer op. Raises ``ValueError`` when
    no shape is inferable."""

def op_identity_rule(op_json: str) -> Any:
    """The op's identity rule — the condition an identity-elimination pass uses."""

def binary_output_dtype(op: str, left_dtype: str, right_dtype: str) -> str:
    """Resolve the output dtype of a binary op given both operand dtypes."""

def rotation_matrix_2d(
    angle_deg: float,
    cx: float,
    cy: float,
    scale: float,
) -> Any:
    """The affine parameters a ``rotate`` executes as, for a known input shape."""

def known_ops() -> list[str]:
    """Return the names of every operation the executor can resolve."""

def op_catalog() -> str:
    """Return the typed op catalogue as JSON (see ``tests/golden/op_catalog.json``)."""

def enum_variants(name: str) -> list[str]:
    """Return the string variants of a Rust enum, for Python<->Rust parity checks."""
