"""Registry parity: every op is known to the builder and the executor, registered once, and contracted once (no defaulted rules)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import polars_cv
from tests._discovery import (
    package_modules,
    requires_checkout,
    rust_src_dir,
)
from tests._structural_helpers import (
    _binary_op_names_from_source,
    _lib,
    _node_op_calls_from_source,
    _plan_state,
    _planned_after,
)
from tests.conftest import plugin_required

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


@requires_checkout
def test_namespace_plugin_symbols_match_registrations():
    """The namespace plugin surface is connected in BOTH directions.

    The ``.contour``/``.point``/``.bbox``/``.cv`` namespace accessors call
    individually-registered ``#[polars_expr]`` functions by name (bypassing the
    the ``vb_graph`` graph path, so the op-catalogue tests don't cover them). Both directions are pinned, mirroring the graph-path guarantee:

    - Forward: every ``_plugin("name")`` call resolves to a registered Rust
      symbol — a typo or a rename on one side only fails here instead of only
      at execution time.
    - Reverse: every registered namespace ``#[polars_expr]`` symbol is actually
      reached by a ``_plugin(...)`` call — a Rust namespace function left
      unconnected to the Python API (the same orphan class the graph-path
      reverse-parity test guards) fails here.
    """

    src = rust_src_dir()

    called: set[str] = set()
    # Two spellings reach the same plugin: the direct `self._plugin("name", ...)`
    # and the generated geometry accessors' `self._call("name", ...)`, which
    # routes through `_plugin` after binding each expression to its input.
    # `\s*` spans the newline for multi-line calls.
    patterns = (
        r'_plugin\(\s*"([a-z_0-9]+)"',
        r'\._call\(\s*"([a-z_0-9]+)"',
    )
    for py in package_modules():
        text = py.read_text()
        for pattern in patterns:
            called |= set(re.findall(pattern, text))
    assert called, "no plugin calls found — scan is broken"

    registered: set[str] = set()
    # Two ways a namespace function is declared, and the scan must see both or
    # it silently stops covering whichever it cannot read — the failure mode
    # this file exists to prevent. `contour_accessor!` moved the `.contour`
    # accessors from the first spelling to the second, and a scanner that only
    # knew the first would have reported seventeen "missing" symbols that are
    # in fact registered.
    declarations = (
        # `#[polars_expr(...)]` immediately preceding `pub fn <name>` / `fn <name>`.
        r"#\[polars_expr[^\]]*\]\s*(?:pub\s+)?fn\s+([a-z_0-9]+)",
        # `contour_accessor! { ... <arm> fn <name> / <out_ty> ... }` and
        # `op_accessors! { op fn <name> / <out_ty> => ...; }`, each expanding
        # to exactly that pair. The arm names are matched explicitly so a new
        # arm has to be added here rather than quietly going unscanned.
        r"(?:^|\n)\s*(?:map|zip|op)\s+fn\s+([a-z_0-9]+)\s*/",
    )
    for rs in (
        "namespaces/contour.rs",
        "namespaces/point.rs",
        "image_metadata.rs",
        "read_bytes.rs",
    ):
        text = (src / rs).read_text()
        for pattern in declarations:
            registered |= set(re.findall(pattern, text))
    assert registered, "no #[polars_expr] fns found — scan is broken"
    # Each spelling must actually match something. A regex that quietly matches
    # nothing reads as green forever, so the scan asserts both halves are live
    # rather than trusting their union.
    for pattern in declarations:
        assert any(
            re.findall(pattern, (src / rs).read_text())
            for rs in (
                "namespaces/contour.rs",
                "namespaces/point.rs",
                "image_metadata.rs",
                "read_bytes.rs",
            )
        ), f"declaration pattern matched nothing, so it covers nothing: {pattern}"

    missing = sorted(called - registered)
    assert not missing, (
        "namespace _plugin() names with no matching #[polars_expr] Rust symbol "
        f"(typo or rename): {missing}"
    )
    unreached = sorted(registered - called)
    assert not unreached, (
        "registered #[polars_expr] namespace symbols with no _plugin() caller "
        f"(dead or unconnected namespace path): {unreached}"
    )


@requires_checkout
def test_lib_module_registration_matches_required_hooks():
    """The introspection FFI registered in ``#[pymodule]`` equals the hooks list.

    ``_REQUIRED_LIB_HOOKS`` is hand-maintained (the parity tests skip on any
    missing hook). This pins it to the actual ``wrap_pyfunction!`` registrations
    in ``lib.rs`` so the two cannot drift — e.g. dropping ``op_output_dtype``
    from the module must also drop it from the hooks list, and vice versa.
    """

    src = rust_src_dir()

    text = (src / "lib.rs").read_text()
    registered = set(re.findall(r"wrap_pyfunction!\(\s*([a-z_0-9]+)\s*,", text))
    assert registered, "no wrap_pyfunction! registrations found — scan is broken"
    assert registered == set(_REQUIRED_LIB_HOOKS), (
        "introspection FFI drift between lib.rs #[pymodule] and _REQUIRED_LIB_HOOKS: "
        f"in lib.rs only={sorted(registered - set(_REQUIRED_LIB_HOOKS))} "
        f"in hooks only={sorted(set(_REQUIRED_LIB_HOOKS) - registered)}"
    )


def _emitted_op_names_from_source():
    """Op names actually emitted by the Python builders, scanned from source.

    The generated methods emit ``_append_typed("<name>")`` (``Pipeline``) and
    ``_node_op("<name>")`` (``LazyPipelineExpr``'s ``lazy_only`` ops). Scanning the source keeps the comparison drift-proof without a
    second hand-maintained list.
    """

    pkg = Path(polars_cv.__file__).parent
    names: set[str] = set()
    generated = (pkg / "_ops_generated.py").read_text()
    names |= set(re.findall(r'_append_typed\(\s*"([a-z_0-9]+)"', generated))
    names |= _node_op_calls_from_source()
    return names


def test_every_op_is_emitted_by_a_builder():
    """The builders emit exactly the ops the Rust catalogue defines.

    ``TYPED_OPS`` is generated from the catalogue, so an op here is one Rust
    resolves; the scan pins the other direction too — a Rust op with no
    builder that emits it is dead or unconnected (the gap that once hid
    ``channel_merge`` and the graph-path contour ops), and a builder emitting a
    name the catalogue lacks would fail only at execution.
    """
    from polars_cv._ops_generated import TYPED_OPS

    emitted = _emitted_op_names_from_source()
    assert emitted == set(TYPED_OPS), (
        f"builders out of sync with the catalogue: "
        f"unbuilt={sorted(set(TYPED_OPS) - emitted)} "
        f"unknown={sorted(emitted - set(TYPED_OPS))}"
    )


@plugin_required
def test_registry_parity_no_dead_contracts():
    """Ops the Pipeline never emits are not executable (registry parity: sobel/laplacian/sharpen)."""
    import json

    # sobel/laplacian/sharpen lower to convolve2d; they are not real executable
    # ops, so planning them must fail (their standalone contracts are dead: registry parity).
    for lowered in ("sobel", "laplacian", "sharpen"):
        with pytest.raises(ValueError, match="Unknown operation"):
            _planned_after(json.dumps({"op": lowered}), _plan_state("buffer", "u8", 3))


_REQUIRED_LIB_HOOKS = (
    # A `this` expression's text (`ThisExpr.__repr__`, the engine's Display).
    "_this_repr",
    # The most rows one call ran, read by `test_streaming_morsels.py`.
    "_take_max_split_rows",
    # The most fetched remote bodies one call held at once, read by
    # `test_fetch_window.py`.
    "_last_fetch_peak_resident",
    # How many workers ran the most recent call's rows, read by
    # `test_an_eager_call_after_a_streaming_run_uses_the_pool`.
    "_last_split_workers",
    # The credential check behind `polars_cv.check_cloud_access`.
    "_check_cloud_access",
    # The process-wide row-splitting switch behind
    # `polars_cv.set_row_parallelism` / `row_parallelism`.
    "set_row_parallelism",
    "row_parallelism",
    # Unpickles a `PlanState` (its `__reduce__` names it); Python never calls
    # it directly.
    "_plan_state_from_json",
    # Unpickles a `Plan` (its `__reduce__` names it).
    "_plan_from_json",
    # Every optimisation pass (logical and engine), which OptFlags and
    # OPTIMIZATION_PASSES are generated from.
    "pass_catalog",
    # The `{x, y}` point struct authority, read by the geometry parity test so
    # Python's POINT_SCHEMA cannot drift from `namespaces::schema::point_fields`.
    "point_schema",
    # The contour and bbox field-name authorities, the siblings of
    # `point_schema` that hold Python's CONTOUR_SCHEMA / BBOX_SCHEMA to
    # `namespaces::schema::{CONTOUR,BBOX}_FIELD_NAMES`.
    "contour_schema",
    "bbox_schema",
    # Every extension type as (name, empty storage Series), read by
    # `test_python_types_match_the_rust_declaration` so Python's
    # `EXTENSION_TYPES` cannot drift from `ext_types::ExtType::ALL`.
    "extension_types",
    # The typed op catalogue, read by `test_the_committed_catalog_is_the_built_one`
    # so the committed JSON the Python builder is generated from cannot lag the
    # built extension.
    "op_catalog",
    # The source/sink catalogue, the same check's sibling for `io_catalog.json`.
    "io_catalog",
    # The geometry accessor catalogue, the same check's sibling for
    # `geom_catalog.json`.
    "geom_catalog",
    # Parses a geometry accessor call's literal arguments against its
    # definition as the expression is built (`_GeomNamespace._call`).
    "check_geom_call",
    # The enum catalogue the Python enum classes are generated from.
    "enum_catalog",
    # Compile and plan a whole graph and check its sinks, as the plugin will
    # (what `.sink()` runs).
    "check_graph",
)


@plugin_required
def test_lib_introspection_api_is_present():
    """A built plugin MUST expose every introspection hook (no false-green CI).

    The other parity tests ``pytest.skip`` when a hook is missing so they switch
    on automatically as the API lands. Once the plugin is actually built, a
    missing hook is a real regression (e.g. a function dropped from the
    ``#[pymodule]``), not a not-yet-implemented feature — so this guard turns it
    into a hard failure instead of a silent skip.
    """
    lib = _lib()
    assert lib is not None, "compiled _lib missing despite the plugin .so being present"
    missing = [
        name for name in _REQUIRED_LIB_HOOKS if not callable(getattr(lib, name, None))
    ]
    assert not missing, f"_lib is built but missing introspection hooks: {missing}"


#: Env flag a lane that has built the extension sets so a *missing* extension
#: fails loudly instead of self-skipping the whole structural sweep. Set by CI's
#: "Build and Test" step and by `scripts/verify.sh`, both of which run
#: `maturin develop` first; left unset for local unbuilt dev.
_REQUIRE_PLUGIN_ENV = "POLARS_CV_REQUIRE_PLUGIN"


def test_plugin_is_present_when_required() -> None:
    """In a lane that builds the extension, its absence must be a hard failure.

    Every parity test in this file is ``@plugin_required`` and *skips* without
    ``_lib`` -- and so does the ``test_lib_introspection_api_is_present``
    backstop above, which is itself ``@plugin_required``. So a lane that was
    supposed to build the plugin but somehow did not would go green on a sweep
    that never ran a single assertion. This test is deliberately **not**
    ``@plugin_required``: when ``POLARS_CV_REQUIRE_PLUGIN=1`` it asserts the
    compiled ``_lib`` is importable, turning a silently-unbuilt extension into a
    failure. Local unbuilt dev leaves the flag unset and still self-skips.
    """
    if os.environ.get(_REQUIRE_PLUGIN_ENV) != "1":
        pytest.skip(
            f"{_REQUIRE_PLUGIN_ENV} not set; plugin presence is not required in "
            "this lane (local unbuilt dev)."
        )
    assert _lib() is not None, (
        f"{_REQUIRE_PLUGIN_ENV}=1 but polars_cv._lib is not importable -- the "
        "compiled extension is missing in a lane meant to have built it (did "
        "`maturin develop` run and succeed?). Every @plugin_required parity test "
        "in this file, and its backstop, would otherwise skip silently."
    )


def _binary_dtype(op: str, left: str, right: str) -> str:
    """A binary op's planned dtype over two operand dtypes, via ``Plan.push``."""
    op_json = json.dumps({"op": op, "other": "n0"})
    return _planned_after(
        op_json, _plan_state("buffer", left, 3), {"n0": _plan_state("buffer", right, 3)}
    ).dtype


@plugin_required
def test_binary_dtype_authority():
    """The planner resolves every binary op's two-input dtype, true division included.

    Checked against the binary ops the Python API actually emits (drift-proof:
    the names are scanned from source).
    """
    emitted = _binary_op_names_from_source()
    assert emitted, "no binary ops scanned from source — scan regex out of date?"
    for op in emitted:
        result = _binary_dtype(op, "u8", "u8")
        assert result in {"u8", "f32"}, f"{op}: unexpected dtype {result}"

    # True division promotes integers to float; other ops use plain promotion.
    assert _binary_dtype("divide", "u8", "u8") == "f32"
    assert _binary_dtype("divide", "f64", "f64") == "f64"
    assert _binary_dtype("add", "u8", "u8") == "u8"
    assert _binary_dtype("add", "u8", "u16") == "u16"
    assert _binary_dtype("add", "u8", "f32") == "f32"
    # An unknown operand dtype keeps the result unknown (handled by the sink).
    assert _binary_dtype("divide", "auto", "u8") == "auto"
    assert _binary_dtype("add", "u8", "auto") == "auto"


@plugin_required
def test_a_binary_op_without_its_operand_state_is_refused():
    """A binary op plans over the state of the node it reads; with none it is
    refused rather than planned with the one-input rule."""
    add = json.dumps({"op": "add", "other": "n0"})
    with pytest.raises(ValueError, match="reads node 'n0', which has no planned state"):
        _planned_after(add, _plan_state("buffer", "u8", 3))


@requires_checkout
def test_op_schema_rules_are_required_not_defaulted():
    """The structural schema rules are REQUIRED trait methods (no default
    body). An op that omits one is a compile error, so a new op cannot silently
    inherit ``PreserveInput``/``Preserve``
    and lie about its structure — contract by the type system, not convention.
    This ratchets against re-adding a default body to ``view-buffer``'s ``Op``
    trait. It is the only place a step's shape can come from: ``GraphStep``'s
    rules are an engine op's ``Op`` impl or a graph op's own, and
    ``GraphOp::shape`` is an exhaustive match over its roles, so a new graph
    op without a shape does not compile.
    """

    src = rust_src_dir()
    traits = src.parent.parent / "view-buffer" / "src" / "ops" / "traits.rs"
    if not traits.exists():
        pytest.skip("view-buffer sources not available")
    text = traits.read_text()
    for rule, ret in (
        ("output_dtype_rule", "OutputDTypeRule"),
        ("shape", "OpShape"),
    ):
        required = re.search(rf"fn {rule}\(&self\) -> {ret};", text)
        defaulted = re.search(rf"fn {rule}\(&self\) -> {ret}\s*\{{", text)
        assert required is not None and defaulted is None, (
            f"Op::{rule} must be a required trait method with no default body "
            "so ops cannot inherit a silent, possibly-wrong structural default"
        )
