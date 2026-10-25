"""Plugin-boundary structs stay closed (``deny_unknown_fields``)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest

from polars_cv import Pipeline
from tests._discovery import (
    rust_sources,
)
from tests._kwargs_scan import all_deserialized_structs, open_structs
from tests.conftest import plugin_required

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


# What the scan sees today. The floor exists so that a regex which stops
# matching fails loudly instead of reporting "nothing open" forever — the
# failure mode `CLAUDE.md` calls out and this repo has shipped more than once.
# Counting *every* deserialized struct rather than the open ones is deliberate:
# an open-struct floor would fall to zero exactly when the code became correct.
# C6b took it from 22 to 10: the eleven per-format source/sink structs (and a
# test-only `NoFields`) left serde for `#[derive(Ops)]`, whose wire refuses an
# unknown field by construction.
_DESERIALIZED_STRUCT_FLOOR = 10


def test_every_deserialized_plugin_struct_rejects_unknown_fields() -> None:
    """A struct crossing the plugin boundary must refuse a field it does not declare.

    Without ``#[serde(deny_unknown_fields)]`` a kwarg Python emits and Rust no
    longer reads is dropped in silence. That is how ``sink("jpeg",
    qualtiy=50)`` encoded at the default quality with nothing said, and how
    ``match_detections(strategy=)`` rode the wire unread for releases.

    This is a source scan — the weakest guard kind `CLAUDE.md` ranks — because
    neither stronger kind can express the property: serde cannot *require* the
    attribute at compile time, and a runtime probe only covers structs someone
    listed. Its compensating virtue is catching a struct nobody remembered, so
    it is paired with ``test_plugin_kwargs_reject_unknown_fields`` (which
    proves the attribute actually rejects, at the real entry point) and with
    ``tests/test_kwargs_scan_fixtures.py`` (which pins the matching itself).
    """
    open_by_file: dict[str, list[str]] = {}
    seen = 0
    for path in rust_sources():
        if "polars-cv/src" not in path.as_posix():
            continue
        text = path.read_text()
        seen += len(all_deserialized_structs(text))
        if found := open_structs(text, path.name):
            open_by_file[path.name] = found

    assert seen >= _DESERIALIZED_STRUCT_FLOOR, (
        f"the scan matched only {seen} deserialized structs, below the floor of "
        f"{_DESERIALIZED_STRUCT_FLOOR}. Either the regex in tests/_kwargs_scan.py "
        f"stopped matching (in which case this guard was about to pass "
        f"vacuously), or structs were genuinely removed and the floor needs "
        f"lowering deliberately."
    )
    assert not open_by_file, (
        f"these deserialized structs accept undeclared fields, so a kwarg Rust "
        f"stops reading is discarded in silence: {open_by_file}. Add "
        f"#[serde(deny_unknown_fields)], or add the name to "
        f"tests._kwargs_scan.OPEN_STRUCT_EXEMPT with the reason it cannot "
        f"carry the attribute."
    )


@plugin_required
class TestPluginKwargsRejectUnknownFields:
    """The attribute the scan above checks for must actually reject.

    ``test_every_deserialized_plugin_struct_rejects_unknown_fields`` reads
    source text, so it cannot tell a real ``deny_unknown_fields`` from the
    string appearing in a comment. These probes drive the genuine entry point —
    ``polars_cv._plugin.call``, the package's one route to
    ``register_plugin_function``, with a kwarg Rust does not declare — and
    require the query to fail. `CLAUDE.md`: verify at the user-facing entry
    point, not the helper.

    The scan and these probes cover different halves and neither subsumes the
    other: the scan catches a struct nobody remembered to close, these catch an
    attribute that is present but not doing what it claims.
    """

    # `function name -> a minimal valid kwargs dict for it`. Each probe adds one
    # undeclared key on top, so a rejection can only come from the attribute.
    _ENTRY_POINTS = {
        "read_file_bytes": {"on_error": "raise"},
        "contour_area": {},
        "point_scale": {},
    }

    @staticmethod
    def _call(function_name: str, kwargs: dict[str, object]) -> None:
        from polars_cv import _plugin

        df = pl.DataFrame({"x": ["ignored"]})
        df.select(
            _plugin.call(
                function_name,
                args=[pl.col("x")],
                kwargs=kwargs,
            )
        )

    @pytest.mark.parametrize("function_name", sorted(_ENTRY_POINTS))
    def test_an_undeclared_kwarg_is_rejected(self, function_name: str) -> None:
        base = dict(self._ENTRY_POINTS[function_name])
        with pytest.raises(pl.exceptions.ComputeError) as excinfo:
            self._call(function_name, {**base, "definitely_not_a_real_kwarg": 1})
        # The rejection must be serde's, not an incidental failure on the
        # otherwise-valid base kwargs — otherwise this passes for the wrong
        # reason and would keep passing with the attribute removed.
        assert "definitely_not_a_real_kwarg" in str(excinfo.value), (
            f"{function_name} failed, but not because of the undeclared kwarg: "
            f"{excinfo.value}"
        )


@plugin_required
class TestPointSchemaHasOneDeclaration:
    """The `{x, y}` point struct is declared once in Rust and mirrored once in Python.

    It used to be spelled out eight times — four inline constructions in
    `namespaces/contour.rs::contour_to_anyvalue`, `namespaces/contour.rs::point_struct_dtype`,
    `namespaces/contour.rs::point_anyvalue`, `namespaces/point.rs::point_output_type` and
    `geometry/schemas.py::POINT_SCHEMA` — with nothing relating them. Dtypes,
    op names and enum variants all have a named authority and a bidirectional
    parity guard; this surface had neither.

    Rust now reads `namespaces::schema::point_fields`. Python still declares
    `POINT_SCHEMA` itself, so `polars_cv.geometry` imports with no compiled
    extension present, and this test is what stops the two drifting — the
    `point_schema` FFI is decoration without it.
    """

    def test_point_schema_matches_the_rust_declaration(self) -> None:
        """Both directions, so neither side can add or drop a field alone."""
        from polars_cv._lib import point_schema
        from polars_cv.geometry.schemas import POINT_SCHEMA

        rust_names = list(point_schema())
        python_names = [field.name for field in POINT_SCHEMA.fields]

        assert python_names == rust_names, (
            f"POINT_SCHEMA and namespaces::schema::point_fields disagree: Python has "
            f"{python_names}, Rust has {rust_names}. The Rust declaration in "
            f"polars-cv/src/namespaces/schema.rs is the authority — change it there "
            f"and mirror it here."
        )

    def test_the_point_fields_are_all_float64(self) -> None:
        """Names alone would let a dtype change pass unnoticed.

        The FFI publishes names, not dtypes, so a `Float64` -> `Float32` change
        on either side would slip past the name comparison above. Rust's own
        `a_point_value_matches_the_published_dtype` covers its half; this pins
        Python's.
        """
        from polars_cv.geometry.schemas import POINT_SCHEMA

        dtypes = {field.name: field.dtype for field in POINT_SCHEMA.fields}
        assert all(dtype == pl.Float64 for dtype in dtypes.values()), (
            f"point fields must be Float64 to match namespaces::schema::point_fields, "
            f"got {dtypes}"
        )


@plugin_required
class TestContourAndBboxSchemaHaveOneDeclaration:
    """Contour and bbox structs are declared once in Rust, mirrored once in Python.

    The same treatment the point struct has, extended to the other two geometry
    structs. Before this, the contour `{exterior, holes, is_closed}` dtype was
    spelled in `contour_to_anyvalue` and again in a test helper, `parse_bbox`
    existed in both `namespaces/point.rs` and `namespaces/contour.rs`, and Python's CONTOUR_SCHEMA /
    BBOX_SCHEMA were tied to Rust only by round-trip tests. Now Rust reads
    `namespaces::schema::{contour,bbox}_fields`, and these tests hold the Python
    mirrors to `namespaces::schema::{CONTOUR,BBOX}_FIELD_NAMES` — the FFIs are
    decoration without them.
    """

    def test_contour_schema_matches_the_rust_declaration(self) -> None:
        """CONTOUR_SCHEMA field names, held to the Rust authority both ways.

        The sibling of the point guard for contours. Rust reads
        `namespaces::schema::CONTOUR_FIELD_NAMES` (via `contour_to_anyvalue` and the
        `contour_schema` FFI); Python declares CONTOUR_SCHEMA. The nested field
        *dtypes* are pinned structurally by
        `test_geometry_schemas.py::TestContourSchema`; this pins the names.
        """
        from polars_cv._lib import contour_schema
        from polars_cv.geometry.schemas import CONTOUR_SCHEMA

        rust_names = list(contour_schema())
        python_names = [field.name for field in CONTOUR_SCHEMA.fields]

        assert python_names == rust_names, (
            f"CONTOUR_SCHEMA and namespaces::schema::contour_fields disagree: Python "
            f"has {python_names}, Rust has {rust_names}. The Rust declaration in "
            f"polars-cv/src/namespaces/schema.rs is the authority — change it there "
            f"and mirror it here."
        )

    def test_bbox_schema_matches_the_rust_declaration(self) -> None:
        """BBOX_SCHEMA field names, held to the Rust authority both ways."""
        from polars_cv._lib import bbox_schema
        from polars_cv.geometry.schemas import BBOX_SCHEMA

        rust_names = list(bbox_schema())
        python_names = [field.name for field in BBOX_SCHEMA.fields]

        assert python_names == rust_names, (
            f"BBOX_SCHEMA and namespaces::schema::bbox_fields disagree: Python has "
            f"{python_names}, Rust has {rust_names}. The Rust declaration in "
            f"polars-cv/src/namespaces/schema.rs is the authority — change it there "
            f"and mirror it here."
        )

    def test_the_bbox_fields_are_all_float64(self) -> None:
        """Names alone would let a dtype change pass unnoticed.

        Rust's own `a_bbox_value_matches_the_published_dtype` covers its half;
        this pins Python's, mirroring the point-field dtype guard above.
        """
        from polars_cv.geometry.schemas import BBOX_SCHEMA

        dtypes = {field.name: field.dtype for field in BBOX_SCHEMA.fields}
        assert all(dtype == pl.Float64 for dtype in dtypes.values()), (
            f"bbox fields must be Float64 to match namespaces::schema::bbox_fields, "
            f"got {dtypes}"
        )

    def test_the_point_struct_is_constructed_in_exactly_one_place(self) -> None:
        """A *new* second Rust declaration must fail, not merely diverge later.

        The parity test above compares Python to Rust, so it cannot see a
        second Rust spelling that happens to agree today. This scan can, and it
        is what keeps the consolidation from quietly unwinding — the eight
        original sites all agreed when they were written, too.
        """
        # An `{x, y}` point is an adjacent x/y Float64 pair that is the *whole*
        # field list. A bbox starts `x, y, width, height`, so the third field
        # is what tells the two apart -- matching on `"x"` alone reports
        # bboxes too, and reports nothing useful.
        pair = re.compile(
            r'from_static\("x"\),\s*DataType::Float64\s*\),\s*'
            r'Field::new\(\s*PlSmallStr::from_static\("y"\),\s*DataType::Float64\s*\),'
            r"(?P<after>\s*(?:\]|Field::new))",
            re.S,
        )
        declarations = []
        for path in rust_sources():
            if "polars-cv/src" not in path.as_posix():
                continue
            text = path.read_text()
            for match in pair.finditer(text):
                if "Field::new" in match.group("after"):
                    continue  # a wider struct (bbox) that merely begins x, y
                line = text[: match.start()].count("\n") + 1
                declarations.append(f"{path.name}:{line}")

        assert declarations, (
            "no {x, y} point struct construction found in the plugin sources -- "
            "the probe is broken and would report 'one declaration' forever"
        )
        assert len(declarations) == 1, (
            f"the point struct is built in more than one place: {declarations}. "
            f"polars-cv/src/namespaces/schema.rs is the authority -- call "
            f"point_fields()/point_struct_dtype() instead of respelling it."
        )


@plugin_required
def test_the_committed_catalog_is_the_built_one() -> None:
    """``tests/golden/op_catalog.json`` must be what the built extension emits.

    The Rust test ``catalog_matches_the_committed_file`` pins the file to the
    definitions; this pins it to the ``.so`` actually loaded, and the generated
    module to the file, so the builder Python runs is the one Rust accepts.
    """
    import importlib.util

    from polars_cv._lib import (
        enum_catalog,
        geom_catalog,
        io_catalog,
        op_catalog,
        pass_catalog,
    )

    root = Path(__file__).resolve().parent.parent
    for name, built in (
        ("op_catalog", op_catalog),
        ("io_catalog", io_catalog),
        ("enum_catalog", enum_catalog),
        ("pass_catalog", pass_catalog),
        ("geom_catalog", geom_catalog),
    ):
        committed = (root / "tests" / "golden" / f"{name}.json").read_text()
        assert built() == committed, (
            f"{name}.json differs from the built extension: rebuild (maturin "
            "develop) or re-bless (POLARS_CV_BLESS=1 cargo test -p polars-cv "
            "catalog_matches)"
        )
    spec = importlib.util.spec_from_file_location(
        "gen_ops", root / "scripts" / "gen_ops.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUTPUT.read_text() == module.generate(), (
        "_ops_generated.py is out of date. Run: python scripts/gen_ops.py"
    )
    assert module.FORWARDERS.read_text() == module.render_forwarders(), (
        "_lazy_forwarders.py is out of date. Run: python scripts/gen_ops.py"
    )


def test_every_lazy_only_op_is_a_generated_lazy_method() -> None:
    """A ``lazy_only`` op's builder is the ``LazyPipelineExpr`` method
    ``gen_ops.py`` writes into ``_LazyOpsMixin``.

    Such an op combines this expression with other graph nodes (a binary op,
    ``apply_mask``, ``channel_merge``), so it has no ``Pipeline`` method. Its
    lazy method is generated from the catalogue like every other builder, so
    its signature, defaults and docstring cannot drift: one defined in
    ``LazyPipelineExpr`` itself would shadow it, and is refused.
    """

    from polars_cv._ops_generated import TYPED_OPS, _LazyOpsMixin
    from polars_cv.lazy import LazyPipelineExpr

    root = Path(__file__).resolve().parent.parent
    catalog = json.loads((root / "tests" / "golden" / "op_catalog.json").read_text())
    lazy_only = [op for op in catalog if op["visibility"] == "lazy_only"]
    assert lazy_only, "no lazy_only op in the catalogue — the scan matched nothing"
    for op in lazy_only:
        name = op["python"]
        assert name in vars(_LazyOpsMixin), (
            f"lazy_only op {op['name']!r} has no generated lazy method"
        )
        assert name not in vars(LazyPipelineExpr), (
            f"lazy_only op {op['name']!r} is hand-written on LazyPipelineExpr, "
            "shadowing the generated method"
        )
        assert not hasattr(Pipeline, name), (
            f"lazy_only op {op['name']!r} must not be a Pipeline method"
        )
        assert op["name"] in TYPED_OPS
