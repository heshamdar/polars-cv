"""Executable specifications for defects that are verified but not yet fixed.

Every test in this module is ``xfail(strict=True)``. Each one:

* describes a defect that has been **confirmed against running code or source**,
  not a suspicion;
* asserts the behaviour the codebase *should* have, so it fails today; and
* names, in its docstring, what "fixed" looks like.

``strict=True`` is the point. When someone lands the fix, the test XPASSes and
the suite goes **red** — which is the signal to delete the marker and let the
test join the suite proper. A backlog that lives in prose is a backlog that
rots; this one cannot silently become stale in either direction. It is the same
reasoning as `AGENTS.md`'s "The Single-Authority Refactor: What Was Done, What
Is Left" section, made executable.

These are the items recorded as deferred in that review: the ones where a fix is
a design change rather than a correction, and so wants its own commit. Adding a
test here is not a way to avoid fixing something — it is a way to stop the
knowledge evaporating between sessions.

Do **not** put a flaky or environment-dependent test here. `xfail` marks "known
broken", never "sometimes fails".
"""

from __future__ import annotations

import inspect
import re

import polars as pl
import pytest

from ._discovery import (
    package_modules,
    requires_checkout,
    rust_sources,
    suite_files,
)
from .conftest import plugin_required

#: Source-scanning members of this module check codebase shape, so they belong
#: in the lane pre-commit runs. The runtime ones carry `plugin_required`.
pytestmark = pytest.mark.structural


def _gap(reason: str) -> pytest.MarkDecorator:
    """Mark a verified, unfixed defect. Strict, so a fix fails the suite."""
    return pytest.mark.xfail(strict=True, reason=reason)


# ---------------------------------------------------------------------------
# Wire format: `deny_unknown_fields` has no mechanism behind it
# ---------------------------------------------------------------------------


@requires_checkout
@_gap(
    "the geometry/read_bytes kwargs structs are still open; only the graph "
    "wire-format structs were closed"
)
def test_every_plugin_kwargs_struct_rejects_unknown_fields() -> None:
    """Every `#[polars_expr]` kwargs struct should refuse a field it does not declare.

    `GraphNode`, `UnifiedGraph`, `OutputSpec`, `SourceSpec`, `SinkSpec` and
    `GraphKwargs` carry `#[serde(deny_unknown_fields)]`. `ContourKwargs`,
    `PointKwargs` and `ReadBytesKwargs` do not, so a kwarg Python emits and
    Rust no longer reads is silently dropped — which is exactly how
    `sink("jpeg", qualtiy=50)` encoded at the default quality with nothing
    said, and how `match_detections(strategy=)` survived unread for releases.

    Fixed by: adding the attribute to the three structs, and replacing this
    scan with a mechanism a new struct cannot step around — the scan itself is
    the weakest of the three guard kinds `CLAUDE.md` ranks, and is here only
    because nothing better exists yet.
    """
    open_structs: list[str] = []
    for path in rust_sources():
        if "polars-cv/src" not in path.as_posix():
            continue
        text = path.read_text()
        for match in re.finditer(
            r"#\[derive\([^)]*Deserialize[^)]*\)\]\s*((?:#\[[^\]]*\]\s*)*)"
            r"pub struct (\w+)",
            text,
        ):
            attrs, name = match.group(1), match.group(2)
            if "deny_unknown_fields" in attrs:
                continue
            # `OpSpec` is the one documented, permanent exception: its params
            # ride on `#[serde(flatten)]`, which serde documents as
            # incompatible with `deny_unknown_fields`.
            if name == "OpSpec":
                continue
            open_structs.append(f"{path.name}::{name}")

    assert not open_structs, (
        f"these deserialized structs accept undeclared fields: {sorted(open_structs)}"
    )


# ---------------------------------------------------------------------------
# Two surfaces for one operation, with drifted semantics
# ---------------------------------------------------------------------------


@plugin_required
@_gap(
    "Pipeline.scale_contour hardcodes ScaleOrigin::Centroid and exposes no "
    "origin=; .contour.scale defaults to origin='origin'"
)
def test_the_two_contour_scale_surfaces_agree() -> None:
    """`Pipeline.scale_contour` and `.contour.scale` should mean the same thing.

    Eight op names exist on both the graph path and the `.contour` namespace.
    `scale` has already drifted: `execute.rs` hardcodes
    `ScaleOrigin::Centroid` with no parameter, while `contours.py` exposes
    `origin=` defaulting to `"origin"`. A 2x2 square at (2,2)-(4,4) scaled by 2
    lands at (4,4)-(8,8) through the namespace and (1,1)-(5,5) through the
    pipeline — same name, different geometry, no warning.

    Fixed by: deciding which is authoritative and making the other read it, or
    renaming one so the collision is visible. Whichever way, the loser should
    fail rather than diverge.
    """
    from polars_cv.geometry import CONTOUR_SCHEMA

    square = {
        "exterior": [
            {"x": 2.0, "y": 2.0},
            {"x": 4.0, "y": 2.0},
            {"x": 4.0, "y": 4.0},
            {"x": 2.0, "y": 4.0},
        ],
        "holes": [],
    }
    df = pl.DataFrame({"c": [square]}, schema={"c": CONTOUR_SCHEMA})

    namespace = df.select(r=pl.col("c").contour.scale(2.0, 2.0))["r"].to_list()[0]
    namespace_pts = sorted((p["x"], p["y"]) for p in namespace["exterior"])

    centroid = df.select(r=pl.col("c").contour.scale(2.0, 2.0, origin="centroid"))[
        "r"
    ].to_list()[0]
    centroid_pts = sorted((p["x"], p["y"]) for p in centroid["exterior"])

    assert namespace_pts == centroid_pts, (
        f"the namespace default ({namespace_pts}) differs from what the graph "
        f"path's hardcoded centroid origin produces ({centroid_pts}); the same "
        f"op name means two things"
    )


@requires_checkout
@_gap("Pipeline.scale_contour still takes no origin= parameter")
def test_pipeline_scale_contour_exposes_the_origin_it_uses() -> None:
    """The graph path should let a caller choose the origin it silently picks.

    Fixed by: threading `origin` through `scale_contour` -> `OpSpec` ->
    `resolve_op`, so the parameter that already exists in the engine
    (`ScaleOrigin`) is reachable from both surfaces.
    """
    from polars_cv import Pipeline

    assert "origin" in inspect.signature(Pipeline.scale_contour).parameters


# ---------------------------------------------------------------------------
# Facts restated outside their authority
# ---------------------------------------------------------------------------


@requires_checkout
@_gap(
    "metrics/_matching/_contour.py still maps a column dtype to a source "
    "format itself instead of asking Rust"
)
def test_metrics_does_not_reimplement_the_auto_source_dispatch() -> None:
    """ "Which source does this column want" has an authority; metrics ignores it.

    `resolve_auto_format` (graph/compiled.rs) is what `source("auto")` — the
    default — uses, and it distinguishes a VIEW blob from image bytes by magic
    number. `_detect_source_info` maps `pl.Binary` to `"blob"` unconditionally,
    so the two already disagree for an image column.

    Fixed by: a `source_schema`-style FFI the planner and metrics both read, or
    by metrics calling `source("auto")` and letting Rust decide.
    """
    matching = next(p for p in package_modules() if p.name == "_contour.py")
    source = matching.read_text()
    assert "_detect_source_info" in source, (
        f"probe is broken: _detect_source_info not found in {matching}"
    )
    named = sorted(set(re.findall(r'format="(\w+)"', source)))
    assert not named, (
        f"metrics names concrete source formats {named}; the column->format "
        f"decision belongs to resolve_auto_format, which distinguishes a VIEW "
        f"blob from image bytes by magic number where this maps all Binary to "
        f"'blob'"
    )


@requires_checkout
@_gap("display.py still carries its own {wire code -> numpy dtype} table")
def test_display_reads_the_generated_dtype_table() -> None:
    """`display.py` should compose `_dtype_names`, not restate it.

    `_dtype_names.py` is generated from `dtype_table!` and guarded by a
    regenerate-and-diff. `display.py` hand-writes a third mapping of the same
    fact, guarded only by a regex over its own source — which `CLAUDE.md` ranks
    last of the three guard kinds, and which a reformat silently defeats.

    Fixed by: building the map from `_dtype_names.WIRE_CODES` and
    `NUMPY_NAMES`, and deleting the source-scanning guard with it.
    """
    display = next(p for p in package_modules() if p.name == "display.py")
    text = display.read_text()
    assert "_dtype_names" in text, "display.py does not read the generated dtype table"
    assert not re.search(r"dtype_map\s*=\s*\{", text), (
        "display.py still declares its own dtype_map literal"
    )


@requires_checkout
@_gap(
    "the {x, y} point struct is declared in contour.rs, point.rs and "
    "geometry/schemas.py with nothing relating them"
)
def test_the_point_struct_schema_has_one_declaration() -> None:
    """One wire schema, one declaration.

    `contour.rs::point_struct_dtype`, `point.rs::point_output_type` and
    `geometry/schemas.py::POINT_SCHEMA` each spell out `{x: f64, y: f64}`.
    Dtypes, op names and enum variants all have a named authority and a
    bidirectional parity guard; this surface has neither.

    Fixed by: one Rust declaration both plugin modules read, surfaced to Python
    the way `enum_variants` surfaces the naming registry.
    """
    # An `{x, y}` point is an adjacent x/y Float64 pair that is the *whole*
    # field list. A bbox starts `x, y, width, height`, so the third field is
    # what tells the two apart -- matching on `"x"` alone reports bboxes too,
    # and reports nothing useful.
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
        f"the point struct is built in more than one place: {declarations}"
    )


# ---------------------------------------------------------------------------
# Engine-side contract overrides and dead surface
# ---------------------------------------------------------------------------


@requires_checkout
@_gap(
    "ViewExpr::apply_op still hardcodes DType::U8 for Canny and "
    "HistogramEqualize instead of reading their declared dtype rule"
)
def test_viewexpr_reads_the_declared_output_dtype() -> None:
    """`Op::output_dtype_rule` is the authority; `apply_op` overrides it twice.

    `image.rs` already declares `OutputDTypeRule::Fixed(DType::U8)` for both
    ops, and every other arm of the same match resolves through
    `resolve_output_dtype`. Change Canny to `Fixed(U16)` and the planner
    publishes u16 while `ViewExpr` keeps tracking u8 — a second copy in the
    module the FFI reads the first copy from.

    Fixed by: routing both arms through `resolve_output_dtype` /
    `calc_strides` like their neighbours.
    """
    expr_rs = next(p for p in rust_sources() if p.name == "expr.rs").read_text()
    assert "apply_op" in expr_rs, "probe is broken: apply_op not found in expr.rs"
    assert "dtype: DType::U8" not in expr_rs, (
        "expr.rs hardcodes an output dtype rather than resolving the op's rule"
    )


@requires_checkout
@_gap("DomainOp has no implementors and is still exported and documented")
def test_domain_op_is_either_implemented_or_deleted() -> None:
    """A public trait nothing implements reads as coverage.

    `ops/traits.rs` declares `DomainOp` with a doc example naming a type that
    does not exist; nothing in either crate implements it; `ops/mod.rs`
    re-exports it and `view-buffer/AGENTS.md` describes it as live. It is also
    the only reason `traits.rs` imports `NodeOutput` — a graph concept in the
    crate whose own docs say graph concerns live in the plugin's `GraphStep`.

    Fixed by: deleting it (and its `AGENTS.md` entry), with a
    `test_removed_surfaces.py` pin so it is not "restored".
    """
    implementors = 0
    for path in rust_sources():
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("///") or stripped.startswith("//"):
                continue
            if re.search(r"\bimpl\b.*\bDomainOp\b.*\bfor\b", stripped):
                implementors += 1

    declared = any("trait DomainOp" in p.read_text() for p in rust_sources())
    assert not declared or implementors > 0, (
        "DomainOp is declared and exported but has zero implementors"
    )


# ---------------------------------------------------------------------------
# Guards that are still hand-maintained lists
# ---------------------------------------------------------------------------


@requires_checkout
@_gap("test_matchers.py still parametrizes a literal matcher list")
def test_the_matcher_conformance_sweep_is_derived() -> None:
    """A fourth matcher should join the conformance sweep by existing.

    `test_matchers.py` names `[PreMatchedAdapter, BBoxMatcher, ContourMatcher]`
    literally. That is the full set today, so the sweep is correct — and it
    stays green the day a fourth one is added without an entry, which is the
    failure mode `CLAUDE.md` describes as "a ratchet enumerating what you must
    also remember".

    Fixed by: deriving the set from `polars_cv.metrics._matching`'s public
    exports, with a floor assertion so an empty derivation fails.
    """
    module = next((p for p in suite_files() if p.name == "test_matchers.py"), None)
    assert module is not None, (
        "probe is broken: tests/test_matchers.py not found via _discovery"
    )
    suite = module.read_text()
    assert "[PreMatchedAdapter, BBoxMatcher, ContourMatcher]" not in suite, (
        "the conformance sweep enumerates its subjects by hand"
    )
