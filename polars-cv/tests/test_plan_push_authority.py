"""``Plan.push`` is the single per-op schema authority (domain, dtype, ndim), and enum parameters validate uniformly."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from polars_cv import Pipeline
from tests._structural_helpers import (
    _plan_state,
    _planned_after,
    _rust_enum_variants,
)
from tests.conftest import plugin_required

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def _op_json(op: str, **params: object) -> str:
    """An all-literal op's wire JSON."""
    return json.dumps({"op": op, **params})


@plugin_required
@pytest.mark.parametrize(
    ("op_json", "state_in", "expected"),
    [
        # cast: dtype comes from the op's own target parameter.
        (_op_json("cast", dtype="f32"), ("buffer", "u8", 3), ("buffer", "f32", 3)),
        # histogram modes: quantized keeps a buffer; buckets are struct-encoded
        # (dtype is an encoding concern -> "auto"); counts/normalized/edges are
        # typed vectors.
        (
            _op_json(
                "histogram", bins=8, range=None, closed="left", output="quantized"
            ),
            ("buffer", "u8", 3),
            ("buffer", "u32", 3),
        ),
        (
            _op_json("histogram", bins=8, range=None, closed="left", output="buckets"),
            ("buffer", "u8", 3),
            ("vector", "auto", 1),
        ),
        (
            _op_json("histogram", bins=8, range=None, closed="left", output="counts"),
            ("buffer", "u8", 3),
            ("vector", "u64", 1),
        ),
        (
            _op_json(
                "histogram", bins=8, range=None, closed="left", output="normalized"
            ),
            ("buffer", "u8", 3),
            ("vector", "f64", 1),
        ),
        (
            _op_json("histogram", bins=8, range=None, closed="left", output="edges"),
            ("buffer", "u8", 3),
            ("vector", "f64", 1),
        ),
        # reductions: axis presence decides scalar-vs-buffer.
        (_op_json("reduce_max", axis=0), ("buffer", "u8", 3), ("buffer", "u8", 2)),
        (_op_json("reduce_max"), ("buffer", "u8", 3), ("scalar", "u8", 0)),
        (_op_json("reduce_sum"), ("buffer", "u8", 3), ("scalar", "f64", 0)),
        # domain transitions.
        (_op_json("extract_shape"), ("buffer", "u8", 3), ("vector", "f64", 1)),
        # extract_contours: contour coordinates are f64 by the geometry contract.
        (
            _op_json("extract_contours", mode="external", method="simple"),
            ("buffer", "u8", 3),
            ("contour", "f64", None),
        ),
        (
            _op_json("rasterize", size=[8, 8], fill_value=255, background=0),
            ("contour", "f64", None),
            ("buffer", "u8", 3),
        ),
        # ordinary buffer op: rank/dtype preserved.
        (_op_json("grayscale"), ("buffer", "u8", 3), ("buffer", "u8", 3)),
    ],
)
def test_op_schema_authority(op_json, state_in, expected) -> None:
    """``Plan.push`` resolves the param-dependent schema cases in Rust —
    including everything the Python planner used to special-case."""
    step = _planned_after(op_json, _plan_state(*state_in))
    assert (step.domain, step.dtype, step.ndim) == expected


@plugin_required
def test_replanning_reproduces_the_appended_states() -> None:
    """Planning a pipeline's ops again from its first entering state (what
    every ``Plan.select`` — slice, reorder, deletion — does) reproduces the
    states its appends planned, at every position — so a rewrite cannot shift
    the plan."""
    corpus = [
        Pipeline().source("blob", dtype="u8").grayscale().threshold(128),
        Pipeline().source("blob", dtype="u8").cast("f32").scale(2.0),
        Pipeline()
        .source("blob", dtype="u8")
        .grayscale()
        .histogram(bins=8, output="counts"),
        Pipeline().source("blob", dtype="u8").reduce_max(axis=0),
        Pipeline().source("blob", dtype="u8").reduce_sum(),
        Pipeline()
        .source("blob", dtype="u8")
        .grayscale()
        .threshold(1)
        .extract_contours(),
        Pipeline().source("blob", dtype="u8").perceptual_hash(),
        # Ops whose builders skipped the incremental schema call entirely
        # (they append + update shape hints only) — regression coverage for
        # the tracked-state == fold invariant on that path.
        Pipeline().source("blob", dtype="u8").reshape([16]).cast("f32"),
        Pipeline().source("blob", dtype="u8").flip([0]).transpose([1, 0, 2]),
        Pipeline().source("blob", dtype="u8").crop(top=0, left=0, height=2, width=2),
        Pipeline().source("blob", dtype="u8").pad(top=1, bottom=1, left=1, right=1),
        Pipeline().source("blob", dtype="u8").pad_to_size(height=8, width=8),
        Pipeline().source("blob", dtype="u8").letterbox(height=8, width=8),
        Pipeline()
        .source("blob", dtype="u8")
        .grayscale()
        .threshold(1)
        .extract_contours()
        .simplify(tolerance=0.5)
        .convex_hull(),
    ]
    for pipe in corpus:
        plan = pipe._plan
        replayed = plan.select(list(range(len(plan))), start=0)
        ops = [json.loads(o)["op"] for o in plan.ops_json()]
        assert replayed.ops_json() == plan.ops_json(), ops
        for i in range(len(plan) + 1):
            assert replayed.state_at(i) == plan.state_at(i), f"drift at {i}: {ops}"


@plugin_required
def test_axis_reduction_ndim_decrements_exactly_once() -> None:
    """Regression: the old full-replay tracking re-subtracted axis
    reductions' ndim on every subsequent append."""
    pipe = Pipeline().source("blob", dtype="u8").assert_shape(dims=[None] * 3)
    pipe = pipe.reduce_max(axis=0)
    assert pipe._state.ndim == 2
    pipe = pipe.reduce_min(axis=0)
    assert pipe._state.ndim == 1


@plugin_required
def test_reshape_rank_tracked_eagerly() -> None:
    """Regression: reshape's builder skipped the incremental schema call, so
    the eager pipeline kept the stale pre-reshape ndim while the lazy fold
    saw the op — eager and lazy tracking disagreed. Reshape's rank is
    structural (= len(shape)), so both paths now report it exactly."""
    pipe = Pipeline().source("blob", dtype="u8").assert_shape(dims=[None] * 3)

    flat = pipe.reshape([16])
    assert flat._state.ndim == 1

    grid = pipe.reshape([2, 2, 2, 2])
    assert grid._state.ndim == 4

    # Per-row expression entries do not hide the rank: it is the entry count.
    dyn = pipe.reshape([pl.col("n"), 4])
    assert dyn._state.ndim == 2


#: ``(build, label, rust enum, a real variant)`` per enum-valued builder
#: parameter. ``build`` takes the value so the same call can be exercised with
#: a bogus one and a good one.
_ENUM_VALIDATION_CASES = [
    (
        lambda v: (
            Pipeline().source("blob", dtype="u8").resize(height=8, width=8, filter=v)
        ),
        "filter",
        "FilterType",
        "nearest",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").normalize(method=v),
        "normalize method",
        "NormalizeMethod",
        "minmax",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").perceptual_hash(algorithm=v),
        "algorithm",
        "HashAlgorithm",
        "average",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").grayscale().histogram(output=v),
        "histogram output mode",
        "HistogramOutput",
        "counts",
    ),
    (
        lambda v: Pipeline().source("blob", dtype="u8").cast(v),
        "dtype",
        "DType",
        "f32",
    ),
]


@pytest.mark.parametrize(
    ("build", "label", "enum_name", "good"),
    _ENUM_VALIDATION_CASES,
    ids=[c[1] for c in _ENUM_VALIDATION_CASES],
)
def test_enum_validation_uniform(build, label: str, enum_name: str, good: str) -> None:
    """Each enum parameter rejects a bad value *and* accepts a real one.

    The rejection half is the Rust definition's uniform error, naming the enum
    and every valid spelling.

    The acceptance half is what stops the rejection half being vacuous. On its
    own, "'bogus' raises" passes just as well when the builder rejects
    *everything* — an accidentally empty valid-set, a parameter renamed so the
    keyword lands in ``**kwargs`` and dies for an unrelated reason, or a domain
    check firing first. Each case therefore also builds with a real variant,
    and asserts that variant is one the Rust enum actually publishes rather
    than a name hard-coded here that both sides might have dropped.
    """
    with pytest.raises(ValueError, match=r"unknown \w+ \"__bogus__\""):
        build("__bogus__")

    build(good)  # must not raise

    rust = _rust_enum_variants(enum_name)
    if rust is None:
        pytest.skip("_lib.enum_catalog() not built")
    assert good in rust, (
        f"'{good}' is used here as a known-good {enum_name}, but Rust publishes "
        f"{sorted(rust)}. The positive half of this test is checking a value "
        f"the engine no longer has."
    )


#: Geometry-accessor enum parameters that are deliberately literal-only, with
#: the reason. A parameter belongs here **only** if its value fixes the output
#: shape, rank or dtype — the eligibility rule in root ``CLAUDE.md``. Empty
#: today: none of the geometry enums is structural.
_LITERAL_ONLY_GEOM_ENUMS: dict[str, str] = {}


def test_non_structural_geometry_enums_accept_an_expression() -> None:
    """A geometry enum that changes no output schema must be per-row capable.

    The rule for per-row eligibility is not the parameter's *type*: it is
    whether the value affects output shape, rank or dtype. None of the geometry
    namespaces' enums does — a winding, a scale origin, a reduction and a
    region mode all leave `List(Struct(CONTOUR_SCHEMA))` (or the reduction's
    `List(Float64)`) exactly as it was.

    Two of the four were literal-only once, and the rejection they raised
    claimed the opposite. Each accessor's fields are its Rust definition's
    (`src/namespaces/fns.rs`, or the pipeline op it shares), catalogued with whether
    each may be per-row, so this reads the catalogue: a new literal-only enum
    field has to be listed above with a reason.
    """
    catalog = json.loads(
        (Path(__file__).parent / "golden" / "geom_catalog.json").read_text()
    )
    found = {
        f"{fn['namespace']}.{fn['python']}.{field['name']}": field["type"]
        for fn in catalog
        for field in fn["fields"]
        if field["type"].get("variants")
    }
    # Non-vacuity: a catalogue shape change must fail here rather than silently
    # checking an empty set.
    assert len(found) >= 4, (
        f"found {len(found)} enum geometry parameters — the catalogue scan is "
        f"broken: {found}"
    )
    literal_only = {
        key: ty
        for key, ty in found.items()
        if not ty["per_row"] and key not in _LITERAL_ONLY_GEOM_ENUMS
    }
    assert not literal_only, (
        f"these geometry enum parameters are literal-only but change no output "
        f"shape, rank or dtype, so they are eligible to be per-row: "
        f"{list(literal_only)}. Declare them `M::V<_>`, or add them to "
        f"_LITERAL_ONLY_GEOM_ENUMS with the structural reason."
    )


def test_the_literal_only_geometry_exemptions_carry_a_reason() -> None:
    """An exemption is a decision about the eligibility rule, so it says why."""
    blank = [k for k, why in _LITERAL_ONLY_GEOM_ENUMS.items() if not why.strip()]
    assert not blank, f"exemptions without a reason: {blank}"
