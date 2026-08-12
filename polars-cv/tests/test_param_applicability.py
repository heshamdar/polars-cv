"""Guards for spec-parameter applicability: source keywords and sink keywords.

A parameter that the chosen format never reads is rejected — not warned about,
not dropped. One table per surface (`SOURCE_PARAM_APPLIES`, `SINK_PARAM_APPLIES`)
lists each parameter against the formats whose decode or encode actually reads
it, and one checker (`reject_inapplicable_params`) answers both. These tests
exist because the question used to be answered per parameter: of the source's
seven scoped keywords one raised, one warned and five were silently dropped,
while `.sink()` — an open `**kwargs` — accepted literally any keyword, spread it
into the graph JSON, and let serde drop it.

The sink half is the reason this is not a `source()` file: the same defect, the
same fix, and the worse of the two surfaces.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline
from polars_cv._types import (
    PARAM_HINTS,
    SINK_PARAM_APPLIES,
    SOURCE_PARAM_APPLIES,
    SinkFormat,
    SourceFormat,
)

from .conftest import plugin_required

#: Every test here is a structural guard: it checks the *shape* of the
#: codebase rather than the behaviour of a pipeline, so it needs no compiled
#: extension and runs in milliseconds. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def _pipeline_ast() -> ast.ClassDef:
    source = Path(polars_cv.pipeline.__file__).read_text()
    tree = ast.parse(source)
    return next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline"
    )


# ---------------------------------------------------------------------------
# Source keywords
# ---------------------------------------------------------------------------


#: A non-default value per `source()` keyword, for the sweep below. Asserted
#: complete against the table, so a new parameter cannot join with no case —
#: a sweep that silently skips the parameter someone just added is the failure
#: mode this file exists to prevent.
_SAMPLE_VALUES: dict[str, object] = {
    "dtype": "f32",
    "width": 32,
    "height": 16,
    "shape": None,  # filled in per-call: it must be a LazyPipelineExpr
    "fill_value": 7,
    "background": 3,
    "cloud_options": {"aws_region": "eu-west-1"},
    "allowed_roots": ["/tmp"],
    "require_contiguous": True,
    "decode_max_size": 64,
    "on_error": "null",
}


def _sample_for(name: str) -> object:
    if name == "shape":
        return pl.col("i").cv.pipe(Pipeline().source("image_bytes"))
    return _SAMPLE_VALUES[name]


def test_every_source_parameter_declares_where_it_applies() -> None:
    """No `source()` keyword may sit outside the applicability table.

    The table is what the rejection reads, so a parameter missing from it is
    a parameter that applies everywhere by omission — silently ignored by the
    formats that do not read it, which is exactly the state this replaced
    (`width` on an image source, `require_contiguous` on a contour source,
    `allowed_roots` on anything but a path source: all accepted, all dropped).
    """
    import inspect

    declared = set(SOURCE_PARAM_APPLIES)
    keywords = {
        name
        for name, param in inspect.signature(Pipeline.source).parameters.items()
        if name not in ("self", "format")
        and param.kind is not inspect.Parameter.VAR_KEYWORD
    }
    assert declared == keywords, (
        f"undeclared: {sorted(keywords - declared)}; "
        f"declared but not a parameter: {sorted(declared - keywords)}"
    )
    assert set(_SAMPLE_VALUES) == keywords, (
        f"the applicability sweep has no sample value for "
        f"{sorted(keywords - set(_SAMPLE_VALUES))}"
    )
    assert {name for kind, name in PARAM_HINTS if kind == "source"} <= keywords


def test_source_applicability_reads_every_parameter() -> None:
    """The check must be handed the parameters, not a copy of their names.

    `source()` snapshots `locals()` before binding anything else, so the list
    cannot drift from the signature. A hand-written dict there would pass the
    table guard above and still skip whichever parameter its author forgot.
    """
    source = next(
        m
        for m in _pipeline_ast().body
        if isinstance(m, ast.FunctionDef) and m.name == "source"
    )
    first = source.body[0]
    while isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        source.body.pop(0)  # the docstring
        first = source.body[0]
    assert isinstance(first, ast.Assign), (
        "source() must capture its parameters as its first statement"
    )
    assert "locals" in ast.dump(first.value), (
        "source() must read its parameters from locals(), so the applicability "
        "check cannot be given a stale list of them"
    )


@pytest.mark.parametrize("fmt", [f.value for f in SourceFormat])
@pytest.mark.parametrize("name", sorted(SOURCE_PARAM_APPLIES))
def test_a_parameter_is_rejected_by_every_format_that_ignores_it(
    name: str, fmt: str
) -> None:
    """The whole (parameter x format) grid, decided by the table.

    Applicable pairs must be accepted — a build error here would mean the table
    is stricter than the decode. Inapplicable pairs must raise: not warn, not
    proceed. Formats with their own requirements (`raw` needs a dtype, `contour`
    needs a canvas) can still reject an *applicable* pair for that reason, so
    only the rejection message is asserted, not the fact of raising.
    """
    kwargs: dict[str, object] = {}
    if fmt == "contour" and name != "shape":
        # A contour source needs a canvas; `shape=` *is* one, and refuses to
        # share with explicit dims.
        kwargs.update(width=8, height=8)
    if fmt == "raw" and name != "dtype":
        kwargs["dtype"] = "u8"  # raw has no type metadata to infer from
    kwargs[name] = _sample_for(name)

    applies = SourceFormat(fmt) in SOURCE_PARAM_APPLIES[name]
    if applies:
        Pipeline().source(fmt, **kwargs)  # type: ignore[arg-type]
        return
    with pytest.raises(ValueError, match=f"{name} does not apply"):
        Pipeline().source(fmt, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sink keywords
# ---------------------------------------------------------------------------

#: A non-default value per sink keyword, for the grid below.
_SINK_SAMPLES: dict[str, object] = {
    "quality": 50,
    "shape": [2, 2, 1],
    "dtype": "f16",
}


def _sinkable() -> "pl.Expr":
    """A pipeline whose element dtype is known, so a typed sink can be planned."""
    return pl.col("i").cv.pipe(
        Pipeline().source("image_bytes", dtype="u8").resize(width=2, height=2)
    )


def test_every_sink_wire_field_declares_where_it_applies() -> None:
    """The table must cover exactly the fields the sink sends to Rust.

    `.sink()` takes `**kwargs`, so unlike `source()` there is no signature to
    compare against — the wire struct is the surface. Reading `SinkSpec` from
    the Rust source is a scan, with the limits scans have: it matches `pub`
    fields by name and knows one serde alias (`out_dtype` is spelled `dtype` by
    the user). It cannot tell whether the encoder *reads* a field, which is why
    the grid below asserts behaviour rather than trusting the list.
    """
    rust = (Path(polars_cv.__file__).parents[2] / "src" / "pipeline.rs").read_text()
    body = rust.split("pub struct SinkSpec {", 1)[1].split("\n}", 1)[0]
    fields = set(re.findall(r"pub (\w+):", body)) - {"format"}
    spelled = {"out_dtype": "dtype"}
    declared = {spelled.get(f, f) for f in fields}
    assert declared == set(SINK_PARAM_APPLIES), (
        f"on the wire but undeclared: {sorted(declared - set(SINK_PARAM_APPLIES))}; "
        f"declared but not on the wire: {sorted(set(SINK_PARAM_APPLIES) - declared)}"
    )
    assert set(_SINK_SAMPLES) == set(SINK_PARAM_APPLIES)
    assert {name for kind, name in PARAM_HINTS if kind == "sink"} <= set(
        SINK_PARAM_APPLIES
    )


@plugin_required
@pytest.mark.parametrize("fmt", [f.value for f in SinkFormat])
@pytest.mark.parametrize("name", sorted(SINK_PARAM_APPLIES))
def test_a_sink_parameter_is_rejected_by_every_format_that_ignores_it(
    name: str, fmt: str
) -> None:
    """The whole (keyword x sink format) grid, decided by the table.

    `quality` is the case worth naming: `SinkSpec` called it "JPEG and WebP"
    and the sink docstring said "jpeg/webp", but the WebP arm of `encode_image`
    calls an encoder that takes no quality. A webp quality was accepted and
    dropped; it is now rejected, which says the true thing.
    """
    kwargs = {name: _SINK_SAMPLES[name]}
    if SinkFormat(fmt) in SINK_PARAM_APPLIES[name]:
        _sinkable().sink(fmt, return_expr=False, **kwargs)
        return
    with pytest.raises(ValueError, match=f"{name} does not apply"):
        _sinkable().sink(fmt, return_expr=False, **kwargs)


@plugin_required
def test_a_misspelled_sink_keyword_is_rejected() -> None:
    """An open `**kwargs` accepted anything; the table closes it.

    `sink("jpeg", qualtiy=50)` built a graph carrying `qualtiy`, which serde
    dropped as an unknown field — so the query encoded at quality 85 and said
    nothing.
    """
    with pytest.raises(ValueError, match="qualtiy is not a sink parameter"):
        _sinkable().sink("jpeg", return_expr=False, qualtiy=50)


@plugin_required
def test_the_sink_wire_rejects_an_unknown_field() -> None:
    """And the wire itself is closed, as the node end already was.

    The builder can no longer emit an unknown key, so this pins the other way
    in: a hand-built graph must not be able to carry a sink field nothing
    reads. Same mechanism as `test_graph_node_rejects_unknown_fields`.
    """
    df = pl.DataFrame({"i": [b""]})
    graph = _sinkable().sink("png", return_expr=False)
    spec = json.loads(graph._to_json())
    for output in spec["outputs"].values():
        output["sink"]["definitely_not_a_field"] = 1

    expr = pl.col("i").cv._plugin(  # type: ignore[attr-defined]
        "vb_graph",
        kwargs={"graph_json": json.dumps(spec), "expr_column_names": []},
    )
    with pytest.raises(Exception, match="definitely_not_a_field|unknown field"):
        df.lazy().select(out=expr).collect()


@plugin_required
def test_the_quality_declaration_matches_what_the_encoders_do() -> None:
    """Check the table's `quality` claim against the encoders themselves.

    The grid above holds the table only to its own word: declare `quality` for
    webp and the grid happily accepts webp qualities again. This asserts the
    thing the declaration rests on — for each image sink, whether the bytes
    change with quality must equal what the table says. `SinkSpec` called the
    field "JPEG and WebP" and the sink docstring said "jpeg/webp", but the WebP
    arm of `encode_image` calls an encoder that takes no quality, so restoring
    that reading fails here, where it is wrong.

    Driven through a hand-built graph because the builder now refuses the
    keyword for the formats under test: that is the point, and it is the only
    way to observe what an encoder does with a field it is handed anyway.
    """
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
        buf, format="PNG"
    )
    df = pl.DataFrame({"i": [buf.getvalue()]})
    pipe = pl.col("i").cv.pipe(Pipeline().source("image_bytes"))

    def encoded(fmt: str, quality: int) -> bytes:
        graph = pipe.sink(fmt, return_expr=False)
        spec = json.loads(graph._to_json())
        # `_to_json()` alone carries no column bindings — `to_expr()` builds
        # them — so bind the single root here to make the graph runnable.
        spec["column_bindings"] = {node: 0 for node in spec["nodes"]}
        for output in spec["outputs"].values():
            output["sink"]["quality"] = quality
        expr = pl.col("i").cv._plugin(  # type: ignore[attr-defined]
            "vb_graph",
            kwargs={"graph_json": json.dumps(spec), "expr_column_names": []},
        )
        return df.lazy().select(out=expr).collect()["out"][0]

    for fmt in ("png", "jpeg", "webp", "tiff"):
        declared = SinkFormat(fmt) in SINK_PARAM_APPLIES["quality"]
        observed = encoded(fmt, 10) != encoded(fmt, 95)
        assert observed == declared, (
            f"SINK_PARAM_APPLIES says the {fmt} encoder "
            f"{'reads' if declared else 'ignores'} quality, but its output "
            f"{'changed' if observed else 'did not change'} between quality 10 "
            f"and 95"
        )
