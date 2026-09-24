"""Guards for spec-parameter applicability: source keywords and sink keywords.

A parameter that the chosen format never reads is rejected — not warned about,
not dropped. Each source and sink format is a typed Rust struct carrying
exactly the fields its decode or encode reads (`src/formats/`), and the builder
validates what the caller passed against it over `io_check`. These tests
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
from pathlib import Path

import polars as pl
import pytest

import polars_cv
from polars_cv import Pipeline
from polars_cv._types import SinkFormat, SourceFormat

from .conftest import plugin_required

#: Every test here is a structural guard: it checks the *shape* of the codebase
#: -- registries, authorities, removed surfaces, documented vocabularies --
#: rather than the numerical behaviour of a pipeline. `-m structural` is the
#: lane pre-commit runs; see `tests/AGENTS.md`. Note that the lane as a whole
#: does need the compiled extension: many structural facts are only observable
#: through the FFI, and those tests fail rather than skip without it.
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


#: Each format's fields in its typed definition (`src/formats/`, committed as
#: `tests/golden/io_catalog.json`).
_IO_CATALOG = json.loads(
    (Path(__file__).parent / "golden" / "io_catalog.json").read_text()
)
_SOURCE_FIELDS: dict[str, set[str]] = {
    fmt["name"]: {field["name"] for field in fmt["fields"]}
    for fmt in _IO_CATALOG["sources"]
}

#: The `source()` keywords that are spelled differently on the wire: a contour
#: canvas is one `size` field, `[height, width]` or a node.
_WIRE_FIELD = {"width": "size", "height": "size", "shape": "size"}


def test_every_source_parameter_is_a_typed_source_field() -> None:
    """Every `source()` keyword is a field of some typed source, and back.

    A keyword that is no format's field could only be dropped; a field no
    keyword reaches is a setting the builder cannot make.
    """
    import inspect

    keywords = {
        name
        for name, param in inspect.signature(Pipeline.source).parameters.items()
        if name not in ("self", "format")
        and param.kind is not inspect.Parameter.VAR_KEYWORD
    }
    fields = set().union(*_SOURCE_FIELDS.values())
    assert {_WIRE_FIELD.get(k, k) for k in keywords} == fields, (
        f"keywords {sorted(keywords)} do not cover the typed source fields "
        f"{sorted(fields)}"
    )
    assert set(_SAMPLE_VALUES) == keywords, (
        f"the applicability sweep has no sample value for "
        f"{sorted(keywords - set(_SAMPLE_VALUES))}"
    )
    assert set(_SOURCE_FIELDS) == {f.value for f in SourceFormat}


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


@plugin_required
@pytest.mark.parametrize("fmt", [f.value for f in SourceFormat])
@pytest.mark.parametrize("name", sorted(_SAMPLE_VALUES))
def test_a_parameter_is_rejected_by_every_format_that_ignores_it(
    name: str, fmt: str
) -> None:
    """The whole (parameter x format) grid, decided by the typed sources.

    Applicable pairs must be accepted. Inapplicable pairs must raise, naming
    where the field does apply: not warn, not proceed. Formats with their own requirements (`raw` needs a dtype, `contour`
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

    field = _WIRE_FIELD.get(name, name)
    if field in _SOURCE_FIELDS[fmt]:
        Pipeline().source(fmt, **kwargs)  # type: ignore[arg-type]
        return
    with pytest.raises(ValueError, match=f"'{field}' does not apply .*it applies to"):
        Pipeline().source(fmt, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sink keywords
# ---------------------------------------------------------------------------

#: Each sink format and the fields its typed definition reads
#: (`src/formats/sink.rs`, committed as `tests/golden/io_catalog.json`).
_SINK_FIELDS: dict[str, set[str]] = {
    fmt["name"]: {field["name"] for field in fmt["fields"]}
    for fmt in _IO_CATALOG["sinks"]
}

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


def test_every_sink_field_has_a_sample() -> None:
    """The grid below covers every field any sink format declares."""
    fields = set().union(*_SINK_FIELDS.values())
    assert fields == set(_SINK_SAMPLES), (
        f"no sample for {sorted(fields - set(_SINK_SAMPLES))}; "
        f"stale sample for {sorted(set(_SINK_SAMPLES) - fields)}"
    )
    assert set(_SINK_FIELDS) == {f.value for f in SinkFormat}


@plugin_required
@pytest.mark.parametrize("fmt", [f.value for f in SinkFormat])
@pytest.mark.parametrize("name", sorted(_SINK_SAMPLES))
def test_a_sink_parameter_is_rejected_by_every_format_that_ignores_it(
    name: str, fmt: str
) -> None:
    """The whole (keyword x sink format) grid, decided by the typed sinks.

    `quality` is the case worth naming: `SinkSpec` called it "JPEG and WebP"
    and the sink docstring said "jpeg/webp", but the WebP arm of `encode_image`
    calls an encoder that takes no quality. A webp quality was accepted and
    dropped; it is rejected, naming where it does apply, while the pipeline is
    built.
    """
    kwargs = {name: _SINK_SAMPLES[name]}
    if name in _SINK_FIELDS[fmt]:
        _sinkable().sink(fmt, return_expr=False, **kwargs)
        return
    with pytest.raises(ValueError, match=f"'{name}' does not apply .*it applies to"):
        _sinkable().sink(fmt, return_expr=False, **kwargs)


@plugin_required
def test_a_misspelled_sink_keyword_is_rejected() -> None:
    """An open `**kwargs` accepted anything; the typed sink closes it.

    `sink("jpeg", qualtiy=50)` built a graph carrying `qualtiy`, which serde
    dropped as an unknown field — so the query encoded at quality 85 and said
    nothing.
    """
    with pytest.raises(ValueError, match="'qualtiy' is not a sink parameter"):
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
        kwargs={"graph_json": json.dumps(spec)},
    )
    with pytest.raises(Exception, match="definitely_not_a_field|unknown field"):
        df.lazy().select(out=expr).collect()


@plugin_required
def test_the_quality_declaration_matches_what_the_encoders_do() -> None:
    """Check the typed sinks' `quality` claim against the encoders themselves.

    The grid above holds the definitions only to their own word: give the webp
    sink a `quality` field and the grid happily accepts webp qualities again.
    `SinkSpec` called the field "JPEG and WebP" and the sink docstring said
    "jpeg/webp", but the WebP arm of `encode_image` calls an encoder that takes
    no quality. So: the one format that declares `quality` must encode
    differently at 10 and 95, and every other image format must refuse the
    field on the wire itself — there is no way left to hand an encoder a
    quality it ignores.

    Driven through a hand-built graph, because the builder refuses the keyword
    before the wire is reached.
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
            kwargs={"graph_json": json.dumps(spec)},
        )
        return df.lazy().select(out=expr).collect()["out"][0]

    declared = [
        f for f in ("png", "jpeg", "webp", "tiff") if "quality" in _SINK_FIELDS[f]
    ]
    assert declared == ["jpeg"], declared
    assert encoded("jpeg", 10) != encoded("jpeg", 95), (
        "the jpeg sink declares quality but its output did not change between "
        "quality 10 and 95"
    )
    for fmt in ("png", "webp", "tiff"):
        with pytest.raises(
            pl.exceptions.ComputeError, match="'quality' does not apply"
        ):
            encoded(fmt, 10)
