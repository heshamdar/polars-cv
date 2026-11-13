"""``sink()`` is generated from the sink catalogue, as ``source()`` is (CR-142).

``sink()`` took ``**kwargs``: Rust refused a keyword that did not apply, but an
editor and ``ty`` saw nothing, and a multi-output's parameters were an
untyped ``(format, {params})`` tuple. The signature and the ``Sink.<format>``
constructors are now written by ``scripts/gen_ops.py`` from
``tests/golden/io_catalog.json``, so the tests here compare them with that
file: a sink field added in Rust and not regenerated fails them.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from polars_cv import Pipeline, Sink
from polars_cv._ops_generated import SINK_FIELDS
from polars_cv._types import SinkFormat
from polars_cv.lazy import LazyPipelineExpr

from .conftest import plugin_required

_PKG = Path(__file__).resolve().parents[1]
_SINKS: dict[str, dict[str, dict[str, Any]]] = {
    fmt["name"]: {field["name"]: field for field in fmt["fields"]}
    for fmt in json.loads((_PKG / "tests" / "golden" / "io_catalog.json").read_text())[
        "sinks"
    ]
}
#: `sink()`'s own parameters, which are not a sink's fields.
_OWN = ("self", "format", "opt_flags")


def _gen_ops() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gen_ops", _PKG / "scripts" / "gen_ops.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required(field: dict[str, Any]) -> bool:
    return "default" not in field and field["type"]["kind"] != "optional"


@pytest.mark.structural
class TestTheSignatureIsTheCatalogue:
    def test_every_sink_field_is_a_keyword_and_nothing_else_is(self) -> None:
        params = inspect.signature(LazyPipelineExpr.sink).parameters
        fields = {name: p for name, p in params.items() if name not in _OWN}
        assert set(fields) == set().union(*_SINKS.values())
        for name, param in fields.items():
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
            # `None` is "the format's own default": nothing is sent.
            assert param.default is None, name

    def test_sink_takes_no_open_keywords(self) -> None:
        kinds = {
            p.kind for p in inspect.signature(LazyPipelineExpr.sink).parameters.values()
        }
        assert inspect.Parameter.VAR_KEYWORD not in kinds
        assert inspect.Parameter.VAR_POSITIONAL not in kinds

    def test_the_leading_parameters_keep_their_positions(self) -> None:
        # `sink()` is the most-called method: `format` stays first; `opt_flags`
        # is keyword-only, so the old `sink("png", False)` (`return_expr`) raises.
        assert tuple(inspect.signature(LazyPipelineExpr.sink).parameters)[:3] == _OWN

    def test_every_format_has_a_constructor_taking_exactly_its_fields(self) -> None:
        constructors = {
            name
            for name, member in vars(Sink).items()
            if isinstance(member, classmethod)
        }
        assert constructors == set(_SINKS) == {f.value for f in SinkFormat}
        for fmt, fields in _SINKS.items():
            params = inspect.signature(getattr(Sink, fmt)).parameters
            assert set(params) == set(fields), fmt
            for name, param in params.items():
                assert param.kind is inspect.Parameter.KEYWORD_ONLY, (fmt, name)
                required = param.default is inspect.Parameter.empty
                assert required == _required(fields[name]), (fmt, name)

    def test_sink_fields_is_the_catalogues(self) -> None:
        assert {fmt: set(fields) for fmt, fields in SINK_FIELDS.items()} == {
            fmt: set(fields) for fmt, fields in _SINKS.items()
        }

    def test_a_field_with_two_types_across_sinks_is_refused(self) -> None:
        # One flat signature has one annotation per keyword.
        def sink(name: str, py: str) -> dict[str, Any]:
            ty = {"kind": "scalar", "per_row": False, "py": py, "this": False}
            field = {"name": "level", "doc": "", "type": ty}
            return {"name": name, "python": name, "doc": "", "fields": [field]}

        gen = _gen_ops()
        gen.format_fields([sink("a", "int"), sink("b", "int")], "sink")
        with pytest.raises(ValueError, match="sink field 'level' has two types"):
            gen.format_fields([sink("a", "int"), sink("b", "str")], "sink")


def _branches() -> LazyPipelineExpr:
    base = pl.col("image").cv.pipe(Pipeline().source("image_bytes", dtype="u8"))
    x = base.pipe(Pipeline().resize(height=8, width=8)).alias("x")
    thumb = base.pipe(Pipeline().resize(height=16, width=16)).alias("thumb")
    return x.merge_pipe(thumb)


def _one() -> LazyPipelineExpr:
    return pl.col("image").cv.pipe(
        Pipeline().source("image_bytes", dtype="u8").resize(height=8, width=8)
    )


def _sinks(graph: Any) -> dict[str, Any]:
    return {k: v["sink"] for k, v in graph._to_dict()["outputs"].items()}


@plugin_required
class TestSinkValues:
    def test_a_sink_value_is_its_format_and_the_parameters_given(self) -> None:
        assert Sink.jpeg(quality=85) == Sink("jpeg", {"quality": 85})
        # Nothing given is nothing sent: the format's own default applies.
        assert Sink.jpeg() == Sink("jpeg", {}) == Sink.jpeg(quality=None)
        assert repr(Sink.array(shape=[8, 8, 3])) == "Sink.array(shape=[8, 8, 3])"
        assert Sink.jpeg(quality=85) != Sink.jpeg(quality=84)

    def test_each_output_takes_its_own_sink(self) -> None:
        graph = _branches().graph(
            {"x": Sink.array(shape=[8, 8, 3]), "thumb": Sink.jpeg(quality=85)}
        )
        assert _sinks(graph) == {
            "x": {"format": "array", "shape": [8, 8, 3]},
            "thumb": {"format": "jpeg", "quality": 85},
        }

    def test_a_sink_value_is_accepted_wherever_a_format_is(self) -> None:
        # `meta.eq` compares the plugin call's kwargs, the graph JSON
        # included (it tells quality 40 from 41).
        by_value = _one().sink(Sink.jpeg(quality=40))
        by_keyword = _one().sink("jpeg", quality=40)
        by_enum = _one().sink(SinkFormat.JPEG, quality=40)
        assert by_value.meta.eq(by_keyword) and by_value.meta.eq(by_enum)
        assert by_value.meta.eq(_one().graph(Sink.jpeg(quality=40)).to_expr())
        mixed = _branches().graph({"x": SinkFormat.PNG, "thumb": Sink.jpeg()})
        assert _sinks(mixed) == {"x": {"format": "png"}, "thumb": {"format": "jpeg"}}

    def test_every_constructor_builds_a_sink_rust_accepts(self) -> None:
        samples: dict[str, Any] = {
            "quality": 50,
            "shape": [8, 8, 3],
            "dtype": "f16",
            "compact": True,
            "names": ["r", "g", "b"],
            "where": None,
        }
        for fmt, fields in _SINKS.items():
            value = getattr(Sink, fmt)(**{name: samples[name] for name in fields})
            # `native` is a scalar's sink; every other one takes the image.
            node = _one().reduce_mean() if fmt == "native" else _one()
            assert isinstance(node.sink(value), pl.Expr), fmt

    def test_a_parameter_given_on_the_value_and_as_a_keyword_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'quality'.*both"):
            _one().sink(Sink.jpeg(quality=10), quality=70)
        with pytest.raises(ValueError, match="'thumb'.*'quality'.*both"):
            _branches().sink({"x": "jpeg", "thumb": Sink.jpeg(quality=10)}, quality=70)

    def test_a_keyword_that_is_not_a_sink_field_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="qualtiy"):
            _one().sink("jpeg", qualtiy=50)  # ty: ignore[unknown-argument]
        with pytest.raises(TypeError, match="quality"):
            Sink.png(quality=50)  # ty: ignore[unknown-argument]
