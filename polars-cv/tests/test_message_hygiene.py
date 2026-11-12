"""No message a user reads carries a Debug value, a JSON position or a node id (CR-143).

``_message_hygiene.leaks`` is the checker; its fixtures are below, the three
messages that shipped among them. ``FAILURES`` is a table of deliberate
mistakes, one per way a message is produced (a row failing to decode, a sink
the wire refuses, an op the planner refuses, a geometry call, a path): each
must raise, and what it raises must be clean. Add a row with a new kind of
failure, not a test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from polars_cv import Pipeline

from ._message_hygiene import leaks, own_words
from .conftest import make_test_png, plugin_required

# --- Fixtures: the checker itself ---------------------------------------------

#: Messages as they shipped in 0.41.0, and the rule each must trip.
_KNOWN_BAD = {
    "a Rust Debug value": (
        "Failed to decode image: Unsupported(UnsupportedError "
        "{ format: Unknown, kind: Format(Unknown) })"
    ),
    "a JSON position": (
        "Failed to parse pipeline graph: unknown sink format 'nunpy', expected "
        'one of ["array", "blob"] at line 1 column 230'
    ),
    "an internal node id": "root node 'node_e16ec35b' has no source",
}

_KNOWN_GOOD = [
    "Failed to decode image: The image format could not be determined",
    'unknown sink format \'nunpy\', expected one of ["array", "blob"]',
    "this pipeline has no source; start it with `.source()`",
    # Braces and colons that are not a Debug value.
    'sink("array", shape=[224, 224, 3]) and {"x": "tensor"}',
    "column 'img' is List(Struct({'x': Float64, 'y': Float64}))",
    "the long sink gives List[Struct{row: u32, col: u32}]",
    # A row reference, and a user's own names, are the user's words.
    "resize: height: Parameter column 'h' has a null value (row 3 of 12 in this batch)",
    "output 'node_of_interest' not found; in column 7 of the csv",
]


@pytest.mark.structural
class TestTheChecker:
    @pytest.mark.parametrize("rule", sorted(_KNOWN_BAD))
    def test_a_shipped_leak_trips_its_rule(self, rule: str) -> None:
        found = leaks(_KNOWN_BAD[rule])
        assert len(found) == 1 and found[0].startswith(rule), found

    @pytest.mark.parametrize("message", _KNOWN_GOOD)
    def test_a_clean_message_trips_nothing(self, message: str) -> None:
        assert leaks(message) == []

    def test_polars_trace_is_not_the_plugins_words(self) -> None:
        error = ValueError(
            "decode failed\n\nThis error occurred in the following expression:\n"
            '\tcol("img")./lib.so:vb_graph() at line 1 column 2'
        )
        assert own_words(error) == "decode failed"


# --- The table of deliberate failures -----------------------------------------


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "img": [make_test_png()],
            "junk": [b"not an image"],
            "cut": [make_test_png()[:40]],
            "h": [None],
            "path": ["/no/such/file.png"],
            "poly": [[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]],
        },
        schema_overrides={"h": pl.Int64},
    )


def _src() -> Pipeline:
    return Pipeline().source("image_bytes")


def _run(expr: Callable[[], Any]) -> Callable[[], Any]:
    return lambda: _frame().select(expr())


#: name -> a call that must raise. Rows fail in the plugin; the rest are
#: refused as they are built.
FAILURES: dict[str, Callable[[], Any]] = {
    "a row that is not an image": _run(
        lambda: pl.col("junk").cv.pipe(_src()).sink("tensor")
    ),
    "a truncated image": _run(lambda: pl.col("cut").cv.pipe(_src()).sink("tensor")),
    "a blob that is not one": _run(
        lambda: pl.col("junk").cv.pipe(Pipeline().source("blob")).sink("tensor")
    ),
    "a missing file": _run(
        lambda: (
            pl.col("path")
            .cv.pipe(Pipeline().source("file_path", allowed_roots=["/no"]))
            .sink("tensor")
        )
    ),
    "a null parameter": _run(
        lambda: (
            pl.col("img")
            .cv.pipe(_src().resize(height=pl.col("h"), width=4))
            .sink("tensor")
        )
    ),
    "a row in a multi-output graph": _run(
        lambda: pl.col("junk").cv.pipe(_src()).alias("x").sink({"x": "tensor"})
    ),
    "an unknown sink": _run(lambda: pl.col("img").cv.pipe(_src()).sink("nunpy")),
    "an option of another sink": _run(
        lambda: pl.col("img").cv.pipe(_src()).sink("png", quality=3)
    ),
    "a sink option of the wrong type": _run(
        lambda: pl.col("img").cv.pipe(_src()).sink("jpeg", quality="high")
    ),
    "a sink option out of range": _run(
        lambda: pl.col("img").cv.pipe(_src()).sink("jpeg", quality=1000)
    ),
    "an output alias the graph does not have": _run(
        lambda: pl.col("img").cv.pipe(_src()).alias("x").sink({"y": "tensor"})
    ),
    "a sink the planned dtype cannot take": _run(
        lambda: pl.col("img").cv.pipe(_src().cast("f32")).sink("jpeg")
    ),
    "a pipeline with no source": _run(
        lambda: (
            pl.col("img").cv.pipe(Pipeline().resize(height=4, width=4)).sink("tensor")
        )
    ),
    "an unknown source": lambda: Pipeline().source("image_bites"),
    "a source option of the wrong type": lambda: Pipeline().source(
        "image_bytes",
        decode_max_size="big",  # ty: ignore[invalid-argument-type]
    ),
    "an unknown dtype": lambda: _src().cast("float32"),
    "an op the planner refuses": lambda: _src().blur(-1.0),
    "an op in the wrong domain": lambda: (
        _src().grayscale().extract_contours().blur(1.0)
    ),
    "a geometry call the definition refuses": _run(
        lambda: pl.col("poly").contour.simplify(tolerance="loose")  # ty: ignore[invalid-argument-type]
    ),
}


@plugin_required
@pytest.mark.parametrize("name", sorted(FAILURES))
def test_a_deliberate_failure_raises_a_clean_message(name: str) -> None:
    with pytest.raises(Exception) as raised:  # noqa: PT011 - every kind
        FAILURES[name]()
    message = own_words(raised.value)
    assert message.strip(), "an empty message"
    assert leaks(message) == [], message
