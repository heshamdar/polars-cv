"""ROI decode: a crop right after an image source decodes only its window.

The ``roi_decode`` engine pass hands a node's leading crop to the decoder
(``ImageAdapter::decode_cropped``) instead of decoding the whole image and
cropping after. It must never change a result: every case here runs with the
pass on and off and compares whole output columns — values, nulls and error
messages — at the user-facing entry point (``.sink()``).

"The same result" is the array a ``numpy`` row holds (dtype, shape, values),
not its zero-copy layout: with the pass off a crop is a strided view into the
decoded image, with it on the window may be decoded (or cast) contiguously, so
the strides the struct carries may differ while every element is the same.
Encoded sinks (``png``, ``blob``) are compared byte for byte. Whether it actually
fires is pinned by the Rust tests in ``graph/compiled.rs``, which count the rows
it decoded.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import OptFlags, Pipeline, numpy_from_struct
from polars_cv._optimize import PASS_NAMES
from tests.conftest import make_image_png, plugin_required

if TYPE_CHECKING:
    from pathlib import Path

_ON = OptFlags.all()
_OFF = OptFlags(**{**{n: True for n in PASS_NAMES}, "roi_decode": False})


def _outputs(
    df: pl.DataFrame, pipe: Pipeline, column: str = "b", fmt: str = "numpy"
) -> tuple[list[object], list[object]]:
    """The output column with the pass on, and with it off."""

    def run(flags: OptFlags) -> list[object]:
        expr = pl.col(column).cv.pipe(pipe).sink(fmt, opt_flags=flags)
        rows = df.select(out=expr)["out"].to_list()
        return [_logical(v) if fmt == "numpy" else v for v in rows]

    return run(_ON), run(_OFF)


def _logical(row: object) -> object:
    """A ``numpy`` row as its array's dtype, shape and packed bytes; a null
    row or a ``null_with_message`` struct's fields other than the array as
    they are."""
    if not isinstance(row, dict) or row.get("data") is None:
        return row
    arr = numpy_from_struct(row)
    rest = {k: v for k, v in row.items() if k not in ("data", "strides", "offset")}
    return (arr.dtype.str, arr.shape, np.ascontiguousarray(arr).tobytes(), rest)


def _jpeg(height: int, width: int) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


#: Images of every channel layout the alpha rules distinguish, plus 16-bit and
#: JPEG, at different sizes so per-row windows land differently on each.
def _images() -> list[bytes]:
    return [
        make_image_png(20, 24, channels=1, seed=1),
        make_image_png(22, 20, channels=2, seed=2),
        make_image_png(24, 26, channels=3, seed=3),
        make_image_png(21, 23, channels=4, seed=4),
        make_image_png(20, 20, sixteen_bit=True, seed=5),
        _jpeg(24, 24),
    ]


_CASES: list[tuple[str, Callable[[Pipeline], Pipeline]]] = [
    ("static", lambda p: p.crop(top=3, left=2, height=10, width=12)),
    ("to_the_edges", lambda p: p.crop(top=5, left=7)),
    ("whole_image_origin", lambda p: p.crop(top=0, left=0, height=20, width=20)),
    ("empty", lambda p: p.crop(top=4, left=4, height=0, width=3)),
    (
        "per_row",
        lambda p: p.crop(
            top=pl.col("t"), left=pl.col("l"), height=pl.col("h"), width=pl.col("w")
        ),
    ),
    # Pushdown moves the crop ahead of the pointwise op, to the source.
    ("after_pointwise", lambda p: p.invert().crop(top=2, left=3, height=9, width=8)),
    ("then_more_ops", lambda p: p.crop(top=1, left=1, height=12, width=12).flip_h()),
]


def _frame() -> pl.DataFrame:
    images = _images()
    n = len(images)
    return pl.DataFrame(
        {
            "b": images,
            "t": [i % 4 for i in range(n)],
            "l": [(3 * i) % 5 for i in range(n)],
            "h": [5 + i for i in range(n)],
            "w": [9 - i for i in range(n)],
        }
    )


@plugin_required
class TestRoiDecodeEquivalence:
    """The pass on and off give the same column, case by case."""

    @pytest.mark.parametrize(("case", "build"), _CASES, ids=[c[0] for c in _CASES])
    @pytest.mark.parametrize("dtype", [None, "f32"])
    def test_image_bytes(
        self, case: str, build: Callable[[Pipeline], Pipeline], dtype: str | None
    ) -> None:
        """Every image of the frame, through an ``image_bytes`` source, with
        and without a declared dtype (a cast after the decode)."""
        pipe = build(Pipeline().source("image_bytes", dtype=dtype))
        on, off = _outputs(_frame(), pipe)
        assert any(v is not None for v in off), f"{case}: no output at all"
        assert on == off, case

    @pytest.mark.parametrize(("case", "build"), _CASES, ids=[c[0] for c in _CASES])
    def test_file_path(
        self, tmp_path: Path, case: str, build: Callable[[Pipeline], Pipeline]
    ) -> None:
        """The same through a ``file_path`` source."""
        df = _frame()
        paths = []
        for i, data in enumerate(df["b"]):
            path = tmp_path / f"{i}.img"
            path.write_bytes(data)
            paths.append(str(path))
        df = df.with_columns(p=pl.Series(paths))
        pipe = build(Pipeline().source("file_path"))
        on, off = _outputs(df, pipe, column="p")
        assert on == off, case

    def test_auto_source(self) -> None:
        """An ``auto`` source routed to image bytes takes the pass too."""
        pipe = Pipeline().source().crop(top=2, left=2, height=6, width=6)
        on, off = _outputs(_frame(), pipe)
        assert on == off

    def test_byte_sinks(self) -> None:
        """Encoded outputs agree too (the window reaches the encoder as a
        view in one case and a decoded buffer's view in the other)."""
        pipe = Pipeline().source("image_bytes").crop(top=1, left=2, height=8, width=8)
        for fmt in ("png", "blob"):
            on, off = _outputs(_frame(), pipe, fmt=fmt)
            assert on == off, fmt

    @pytest.mark.parametrize("policy", ["null", "null_with_message"])
    def test_an_out_of_bounds_row_fails_the_same_way(self, policy: str) -> None:
        """A window outside one row's image is that row's error, with the
        crop's own message, whether or not the pass is on."""
        df = _frame().with_columns(
            t=pl.when(pl.int_range(pl.len()) == 2).then(100).otherwise(0)
        )
        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=pl.col("t"), left=0, height=4, width=4)
            .on_error(policy)
        )
        on, off = _outputs(df, pipe)
        assert on == off
        assert on[2] is None or "window" in str(on[2])

    def test_an_out_of_bounds_window_raises_the_same_error(self) -> None:
        """Under ``raise`` the query fails, with the same message."""
        pipe = Pipeline().source("image_bytes").crop(top=100, left=0, height=4, width=4)
        messages = []
        for flags in (_ON, _OFF):
            expr = pl.col("b").cv.pipe(pipe).sink("numpy", opt_flags=flags)
            with pytest.raises(pl.exceptions.ComputeError) as excinfo:
                _frame().select(expr)
            messages.append(str(excinfo.value))
        assert messages[0] == messages[1]
        assert "window" in messages[0]

    def test_a_null_parameter_nulls_the_same_rows(self) -> None:
        """Under ``on_null_param("null")`` a null window parameter nulls its
        row with the pass on and off alike."""
        df = _frame().with_columns(
            t=pl.when(pl.int_range(pl.len()) % 2 == 0).then(pl.col("t")).otherwise(None)
        )
        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=pl.col("t"), left=0, height=4, width=4)
            .on_null_param("null")
        )
        on, off = _outputs(df, pipe)
        assert on == off
        assert [v is None for v in on] == [i % 2 == 1 for i in range(len(on))]

    def test_a_corrupt_image_fails_the_same_way(self) -> None:
        """A row that does not decode is the decoder's error either way."""
        df = _frame().with_columns(
            b=pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.lit(b"not an image"))
            .otherwise(pl.col("b"))
        )
        pipe = (
            Pipeline()
            .source("image_bytes", on_error="null")
            .crop(top=0, left=0, height=4, width=4)
        )
        on, off = _outputs(df, pipe)
        assert on == off
        assert on[1] is None

    def test_decode_max_size_still_scales_then_crops(self) -> None:
        """A scaled decode changes coordinates, so the pass leaves it alone;
        the result is unchanged."""
        pipe = (
            Pipeline()
            .source("image_bytes", decode_max_size=8)
            .crop(top=1, left=1, height=4, width=4)
        )
        on, off = _outputs(_frame(), pipe)
        assert on == off
