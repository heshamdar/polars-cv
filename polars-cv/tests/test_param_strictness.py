"""End-to-end coverage of enum-valued operation parameters.

Every value of every user-facing enum must execute through the Rust plugin.
This binds together the canonical ``NAMED`` tables (view-buffer), the
executor's parameter parsers, and the actual kernels — a renamed or
mis-tabled variant fails here, not in a user's pipeline.

Invalid values are rejected at two independent layers, each with its own
tests: the Python builders raise ``ValueError`` (builder unit tests), and the
Rust executor rejects unknown strings / wrong types / out-of-range values
(``strict_param_tests`` in ``execute.rs``). This file also carries a source
ratchet asserting the two historic error-swallowing idioms never return to
``resolve_op``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import (
    ColorSpace,
    DType,
    FilterType,
    HashAlgorithm,
    HistogramOutput,
    NormalizeMethod,
    PadMode,
    PadPosition,
)
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from typing import Callable


def _run(pipe: Pipeline, sink: str, image_bytes: bytes) -> pl.Series:
    """Execute a one-row pipeline and return the output column."""
    df = pl.DataFrame({"image": [image_bytes]})
    out = df.with_columns(result=pl.col("image").cv.pipe(pipe).sink(sink))
    series = out["result"]
    assert series.null_count() == 0, f"{sink} sink produced null output"
    return series


@plugin_required
class TestEnumValuesExecutable:
    """Every Python-exposed enum value executes end-to-end."""

    @pytest.fixture()
    def image_bytes(self, create_test_png: "Callable") -> bytes:
        return create_test_png(16, 12)

    @pytest.mark.parametrize("filter_type", [f.value for f in FilterType])
    def test_resize_filters(self, image_bytes: bytes, filter_type: str) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=8, width=8, filter=filter_type)
        )
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize("mode", [m.value for m in PadMode])
    def test_pad_modes(self, image_bytes: bytes, mode: str) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=2, bottom=2, left=2, right=2, mode=mode)
        )
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize("position", [p.value for p in PadPosition])
    def test_pad_positions(self, image_bytes: bytes, position: str) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad_to_size(height=32, width=32, position=position)
        )
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize("algorithm", [a.value for a in HashAlgorithm])
    def test_hash_algorithms(self, image_bytes: bytes, algorithm: str) -> None:
        # A hash is not directly sinkable today: the eager builder tracks
        # vector domain (rejecting blob/numpy sinks and even a chained
        # reduce_popcount), while execution produces a byte buffer (rejecting
        # the native sink) — and the lazy continuation path derives *buffer*
        # domain from the Rust contract, disagreeing with the eager builder.
        # Known plan!=exec/domain-tracking gap; consume the hash the way
        # hamming_distance does (lazy continuation into a scalar) until the
        # domain authority is unified.
        df = pl.DataFrame({"image": [image_bytes]})
        hashed = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").perceptual_hash(algorithm=algorithm)
        )
        expr = hashed.pipe(Pipeline().reduce_popcount()).sink("native")
        out = df.with_columns(result=expr)["result"]
        assert out.null_count() == 0

    @pytest.mark.parametrize("output", [o.value for o in HistogramOutput])
    def test_histogram_outputs(self, image_bytes: bytes, output: str) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=8, output=output)
        )
        # quantized stays an image buffer; buckets has a dedicated native
        # encoding; counts/normalized/edges are 1-D results read via the typed
        # "list" sink. (Their "native" sink is a known plan!=exec gap: the
        # executor leaves them as Buffer node outputs — to be fixed when the
        # histogram step becomes domain-aware in the GraphStep executor.)
        if output == "quantized":
            sink = "numpy"
        elif output == "buckets":
            sink = "native"
        else:
            sink = "list"
        _run(pipe, sink, image_bytes)

    @pytest.mark.parametrize("closed", ["left", "right"])
    def test_histogram_closed(self, image_bytes: bytes, closed: str) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=8, closed=closed, output="counts")
        )
        _run(pipe, "list", image_bytes)

    @pytest.mark.parametrize("method", ["minmax", "zscore", "preset"])
    def test_normalize_methods(self, image_bytes: bytes, method: str) -> None:
        assert {m.value for m in NormalizeMethod} == {"minmax", "zscore", "preset"}
        pipe = Pipeline().source("image_bytes")
        if method == "preset":
            pipe = pipe.normalize(
                method=method, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        else:
            pipe = pipe.normalize(method=method)
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize(
        "to_space", [c.value for c in ColorSpace if c.value != "rgb"]
    )
    def test_color_spaces(self, image_bytes: bytes, to_space: str) -> None:
        pipe = Pipeline().source("image_bytes").convert_color("rgb", to_space)
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize("dtype", [d.value for d in DType])
    def test_cast_dtypes(self, image_bytes: bytes, dtype: str) -> None:
        pipe = Pipeline().source("image_bytes").cast(dtype)
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
    def test_rotate_interpolations(
        self, image_bytes: bytes, interpolation: str
    ) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .rotate(angle=30.0, interpolation=interpolation)
        )
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize("border", ["replicate", "zero", "reflect"])
    def test_convolve_borders(self, image_bytes: bytes, border: str) -> None:
        kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .convolve2d(kernel=kernel, ksize=3, border=border)
        )
        _run(pipe, "numpy", image_bytes)

    @pytest.mark.parametrize("mode", ["external", "tree", "all"])
    def test_extract_contour_modes(self, image_bytes: bytes, mode: str) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(1)
            .extract_contours(mode=mode)
        )
        _run(pipe, "native", image_bytes)

    @pytest.mark.parametrize("method", ["none", "simple", "approx"])
    def test_extract_contour_methods(self, image_bytes: bytes, method: str) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(1)
            .extract_contours(method=method)
        )
        _run(pipe, "native", image_bytes)


class TestParamPolicyRatchet:
    """Source ratchet: the two historic error-swallowing idioms must never
    return to ``resolve_op``. The policy (absent optional -> default,
    present-but-invalid -> error) is implemented by ``params::get`` and
    behaviorally guarded by ``strict_param_tests`` in execute.rs; this scan
    only blocks the exact known-bad shortcuts."""

    def test_no_error_swallowing_in_resolve_op(self) -> None:
        execute_rs = Path(__file__).parent.parent / "src" / "execute.rs"
        src = execute_rs.read_text()
        assert ".resolve_usize(row_idx, ctx).ok()" not in src, (
            "resolve_op swallows a parameter resolution error into None; "
            "use params::get::maybe_usize instead"
        )
        assert ".resolve_usize(row_idx, ctx).unwrap_or(" not in src, (
            "resolve_op swallows a parameter resolution error into a default; "
            "use params::get::opt_* instead"
        )
