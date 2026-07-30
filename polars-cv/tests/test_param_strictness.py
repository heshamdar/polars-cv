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

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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
    ParamValue,
)
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from typing import Callable


def _patterned_png(width: int = 16, height: int = 12) -> bytes:
    """A deterministic non-uniform RGB image.

    ``create_test_png`` paints a solid colour, on which resampling filters,
    convolution kernels and channel permutations are all no-ops — a solid
    image cannot distinguish "the parameter varied per row" from "the
    parameter was silently dropped". These tests need real structure.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


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
        # perceptual_hash now produces the `vector` domain from its Rust
        # contract (GraphStep::PerceptualHash), agreeing between the eager
        # builder and the lazy continuation, so the 1-D u8 fingerprint sinks
        # directly through the typed "list" sink.
        pipe = Pipeline().source("image_bytes").perceptual_hash(algorithm=algorithm)
        _run(pipe, "list", image_bytes)

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


@plugin_required
class TestFillRangeParamsAcceptExpressions:
    """Numeric fill/range params flow through Rust's expression-capable
    resolvers, so the Python builders must accept a ``pl.Expr`` end-to-end
    (aligning with ``pad(value)`` / ``histogram(bins)``)."""

    @pytest.fixture()
    def image_bytes(self, create_test_png: "Callable") -> bytes:
        return create_test_png(16, 12)

    def test_rotate_border_value_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes], "bg": [128.0]})
        pipe = Pipeline().source("image_bytes").rotate(30, border_value=pl.col("bg"))
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        assert out["r"].null_count() == 0

    def test_warp_affine_border_value_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes], "bg": [64.0]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 2.0, 0.0, 1.0, 3.0],
                output_size=(16, 12),
                border_value=pl.col("bg"),
            )
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        assert out["r"].null_count() == 0

    def test_histogram_range_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes], "lo": [0.0], "hi": [255.0]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .histogram(bins=8, range=(pl.col("lo"), pl.col("hi")), output="counts")
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))
        assert out["r"].null_count() == 0

    def test_extract_contours_min_area_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes], "area": [1.0]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours(min_area=pl.col("area"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("native"))
        # Contour extraction may legitimately yield empty rows; just assert the
        # expression-backed threshold executed without error.
        assert out.height == 1


class TestStructuralParamsRejectExpressions:
    """Structural params fix output shape/rank at planning time and must be
    literals. Passing a ``pl.Expr`` is rejected at build time (the Rust
    resolver also rejects a bound slot, as defense-in-depth)."""

    def test_reduce_axis_rejects_expr(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").reduce_max(axis=pl.col("ax"))

    def test_reduce_argmax_axis_rejects_expr(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").reduce_argmax(axis=pl.col("ax"))

    def test_perceptual_hash_size_rejects_expr(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").perceptual_hash(hash_size=pl.col("hs"))

    def test_rotate_expand_rejects_expr(self) -> None:
        """``expand`` changes the output height/width, so it is structural."""
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").rotate(45, expand=pl.col("e"))

    def test_cast_dtype_rejects_expr(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").cast(pl.col("dt"))

    def test_normalize_method_rejects_expr(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").normalize(method=pl.col("m"))

    def test_histogram_output_rejects_expr(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").histogram(output=pl.col("o"))

    def test_transpose_axes_rejects_expr(self) -> None:
        with pytest.raises(TypeError, match="structural"):
            Pipeline().source("image_bytes").transpose(axes=[pl.col("a"), 1, 2])


@plugin_required
class TestListParamElementsAcceptExpressions:
    """List-valued params keep a structural *length* but per-row *values*.

    The element count fixes a kernel size or channel count at planning time;
    the coefficients themselves resolve per row through ``ParamValue::List``
    (the same encoding ``warp_affine``'s matrix has always used).
    """

    @pytest.fixture()
    def image_bytes(self) -> bytes:
        return _patterned_png()

    def test_convolve2d_kernel_elements_accept_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes, image_bytes], "k": [0.0, 1.0]})
        # Identity kernel in row 0 (centre 1, rest 0); row 1 adds neighbours.
        kernel = [pl.col("k")] * 4 + [1.0] + [pl.col("k")] * 4
        pipe = Pipeline().source("image_bytes").convolve2d(kernel, 3)
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_sharpen_strength_accepts_expr(self, image_bytes: bytes) -> None:
        """``sharpen`` builds its kernel from ``strength`` element-wise."""
        df = pl.DataFrame({"image": [image_bytes, image_bytes], "s": [0.0, 2.0]})
        pipe = Pipeline().source("image_bytes").sharpen(strength=pl.col("s"))
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        # strength=0 is the identity kernel, so row 0 differs from row 1.
        assert values[0] != values[1]

    def test_normalize_mean_std_elements_accept_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes, image_bytes], "m": [0.0, 0.5]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .normalize(
                method="preset",
                mean=[pl.col("m"), pl.col("m"), pl.col("m")],
                std=[1.0, 1.0, 1.0],
            )
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_channel_swap_order_elements_accept_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame(
            {
                "image": [image_bytes, image_bytes],
                "i": pl.Series("i", [0, 2], dtype=pl.Int64),
            }
        )
        pipe = Pipeline().source("image_bytes").channel_swap(order=[pl.col("i"), 1, 0])
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_convolve2d_rejects_a_non_square_kernel(self) -> None:
        """The kernel *length* is checkable even when ``ksize`` is dynamic."""
        with pytest.raises(ValueError, match="square of an odd number"):
            Pipeline().source("image_bytes").convolve2d([1.0] * 8, pl.col("k"))


@plugin_required
class TestEnumParamsAcceptExpressions:
    """Enums with no shape/rank/dtype effect resolve per row.

    Plan-time shape probing binds expression params to integer placeholders, so
    these also exercise ``ParamCtx::probe`` substituting the default — if that
    path were broken, building the pipeline would fail before execution.
    """

    @pytest.fixture()
    def image_bytes(self) -> bytes:
        return _patterned_png()

    def test_resize_filter_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame(
            {"image": [image_bytes, image_bytes], "f": ["nearest", "lanczos3"]}
        )
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=7, width=5, filter=pl.col("f"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_dynamic_filter_keeps_shape_known_at_plan_time(self) -> None:
        """A dynamic filter must not erase the plan-time shape.

        Only the *dimensions* determine output geometry, so probing with the
        default filter still yields exact shape hints.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=7, width=5, filter=pl.col("f"))
        )
        assert pipe._shape_hints.height == ParamValue(is_expr=False, value=7)
        assert pipe._shape_hints.width == ParamValue(is_expr=False, value=5)

    def test_rotate_interpolation_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame(
            {"image": [image_bytes, image_bytes], "i": ["nearest", "bilinear"]}
        )
        pipe = Pipeline().source("image_bytes").rotate(33, interpolation=pl.col("i"))
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_pad_mode_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame(
            {"image": [image_bytes, image_bytes], "m": ["constant", "edge"]}
        )
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad(top=2, bottom=2, left=2, right=2, mode=pl.col("m"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_pad_to_size_position_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame(
            {"image": [image_bytes, image_bytes], "p": ["center", "top-left"]}
        )
        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad_to_size(height=20, width=20, position=pl.col("p"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_unknown_enum_value_errors_at_execution(self, image_bytes: bytes) -> None:
        """A per-row enum cannot be validated at build time, so Rust does it."""
        df = pl.DataFrame({"image": [image_bytes], "f": ["not-a-filter"]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=8, width=8, filter=pl.col("f"))
        )
        with pytest.raises(Exception, match="unknown value"):
            df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))

    def test_wrong_dtype_for_enum_param_errors(self, image_bytes: bytes) -> None:
        """An integer column routed to an enum param must not silently default."""
        df = pl.DataFrame({"image": [image_bytes], "f": [3]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=8, width=8, filter=pl.col("f"))
        )
        with pytest.raises(Exception, match="String column"):
            df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))

    def test_literal_enum_still_validated_at_build_time(self) -> None:
        with pytest.raises(ValueError, match="Invalid filter"):
            Pipeline().source("image_bytes").resize(height=8, width=8, filter="bogus")


@plugin_required
class TestLetterboxFilterExposed:
    """``letterbox`` now exposes the ``filter`` Rust already read."""

    @pytest.fixture()
    def image_bytes(self) -> bytes:
        return _patterned_png()

    def test_filter_changes_output(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes]})
        near = (
            Pipeline()
            .source("image_bytes")
            .letterbox(height=32, width=32, filter="nearest")
        )
        lanc = Pipeline().source("image_bytes").letterbox(height=32, width=32)
        out = df.with_columns(
            a=pl.col("image").cv.pipe(near).sink("numpy"),
            b=pl.col("image").cv.pipe(lanc).sink("numpy"),
        )
        assert out["a"].to_list() != out["b"].to_list()

    def test_default_filter_is_lanczos3(self, image_bytes: bytes) -> None:
        """The historical default must not change now that it is exposed."""
        df = pl.DataFrame({"image": [image_bytes]})
        default = Pipeline().source("image_bytes").letterbox(height=32, width=32)
        explicit = (
            Pipeline()
            .source("image_bytes")
            .letterbox(height=32, width=32, filter="lanczos3")
        )
        out = df.with_columns(
            a=pl.col("image").cv.pipe(default).sink("numpy"),
            b=pl.col("image").cv.pipe(explicit).sink("numpy"),
        )
        assert out["a"].to_list() == out["b"].to_list()


@plugin_required
class TestRotateAndScaleAcceptsExpressions:
    """``rotate_and_scale`` builds its matrix from expression arithmetic."""

    @pytest.fixture()
    def image_bytes(self) -> bytes:
        return _patterned_png(16, 16)

    def test_angle_accepts_expr(self, image_bytes: bytes) -> None:
        df = pl.DataFrame({"image": [image_bytes, image_bytes], "a": [0.0, 90.0]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(angle=pl.col("a"), center=(8, 8), output_size=(16, 16))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]

    def test_literal_path_matches_expression_path(self, image_bytes: bytes) -> None:
        """The float and expression paths compute the same matrix."""
        df = pl.DataFrame({"image": [image_bytes], "a": [30.0]})
        lit = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(angle=30.0, center=(8, 8), output_size=(16, 16))
        )
        dyn = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(angle=pl.col("a"), center=(8, 8), output_size=(16, 16))
        )
        out = df.with_columns(
            a=pl.col("image").cv.pipe(lit).sink("numpy"),
            b=pl.col("image").cv.pipe(dyn).sink("numpy"),
        )
        assert out["a"].to_list() == out["b"].to_list()

    def test_expression_matrix_drops_out_of_affine_fusion(self) -> None:
        """Fusion needs concrete numbers to compose matrices.

        An expression matrix must fall back to executing as its own warp
        rather than being folded into a neighbouring affine.
        """
        from polars_cv.pipeline import _literal_matrix_values

        dyn = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(angle=pl.col("a"), center=(8, 8), output_size=(16, 16))
        )
        matrix_param = dyn._ops[-1].params["matrix"]
        assert _literal_matrix_values(matrix_param) is None

        lit = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(angle=30.0, center=(8, 8), output_size=(16, 16))
        )
        assert _literal_matrix_values(lit._ops[-1].params["matrix"]) is not None


@plugin_required
class TestContourSourceFillAcceptsExpressions:
    """Contour-source fill/background match the identical `rasterize` params."""

    def test_fill_value_accepts_expr(self) -> None:
        square = {
            "exterior": [
                {"x": 1.0, "y": 1.0},
                {"x": 8.0, "y": 1.0},
                {"x": 8.0, "y": 8.0},
                {"x": 1.0, "y": 8.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame({"c": [square, square], "fv": [100, 200]})
        pipe = Pipeline().source(
            "contour", width=10, height=10, fill_value=pl.col("fv")
        )
        out = df.with_columns(r=pl.col("c").cv.pipe(pipe).sink("numpy"))
        values = out["r"].to_list()
        assert values[0] != values[1]
