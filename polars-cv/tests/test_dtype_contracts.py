"""
Tests for the operation contract system and dtype preservation.

Verifies that:
- The planner (sourcing dtype from view-buffer's per-op contract) tracks dtypes
  through pipelines.
- Resize operations preserve input dtype (the core bug fix).
- The Rust execution layer honours the planned dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def test_image_png(encode_png: Callable[[np.ndarray], bytes]) -> bytes:
    """A 64x64 RGB test image as PNG bytes."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return encode_png(img)


# ---------------------------------------------------------------------------
# 1. Pipeline output_dtype inference (pure Python, no plugin)
# ---------------------------------------------------------------------------


class TestPipelineOutputDtype:
    """Verify that Pipeline.output_dtype tracks dtype correctly."""

    def test_image_source_resize_stays_auto(self) -> None:
        """image_bytes source -> resize should keep auto (preserve propagates auto)."""
        pipe = Pipeline().source("image_bytes").resize(height=224, width=224)
        assert pipe.output_dtype() == "auto"

    def test_list_f32_source_resize_preserves_f32(self) -> None:
        """list source with dtype=f32 -> resize should keep f32."""
        pipe = Pipeline().source("list", dtype="f32").resize(height=224, width=224)
        assert pipe.output_dtype() == "f32"

    def test_list_u16_source_resize_preserves_u16(self) -> None:
        """list source with dtype=u16 -> resize should keep u16."""
        pipe = Pipeline().source("list", dtype="u16").resize(height=224, width=224)
        assert pipe.output_dtype() == "u16"

    def test_list_f64_source_resize_preserves_f64(self) -> None:
        """list source with dtype=f64 -> resize should keep f64."""
        pipe = Pipeline().source("list", dtype="f64").resize(height=100, width=100)
        assert pipe.output_dtype() == "f64"

    def test_resize_scale_preserves_dtype(self) -> None:
        """resize_scale should also preserve dtype."""
        pipe = Pipeline().source("list", dtype="f32").resize_scale(scale=0.5)
        assert pipe.output_dtype() == "f32"

    def test_resize_to_height_preserves_dtype(self) -> None:
        """resize_to_height should preserve dtype."""
        pipe = Pipeline().source("list", dtype="f32").resize_to_height(224)
        assert pipe.output_dtype() == "f32"

    def test_resize_to_width_preserves_dtype(self) -> None:
        """resize_to_width should preserve dtype."""
        pipe = Pipeline().source("list", dtype="f32").resize_to_width(224)
        assert pipe.output_dtype() == "f32"

    def test_resize_max_preserves_dtype(self) -> None:
        """resize_max should preserve dtype."""
        pipe = Pipeline().source("list", dtype="f32").resize_max(224)
        assert pipe.output_dtype() == "f32"

    def test_resize_min_preserves_dtype(self) -> None:
        """resize_min should preserve dtype."""
        pipe = Pipeline().source("list", dtype="f32").resize_min(224)
        assert pipe.output_dtype() == "f32"

    def test_cast_then_resize_preserves_cast_dtype(self) -> None:
        """cast('f32') -> resize should keep f32."""
        pipe = (
            Pipeline().source("image_bytes").cast("f32").resize(height=224, width=224)
        )
        assert pipe.output_dtype() == "f32"

    def test_grayscale_preserves_dtype(self) -> None:
        """Grayscale should preserve input dtype (channel reduction only)."""
        pipe_f32 = Pipeline().source("list", dtype="f32").grayscale()
        assert pipe_f32.output_dtype() == "f32"

        # image_bytes starts as "auto"; grayscale (PRESERVE) keeps it "auto"
        pipe_auto = Pipeline().source("image_bytes").grayscale()
        assert pipe_auto.output_dtype() == "auto"

        pipe_u16 = Pipeline().source("list", dtype="u16").grayscale()
        assert pipe_u16.output_dtype() == "u16"

    def test_rotate_preserves_dtype(self) -> None:
        """Rotate should preserve input dtype (spatial transformation)."""
        pipe_f32 = Pipeline().source("list", dtype="f32").rotate(45.0)
        assert pipe_f32.output_dtype() == "f32"

        # image_bytes starts as "auto"; rotate (PRESERVE) keeps it "auto"
        pipe_auto = Pipeline().source("image_bytes").rotate(90.0)
        assert pipe_auto.output_dtype() == "auto"

        pipe_f64 = Pipeline().source("list", dtype="f64").rotate(30.0)
        assert pipe_f64.output_dtype() == "f64"

    def test_threshold_stays_u8_regardless_of_input(self) -> None:
        """Threshold should always produce u8 output (binary mask)."""
        pipe_u8 = Pipeline().source("image_bytes").grayscale().threshold(128)
        assert pipe_u8.output_dtype() == "u8"

        pipe_f32 = Pipeline().source("list", dtype="f32").grayscale().threshold(0.5)
        assert pipe_f32.output_dtype() == "u8"

        pipe_f64 = Pipeline().source("list", dtype="f64").grayscale().threshold(0.5)
        assert pipe_f64.output_dtype() == "u8"

    def test_normalize_defaults_f32(self) -> None:
        """Normalize should default to f32."""
        pipe = Pipeline().source("image_bytes").normalize()
        assert pipe.output_dtype() == "f32"

    def test_scale_promotes_to_float(self) -> None:
        """Scale on auto input should stay auto (PROMOTE_TO_FLOAT on auto)."""
        pipe = Pipeline().source("image_bytes").scale(1.0 / 255.0)
        assert pipe.output_dtype() == "auto"

    def test_scale_promotes_u8_to_f32(self) -> None:
        """Scale on known u8 input should promote to f32."""
        pipe = Pipeline().source("image_bytes", dtype="u8").scale(1.0 / 255.0)
        assert pipe.output_dtype() == "f32"


# ---------------------------------------------------------------------------
# 3. End-to-end execution tests (require compiled plugin)
# ---------------------------------------------------------------------------


@plugin_required
class TestResizeDtypePreservationE2E:
    """End-to-end tests verifying resize preserves dtype at execution time."""

    def test_resize_u8_from_image_bytes(self, test_image_png: bytes) -> None:
        """Resize from decoded image should stay u8."""
        from polars_cv import numpy_from_struct

        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").resize(height=32, width=32)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.uint8
        assert arr.shape == (32, 32, 3)

    def test_resize_f32_from_list_source(self) -> None:
        """Resize from f32 list source should produce f32 output."""
        from polars_cv import numpy_from_struct

        # Create a 4x4x3 f32 array as nested lists
        rng = np.random.default_rng(123)
        data = rng.random((4, 4, 3), dtype=np.float32)

        # Encode as nested Polars lists
        flat = data.flatten().tolist()
        rows = [
            [[flat[i * 12 + j * 3 + k] for k in range(3)] for j in range(4)]
            for i in range(4)
        ]

        df = pl.DataFrame({"buf": [rows]})
        pipe = Pipeline().source("list", dtype="f32").resize(height=2, width=2)
        result = df.select(out=pl.col("buf").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.float32, (
            f"Expected float32 output from f32 resize, got {arr.dtype}"
        )
        assert arr.shape == (2, 2, 3)

    def test_resize_u8_stays_u8(self) -> None:
        """Resize from u8 list source should produce u8 output."""
        from polars_cv import numpy_from_struct

        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)

        flat = data.flatten().tolist()
        rows = [
            [[flat[i * 12 + j * 3 + k] for k in range(3)] for j in range(4)]
            for i in range(4)
        ]

        df = pl.DataFrame({"buf": [rows]})
        pipe = Pipeline().source("list", dtype="u8").resize(height=2, width=2)
        result = df.select(out=pl.col("buf").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.uint8
        assert arr.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# 4. End-to-end tests for rotate / grayscale / threshold generalization
# ---------------------------------------------------------------------------


def _make_f32_3ch_list_df(h: int = 4, w: int = 4, seed: int = 42) -> pl.DataFrame:
    """Create a DataFrame with a single row containing a 3-channel f32 nested list.

    Args:
        h: Height of the array.
        w: Width of the array.
        seed: Random seed.

    Returns:
        DataFrame with column "buf" holding nested list data.
    """
    rng = np.random.default_rng(seed)
    data = rng.random((h, w, 3), dtype=np.float32)
    flat = data.flatten().tolist()
    rows = [
        [[flat[i * w * 3 + j * 3 + k] for k in range(3)] for j in range(w)]
        for i in range(h)
    ]
    return pl.DataFrame({"buf": [rows]})


def _make_f32_1ch_list_df(h: int = 4, w: int = 4, seed: int = 42) -> pl.DataFrame:
    """Create a DataFrame with a single row containing a 1-channel f32 nested list.

    Args:
        h: Height of the array.
        w: Width of the array.
        seed: Random seed.

    Returns:
        DataFrame with column "buf" holding nested list data (shape H x W).
    """
    rng = np.random.default_rng(seed)
    data = rng.random((h, w), dtype=np.float32)
    flat = data.flatten().tolist()
    rows = [[flat[i * w + j] for j in range(w)] for i in range(h)]
    return pl.DataFrame({"buf": [rows]})


@plugin_required
class TestRotateDtypePreservationE2E:
    """End-to-end tests verifying rotate preserves dtype at execution time."""

    def test_rotate_f32_preserves_dtype(self) -> None:
        """Rotate from f32 list source should produce f32 output."""
        from polars_cv import numpy_from_struct

        df = _make_f32_3ch_list_df(h=8, w=8)
        pipe = Pipeline().source("list", dtype="f32").rotate(45.0)
        result = df.select(out=pl.col("buf").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.float32, (
            f"Expected float32 output from f32 rotate, got {arr.dtype}"
        )

    def test_rotate_u8_stays_u8(self, test_image_png: bytes) -> None:
        """Rotate from u8 image source should produce u8 output."""
        from polars_cv import numpy_from_struct

        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").rotate(30.0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.uint8


@plugin_required
class TestGrayscaleDtypePreservationE2E:
    """End-to-end tests verifying grayscale preserves dtype at execution time."""

    def test_grayscale_f32_preserves_dtype(self) -> None:
        """Grayscale from f32 list source should produce f32 output."""
        from polars_cv import numpy_from_struct

        df = _make_f32_3ch_list_df(h=4, w=4)
        pipe = Pipeline().source("list", dtype="f32").grayscale()
        result = df.select(out=pl.col("buf").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.float32, (
            f"Expected float32 output from f32 grayscale, got {arr.dtype}"
        )
        # Grayscale reduces channels to 1
        assert arr.shape[-1] == 1

    def test_grayscale_u8_stays_u8(self, test_image_png: bytes) -> None:
        """Grayscale from u8 image source should produce u8 output."""
        from polars_cv import numpy_from_struct

        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.uint8
        assert arr.shape[-1] == 1


@plugin_required
class TestThresholdDtypeE2E:
    """End-to-end tests verifying threshold accepts any dtype and produces u8."""

    def test_threshold_f32_input_produces_u8(self) -> None:
        """Threshold from f32 single-channel source should produce u8 binary mask."""
        from polars_cv import numpy_from_struct

        df = _make_f32_1ch_list_df(h=4, w=4)
        pipe = Pipeline().source("list", dtype="f32").threshold(0.5)
        result = df.select(out=pl.col("buf").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.uint8, (
            f"Expected uint8 output from threshold, got {arr.dtype}"
        )
        # All values should be 0 or 255
        unique_vals = set(arr.flatten().tolist())
        assert unique_vals.issubset({0, 255}), (
            f"Threshold output should only contain 0 and 255, got {unique_vals}"
        )

    def test_threshold_u8_input_still_works(self, test_image_png: bytes) -> None:
        """Threshold from u8 image source should produce u8 binary mask."""
        from polars_cv import numpy_from_struct

        df = pl.DataFrame({"img": [test_image_png]})
        pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.dtype == np.uint8
        unique_vals = set(arr.flatten().tolist())
        assert unique_vals.issubset({0, 255})


@plugin_required
class TestF64ScalarOpsEndToEnd:
    """f64 data through the PromoteToFloat scalar family must stay f64.

    These ops previously computed everything in f32 and returned f32 for
    f64 inputs, diverging from the declared dtype contract — a user with
    f64 data hit the per-row dtype guard ("planned f64 but produced f32").
    The runtime now honors the contract: plan == exec, values in f64.
    """

    # A value that loses precision when round-tripped through f32.
    PRECISE = 0.12345678901234567

    def _run(self, build_pipe) -> pl.DataFrame:
        # 3D [H, W, C] data: the planner assumes list sources are 3D buffers.
        df = pl.DataFrame(
            {"x": [[[[self.PRECISE]], [[0.5]]]]},
            schema={"x": pl.List(pl.List(pl.List(pl.Float64)))},
        )
        pipe = build_pipe(Pipeline().source("list", dtype="f64"))
        lf = df.lazy().with_columns(out=pl.col("x").cv.pipe(pipe).sink("list"))
        planned = lf.collect_schema()["out"]
        out = lf.collect()
        assert planned == out["out"].dtype, "plan != exec for f64 scalar op"
        return out

    def test_scale_keeps_f64_precision(self) -> None:
        out = self._run(lambda p: p.scale(2.0))
        assert out["out"].dtype == pl.List(pl.List(pl.List(pl.Float64)))
        val = out["out"][0][0][0][0]
        assert val == self.PRECISE * 2.0
        # The old f32 path would have collapsed the low-order bits.
        assert val != float(np.float32(self.PRECISE) * np.float32(2.0))

    def test_relu_keeps_f64(self) -> None:
        out = self._run(lambda p: p.relu())
        assert out["out"][0][0][0][0] == self.PRECISE

    def test_clamp_keeps_f64(self) -> None:
        out = self._run(lambda p: p.clamp(min_val=0.0, max_val=1.0))
        assert out["out"][0][0][0][0] == self.PRECISE

    def test_adjust_gamma_keeps_f64(self) -> None:
        out = self._run(lambda p: p.adjust_gamma(gamma=2.0))
        assert out["out"][0][0][0][0] == self.PRECISE**2.0

    def test_adjust_contrast_keeps_f64(self) -> None:
        out = self._run(lambda p: p.adjust_contrast(factor=1.0))
        # factor=1.0 is the identity: (x - mean) * 1 + mean == x.
        assert out["out"][0][0][0][0] == self.PRECISE


@plugin_required
class TestNormalizeOutDtypeContract:
    """normalize(out_dtype=...) must satisfy plan == production.

    Regression guard for the one op with a ``Configurable`` dtype rule: the
    planner honored ``out_dtype`` (declaring, e.g., u8) but execution built
    ``Normalize`` with no dtype and always produced f32, so any ``out_dtype`` !=
    f32 tripped the runtime guard ("planned <X> but execution produced Float32").
    Execution now casts the f32 result to the configured dtype, so plan == exec.
    This closes the hole in the dtype-contract suite (it never exercised the
    ``out_dtype`` path).
    """

    # 3-D [H, W, 1] single-channel buffer so min/max normalize applies.
    HW1_F32 = {
        "x": [[[[0.0], [85.0]], [[170.0], [255.0]]]],
    }
    SCHEMA = {"x": pl.List(pl.List(pl.List(pl.Float64)))}

    def _run(self, out_dtype: str) -> pl.DataFrame:
        df = pl.DataFrame(self.HW1_F32, schema=self.SCHEMA)
        pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .normalize(method="minmax", out_dtype=out_dtype)
        )
        lf = df.lazy().with_columns(out=pl.col("x").cv.pipe(pipe).sink("list"))
        planned = lf.collect_schema()["out"]
        out = lf.collect()
        assert planned == out["out"].dtype, (
            f"plan != exec for normalize(out_dtype={out_dtype!r}): "
            f"planned {planned}, produced {out['out'].dtype}"
        )
        return out

    def _inner_dtype(self, dt: pl.DataType) -> pl.DataType:
        while isinstance(dt, pl.List):
            dt = dt.inner
        return dt

    def test_default_is_f32(self) -> None:
        out = self._run("f32")
        assert self._inner_dtype(out["out"].dtype) == pl.Float32

    def test_out_dtype_f64(self) -> None:
        out = self._run("f64")
        assert self._inner_dtype(out["out"].dtype) == pl.Float64

    def test_out_dtype_u8(self) -> None:
        out = self._run("u8")
        assert self._inner_dtype(out["out"].dtype) == pl.UInt8

    def test_out_dtype_none_matches_default(self) -> None:
        # No out_dtype at all must also be consistent (f32).
        df = pl.DataFrame(self.HW1_F32, schema=self.SCHEMA)
        pipe = Pipeline().source("list", dtype="f32").normalize(method="minmax")
        lf = df.lazy().with_columns(out=pl.col("x").cv.pipe(pipe).sink("list"))
        planned = lf.collect_schema()["out"]
        out = lf.collect()
        assert planned == out["out"].dtype
        assert self._inner_dtype(out["out"].dtype) == pl.Float32

    def test_preset_method_honors_out_dtype(self) -> None:
        # The contract must hold for the preset method too (3-channel HWC).
        hwc = {"x": [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]]}  # [1, 2, 3]
        schema = {"x": pl.List(pl.List(pl.List(pl.Float64)))}
        df = pl.DataFrame(hwc, schema=schema)
        pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .normalize(
                method="preset",
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                out_dtype="f64",
            )
        )
        lf = df.lazy().with_columns(out=pl.col("x").cv.pipe(pipe).sink("list"))
        planned = lf.collect_schema()["out"]
        out = lf.collect()
        assert planned == out["out"].dtype
        assert self._inner_dtype(out["out"].dtype) == pl.Float64

    def test_out_dtype_preserve_rejected(self) -> None:
        # "preserve" is not a dtype — `parse_dtype` has no such name — so it is
        # rejected by validating against DType rather than by a bespoke check
        # here. See `test_removed_surfaces.py` for why it existed at all.
        with pytest.raises(ValueError, match="preserve"):
            Pipeline().source("list", dtype="f32").normalize(
                method="minmax", out_dtype="preserve"
            )


@plugin_required
class TestScalarOpOutDtypeIsHonored:
    """``scale``/``clamp`` must land in the dtype ``out_dtype`` names.

    The parameter used to be accepted, validated, serialized into the op's
    params — so it entered the op's identity and the CSE / compiled-graph cache
    key — and then read by nobody: ``resolve_op``'s arms never looked at it, and
    ``output_dtype_for`` honours an override only for the ``Configurable`` rule,
    which is ``normalize``'s, not theirs (``PromoteToFloat``).

    That made it invisible to a plan-vs-exec test: plan and execution *agreed*,
    both reporting the promoted float. What was wrong was the value they agreed
    on. So these assert the dtype itself, on both halves of the promise.
    """

    #: A value that survives a u8 round trip exactly, so an integer target can
    #: be asserted on the value as well as the dtype.
    SRC = 3.0

    def _run(self, build_pipe, src_dtype: str = "f64") -> pl.DataFrame:
        df = pl.DataFrame(
            {"x": [[[[self.SRC]], [[1.0]]]]},
            schema={"x": pl.List(pl.List(pl.List(pl.Float64)))},
        )
        pipe = build_pipe(Pipeline().source("list", dtype=src_dtype))
        lf = df.lazy().with_columns(out=pl.col("x").cv.pipe(pipe).sink("list"))
        planned = lf.collect_schema()["out"]
        out = lf.collect()
        assert planned == out["out"].dtype, f"plan {planned} != exec {out['out'].dtype}"
        return out

    @pytest.mark.parametrize(
        ("out_dtype", "leaf"),
        [
            ("u8", pl.UInt8),
            ("i32", pl.Int32),
            ("f32", pl.Float32),
            ("f64", pl.Float64),
        ],
    )
    def test_scale_out_dtype(self, out_dtype: str, leaf: pl.DataType) -> None:
        out = self._run(lambda p: p.scale(2.0, out_dtype=out_dtype))
        assert self._inner_dtype(out["out"].dtype) == leaf
        assert out["out"][0][0][0][0] == self.SRC * 2.0

    @pytest.mark.parametrize(
        ("out_dtype", "leaf"),
        [("u8", pl.UInt8), ("i32", pl.Int32), ("f32", pl.Float32)],
    )
    def test_clamp_out_dtype(self, out_dtype: str, leaf: pl.DataType) -> None:
        out = self._run(lambda p: p.clamp(0.0, 2.0, out_dtype=out_dtype))
        assert self._inner_dtype(out["out"].dtype) == leaf
        # 3.0 clamped to [0, 2] is 2.0, in whatever dtype was asked for.
        assert out["out"][0][0][0][0] == 2.0

    @staticmethod
    def _inner_dtype(dtype: pl.DataType) -> pl.DataType:
        while isinstance(dtype, pl.List):
            dtype = dtype.inner
        return dtype

    def test_out_dtype_does_not_downgrade_the_default(self) -> None:
        """Omitting ``out_dtype`` must still promote-to-float, preserving f64.

        The alternative fix — giving the ops a ``Configurable(F32)`` rule like
        ``normalize``'s — would have silently turned f64 input into f32 here.
        """
        out = self._run(lambda p: p.scale(2.0))
        assert self._inner_dtype(out["out"].dtype) == pl.Float64

    def test_preserve_dtype_and_out_dtype_reach_the_same_place(self) -> None:
        """The two spellings are one mechanism, so they must agree.

        ``preserve_dtype=True`` names the target by reading the pipeline's
        pre-op dtype; ``out_dtype`` names it outright. Given the same target
        they must produce the same column.
        """
        explicit = self._run(lambda p: p.scale(2.0, out_dtype="f64"))
        derived = self._run(lambda p: p.scale(2.0, preserve_dtype=True))
        assert explicit["out"].dtype == derived["out"].dtype
        assert explicit["out"].to_list() == derived["out"].to_list()

    def test_integer_target_rounds_and_saturates(self) -> None:
        """The cast is the real one, not a reinterpretation.

        ``scale`` computes in f32/f64 and the result is cast round-then-
        saturate, matching what ``preserve_dtype`` has always done and what
        ``normalize(out_dtype=...)`` does in the runner.
        """
        df = pl.DataFrame(
            {"x": [[[[200.0]], [[-5.0]], [[2.6]]]]},
            schema={"x": pl.List(pl.List(pl.List(pl.Float64)))},
        )
        pipe = Pipeline().source("list", dtype="f64").scale(2.0, out_dtype="u8")
        out = (
            df.lazy().with_columns(out=pl.col("x").cv.pipe(pipe).sink("list")).collect()
        )
        vals = [row[0][0] for row in out["out"][0]]
        assert vals == [255, 0, 5], f"expected saturate/round, got {vals}"


@plugin_required
class TestGrayscaleFusionDtypeRegression:
    """Grayscale must not mistrack its dtype and poison downstream fusion.

    ``ViewExpr::grayscale()`` used to hardcode a tracked dtype of ``u8``,
    contradicting its ``PreserveInput`` contract. On normalized ``f32`` data a
    fused block after grayscale (``grayscale -> invert -> scale``) then read the
    mistracked ``u8`` and inverted as ``255 - x`` instead of ``1 - x`` — a
    silent, ~254-off wrong result that the (correct) published schema never
    revealed. Fixed by making ``apply_op`` the single metadata authority.
    """

    @staticmethod
    def _nested_f32(data: np.ndarray) -> list:
        """A single (H, W, C) f32 array as nested Polars lists for ``list``."""
        return data.tolist()

    def test_fused_invert_after_grayscale_matches_unfused(self) -> None:
        from polars_cv import numpy_from_struct

        # 2x2x3 image already normalized to [0, 1].
        data = np.array(
            [
                [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
                [[0.70, 0.80, 0.90], [0.15, 0.25, 0.35]],
            ],
            dtype=np.float32,
        )
        df = pl.DataFrame({"buf": [self._nested_f32(data)]})

        # Unfused: invert's operand is the grayscale (image) node, so no fusion.
        unfused_pipe = Pipeline().source("list", dtype="f32").grayscale().invert()
        # Fused: the trailing scale pulls invert into a fused compute block whose
        # input dtype is read from the grayscale node's tracked dtype.
        fused_pipe = (
            Pipeline().source("list", dtype="f32").grayscale().invert().scale(1.0)
        )

        unfused = numpy_from_struct(
            df.select(out=pl.col("buf").cv.pipe(unfused_pipe).sink("numpy")).row(0)[0]
        )
        fused = numpy_from_struct(
            df.select(out=pl.col("buf").cv.pipe(fused_pipe).sink("numpy")).row(0)[0]
        )

        assert unfused.dtype == np.float32
        assert fused.dtype == np.float32
        # Fusion must not change the values.
        np.testing.assert_allclose(fused, unfused, atol=1e-5)
        # And the correct inverted value of normalized data is 1 - gray, in [0, 1]
        # — not 255 - gray.
        assert unfused.min() >= 0.0 and unfused.max() <= 1.0, (
            f"invert on normalized f32 left [0, 1]: range "
            f"[{unfused.min()}, {unfused.max()}]"
        )
