"""
Tests for affine transform pipeline builder operations.

Tests the Python-side pipeline construction for warp_affine, shear,
and rotate_and_scale. Most need no compiled plugin; the exceptions are the
``rotate_and_scale`` cases that actually build a matrix, which now read it from
the ``rotation_matrix_2d`` FFI (the single rotation-matrix authority) and so
carry ``@plugin_required``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from polars_cv import Pipeline
from tests._plan_view import ops_of, planned
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


class TestWarpAffinePipelineBuilder:
    """Tests for the warp_affine pipeline builder method."""

    def test_warp_affine_identity_matrix(self) -> None:
        """Build a pipeline with identity affine matrix."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
            )
        )
        assert len(ops_of(pipe)) == 1
        assert ops_of(pipe)[0].op == "warp_affine"

    def test_warp_affine_requires_6_elements(self) -> None:
        """Affine matrix must have exactly 6 elements."""
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="length 6"):
            pipe.warp_affine(matrix=[1.0, 0.0], output_size=(100, 100))

    def test_warp_affine_translation(self) -> None:
        """Build a pipeline with a translation matrix."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 50.0, 0.0, 1.0, 30.0],
                output_size=(224, 224),
            )
        )
        op = ops_of(pipe)[0]
        # Each matrix element is tracked on its own so it can be a per-row
        # expression; literal floats round-trip element by element.
        assert op.params["matrix"] == [
            1.0,
            0.0,
            50.0,
            0.0,
            1.0,
            30.0,
        ]
        # `output_size` is one field on the wire, as it is in the signature.
        assert op.params["output_size"] == [224, 224]

    def test_warp_affine_interpolation_default(self) -> None:
        """Default interpolation is bilinear."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
            )
        )
        assert ops_of(pipe)[0].params["interpolation"] == "bilinear"

    def test_warp_affine_nearest_interpolation(self) -> None:
        """Nearest-neighbor interpolation can be specified."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
                interpolation="nearest",
            )
        )
        assert ops_of(pipe)[0].params["interpolation"] == "nearest"

    def test_warp_affine_border_value(self) -> None:
        """Custom border value can be specified."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
                border_value=128.0,
            )
        )
        assert ops_of(pipe)[0].params["border_value"] == 128.0

    def test_warp_affine_domain_validation(self) -> None:
        """warp_affine requires buffer domain."""
        pipe = Pipeline().source("image_bytes").extract_contours()
        with pytest.raises(ValueError, match="buffer"):
            pipe.warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
            )

    def test_warp_affine_preserves_dtype(self) -> None:
        """warp_affine preserves the buffer dtype."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
            )
        )
        # auto dtype should remain auto (preserving)
        assert planned(pipe).dtype == "auto"

    def test_warp_affine_updates_shape_hints(self) -> None:
        """warp_affine updates shape hints to the output_size."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(224, 320),
            )
        )
        assert planned(pipe).height is not None
        assert planned(pipe).height == 224
        assert planned(pipe).width is not None
        assert planned(pipe).width == 320

    def test_warp_affine_chaining(self) -> None:
        """warp_affine can be chained with other operations."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=256, width=256)
            .warp_affine(
                matrix=[1.0, 0.0, 10.0, 0.0, 1.0, 10.0],
                output_size=(256, 256),
            )
            .normalize()
        )
        assert len(ops_of(pipe)) == 3
        assert ops_of(pipe)[1].op == "warp_affine"

    def test_warp_affine_immutability(self) -> None:
        """Pipeline is immutable — warp_affine returns a new instance."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = pipe1.warp_affine(
            matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            output_size=(100, 100),
        )
        assert len(ops_of(pipe1)) == 0
        assert len(ops_of(pipe2)) == 1


class TestShearPipelineBuilder:
    """Tests for the shear convenience method."""

    def test_shear_basic(self) -> None:
        """Shear builds a warp_affine op with shear matrix."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .shear(sx=0.2, sy=0.0, output_size=(100, 100))
        )
        assert len(ops_of(pipe)) == 1
        assert ops_of(pipe)[0].op == "warp_affine"
        matrix = ops_of(pipe)[0].params["matrix"]
        assert matrix == [1.0, 0.2, 0.0, 0.0, 1.0, 0.0]

    def test_shear_requires_output_size(self) -> None:
        """Shear requires output_size (a required keyword-only argument).

        The output shape is part of the plan-time schema and a shear does not
        imply one, so the signature enforces it rather than raising later.
        """
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(TypeError, match="output_size"):
            pipe.shear(sx=0.2, sy=0.0)

    def test_shear_both_axes(self) -> None:
        """Shear on both axes."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .shear(sx=0.3, sy=0.1, output_size=(200, 200))
        )
        matrix = ops_of(pipe)[0].params["matrix"]
        assert matrix == [1.0, 0.3, 0.0, 0.1, 1.0, 0.0]

    def test_shear_domain_validation(self) -> None:
        """Shear requires buffer domain (delegates to warp_affine)."""
        pipe = Pipeline().source("image_bytes").extract_contours()
        with pytest.raises(ValueError, match="buffer"):
            pipe.shear(sx=0.2, sy=0.0, output_size=(100, 100))


class TestRotateAndScalePipelineBuilder:
    """Tests for the rotate_and_scale convenience method."""

    @plugin_required
    def test_rotate_and_scale_basic(self) -> None:
        """Build a rotate_and_scale pipeline."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(
                angle=45.0,
                scale=1.0,
                center=(50.0, 50.0),
                output_size=(100, 100),
            )
        )
        assert len(ops_of(pipe)) == 1
        assert ops_of(pipe)[0].op == "warp_affine"

    def test_rotate_and_scale_requires_center(self) -> None:
        """rotate_and_scale requires center (a required keyword-only argument)."""
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(TypeError, match="center"):
            pipe.rotate_and_scale(angle=45.0, output_size=(100, 100))

    def test_rotate_and_scale_requires_output_size(self) -> None:
        """rotate_and_scale requires output_size (a required keyword-only argument)."""
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(TypeError, match="output_size"):
            pipe.rotate_and_scale(angle=45.0, center=(50.0, 50.0))

    @plugin_required
    def test_rotate_and_scale_matrix_correctness(self) -> None:
        """Verify the rotation matrix is correct for 90 degrees.

        Independent cross-check: the pipeline's matrix comes from the
        ``rotation_matrix_2d`` FFI, and this recomputes it from the reference
        formula in Python, so the two must agree.
        """
        import math

        pipe = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(
                angle=90.0,
                scale=1.0,
                center=(50.0, 50.0),
                output_size=(100, 100),
            )
        )
        matrix = ops_of(pipe)[0].params["matrix"]
        rad = math.radians(90.0)
        cos_a = math.cos(rad) * 1.0
        sin_a = math.sin(rad) * 1.0
        cx, cy = 50.0, 50.0
        tx = (1 - cos_a) * cx + sin_a * cy
        ty = -sin_a * cx + (1 - cos_a) * cy
        expected = [cos_a, -sin_a, tx, sin_a, cos_a, ty]
        for actual, exp in zip(matrix, expected):
            assert abs(actual - exp) < 1e-10

    @plugin_required
    def test_rotate_and_scale_with_scale(self) -> None:
        """rotate_and_scale with scale factor."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .rotate_and_scale(
                angle=0.0,
                scale=2.0,
                center=(50.0, 50.0),
                output_size=(200, 200),
            )
        )
        matrix = ops_of(pipe)[0].params["matrix"]
        # At angle=0 and scale=2: matrix should be [2, 0, -50, 0, 2, -50]
        assert abs(matrix[0] - 2.0) < 1e-10
        assert abs(matrix[4] - 2.0) < 1e-10


class TestRotateShapeHintTracking:
    """Shape-hint tracking for rotates whose effect is not plan-time known."""

    def test_expr_angle_expand_clears_hints(self) -> None:
        """Expression angle + expand=True -> output dims unknowable."""
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(pl.col("angle"), expand=True)
        )
        assert planned(pipe).height is None
        assert planned(pipe).width is None

    def test_expr_angle_non_square_clears_hints(self) -> None:
        """Expression angle on a non-square image: 90/270 would swap H/W,
        other angles keep them -> unknowable at plan time."""
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=50)
            .rotate(pl.col("angle"))
        )
        assert planned(pipe).height is None
        assert planned(pipe).width is None

    def test_expr_angle_square_non_expand_keeps_hints(self) -> None:
        """Square image, no expand: any angle keeps HxW."""
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(pl.col("angle"))
        )
        assert planned(pipe).height is not None
        assert planned(pipe).height == 100
        assert planned(pipe).width is not None
        assert planned(pipe).width == 100


class TestPerRowAffineParams:
    """Per-row expression params on warp_affine and shear (builder-level).

    Enables per-sample random affine/shear in a single batched call.
    """

    def test_warp_affine_accepts_expr_matrix_elements(self) -> None:
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[
                    pl.col("a"),
                    pl.col("b"),
                    pl.col("tx"),
                    pl.col("c"),
                    pl.col("d"),
                    pl.col("ty"),
                ],
                output_size=(64, 64),
            )
        )
        assert ops_of(pipe)[-1].op == "warp_affine"
        # Matrix is serialized as a list of 6 per-element ParamValue dicts.
        matrix = ops_of(pipe)[-1].params["matrix"]
        assert isinstance(matrix, list)
        assert len(matrix) == 6

    def test_warp_affine_mixed_literal_and_expr_matrix(self) -> None:
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, pl.col("b"), 0.0, 0.0, 1.0, pl.col("ty")],
                output_size=(64, 64),
            )
        )
        assert len(ops_of(pipe)[-1].params["matrix"]) == 6

    def test_warp_affine_still_rejects_wrong_length(self) -> None:
        import polars as pl

        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="length 6"):
            pipe.warp_affine(matrix=[pl.col("a"), 1.0], output_size=(10, 10))

    def test_shear_accepts_expr_factors(self) -> None:
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .shear(sx=pl.col("shear_x"), output_size=(64, 64))
        )
        # shear delegates to warp_affine; the matrix carries the expr element.
        assert ops_of(pipe)[-1].op == "warp_affine"
        assert len(ops_of(pipe)[-1].params["matrix"]) == 6


@plugin_required
class TestPerRowAffineExecution:
    """Per-row affine matrix resolves to a different transform per row."""

    def test_per_row_translation_differs(self) -> None:
        import numpy as np
        import polars as pl

        from polars_cv import numpy_from_struct

        # A [4, 4, 1] gradient buffer, two identical rows but different tx.
        img = [[[float(r * 4 + c)] for c in range(4)] for r in range(4)]
        df = pl.DataFrame(
            {
                "x": [img, img],
                "tx": [0.0, 2.0],
            },
            schema={
                "x": pl.List(pl.List(pl.List(pl.Float64))),
                "tx": pl.Float64,
            },
        )
        pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .warp_affine(
                matrix=[1.0, 0.0, pl.col("tx"), 0.0, 1.0, 0.0],
                output_size=(4, 4),
                interpolation="nearest",
            )
        )
        out = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy"))
            .collect()
        )
        a = numpy_from_struct(out["out"][0])
        b = numpy_from_struct(out["out"][1])
        # Different per-row tx => different outputs (a horizontal shift).
        assert not np.array_equal(a, b)

    def test_per_row_shear_differs(self) -> None:
        import numpy as np
        import polars as pl

        from polars_cv import numpy_from_struct

        img = [[[float(r * 4 + c)] for c in range(4)] for r in range(4)]
        df = pl.DataFrame(
            {"x": [img, img], "sx": [0.0, 0.8]},
            schema={
                "x": pl.List(pl.List(pl.List(pl.Float64))),
                "sx": pl.Float64,
            },
        )
        pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .shear(sx=pl.col("sx"), output_size=(4, 4))
        )
        out = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy"))
            .collect()
        )
        a = numpy_from_struct(out["out"][0])  # sx=0 -> identity
        b = numpy_from_struct(out["out"][1])  # sx=0.8 -> sheared
        assert not np.array_equal(a, b)

    def test_per_row_reshape_expr(self) -> None:
        # Regression: nested per-row params (reshape shape) must bind. Before the
        # recursive param binding this errored at compile ("unbound expression
        # parameter") because the op was misclassified as static.
        import numpy as np
        import polars as pl

        from polars_cv import numpy_from_struct

        flat = [float(i) for i in range(6)]
        df = pl.DataFrame(
            {"x": [flat], "h": [2], "w": [3]},
            schema={"x": pl.List(pl.Float64), "h": pl.Int64, "w": pl.Int64},
        )
        pipe = (
            Pipeline().source("list", dtype="f32").reshape([pl.col("h"), pl.col("w")])
        )
        out = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy"))
            .collect()
        )
        arr = numpy_from_struct(out["out"][0])
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr.astype(np.float64).ravel(), flat)


@plugin_required
class TestAffineRankContract:
    """An affine warp's output rank must match its declared `shape`.

    `ComputeOp::Affine`'s `shape` replaces H and W and leaves the rest of the
    input shape alone, so a `[H, W, 1]` input must stay 3-D. The kernel used to
    collapse any single-channel result to `[H, W]`, which the runtime rank guard
    only caught once plan-time rank folding started reporting a rank at all for
    list sources.
    """

    IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    def _warp(self, img: list, schema: Any) -> tuple:
        import polars as pl

        from polars_cv import numpy_from_struct

        df = pl.DataFrame({"x": [img]}, schema={"x": schema})
        pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .warp_affine(matrix=self.IDENTITY, output_size=(2, 2))
        )
        lf = df.lazy().with_columns(out=pl.col("x").cv.pipe(pipe).sink("numpy"))
        planned = (
            df.lazy()
            .with_columns(out=pl.col("x").cv.pipe(pipe).sink("list"))
            .collect_schema()["out"]
        )
        arr = numpy_from_struct(lf.collect()["out"][0])
        return arr.shape, planned

    def test_single_channel_keeps_its_channel_axis(self) -> None:
        import polars as pl

        img = [[[0.0], [1.0]], [[2.0], [3.0]]]  # [2, 2, 1]
        shape, planned = self._warp(img, pl.List(pl.List(pl.List(pl.Float64))))
        assert shape == (2, 2, 1)
        assert planned == pl.List(pl.List(pl.List(pl.Float32)))

    def test_two_dimensional_input_stays_two_dimensional(self) -> None:
        import polars as pl

        img = [[0.0, 1.0], [2.0, 3.0]]  # [2, 2], no channel axis
        shape, planned = self._warp(img, pl.List(pl.List(pl.Float64)))
        assert shape == (2, 2)
        assert planned == pl.List(pl.List(pl.Float32))

    def test_multi_channel_unchanged(self) -> None:
        import polars as pl

        img = [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]  # [1, 2, 3]
        shape, planned = self._warp(img, pl.List(pl.List(pl.List(pl.Float64))))
        assert shape == (2, 2, 3)
        assert planned == pl.List(pl.List(pl.List(pl.Float32)))
