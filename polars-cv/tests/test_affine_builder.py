"""
Tests for affine transform pipeline builder operations.

Tests the Python-side pipeline construction for warp_affine, shear,
and rotate_and_scale. These tests do NOT require the compiled plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from polars_cv import Pipeline
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
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "warp_affine"

    def test_warp_affine_requires_6_elements(self) -> None:
        """Affine matrix must have exactly 6 elements."""
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="6 elements"):
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
        op = pipe._ops[0]
        # Each matrix element is tracked as its own ParamValue dict so it can be
        # a per-row expression; literal floats round-trip through the value key.
        assert [m["value"] for m in op.params["matrix"].value] == [
            1.0,
            0.0,
            50.0,
            0.0,
            1.0,
            30.0,
        ]
        assert op.params["output_height"].value == 224
        assert op.params["output_width"].value == 224

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
        assert pipe._ops[0].params["interpolation"].value == "bilinear"

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
        assert pipe._ops[0].params["interpolation"].value == "nearest"

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
        assert pipe._ops[0].params["border_value"].value == 128.0

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
        assert pipe._output_dtype == "auto"

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
        assert pipe._shape_hints.height is not None
        assert pipe._shape_hints.height.value == 224
        assert pipe._shape_hints.width is not None
        assert pipe._shape_hints.width.value == 320

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
        assert len(pipe._ops) == 3
        assert pipe._ops[1].op == "warp_affine"

    def test_warp_affine_immutability(self) -> None:
        """Pipeline is immutable — warp_affine returns a new instance."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = pipe1.warp_affine(
            matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            output_size=(100, 100),
        )
        assert len(pipe1._ops) == 0
        assert len(pipe2._ops) == 1


class TestShearPipelineBuilder:
    """Tests for the shear convenience method."""

    def test_shear_basic(self) -> None:
        """Shear builds a warp_affine op with shear matrix."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .shear(sx=0.2, sy=0.0, output_size=(100, 100))
        )
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "warp_affine"
        matrix = pipe._ops[0].params["matrix"].value
        assert [m["value"] for m in matrix] == [1.0, 0.2, 0.0, 0.0, 1.0, 0.0]

    def test_shear_requires_output_size(self) -> None:
        """Shear requires output_size."""
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="output_size"):
            pipe.shear(sx=0.2, sy=0.0)

    def test_shear_both_axes(self) -> None:
        """Shear on both axes."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .shear(sx=0.3, sy=0.1, output_size=(200, 200))
        )
        matrix = [m["value"] for m in pipe._ops[0].params["matrix"].value]
        assert matrix == [1.0, 0.3, 0.0, 0.1, 1.0, 0.0]

    def test_shear_domain_validation(self) -> None:
        """Shear requires buffer domain (delegates to warp_affine)."""
        pipe = Pipeline().source("image_bytes").extract_contours()
        with pytest.raises(ValueError, match="buffer"):
            pipe.shear(sx=0.2, sy=0.0, output_size=(100, 100))


class TestRotateAndScalePipelineBuilder:
    """Tests for the rotate_and_scale convenience method."""

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
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "warp_affine"

    def test_rotate_and_scale_requires_center(self) -> None:
        """rotate_and_scale requires center."""
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="center"):
            pipe.rotate_and_scale(angle=45.0, output_size=(100, 100))

    def test_rotate_and_scale_requires_output_size(self) -> None:
        """rotate_and_scale requires output_size."""
        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="output_size"):
            pipe.rotate_and_scale(angle=45.0, center=(50.0, 50.0))

    def test_rotate_and_scale_matrix_correctness(self) -> None:
        """Verify the rotation matrix is correct for 90 degrees."""
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
        matrix = [m["value"] for m in pipe._ops[0].params["matrix"].value]
        rad = math.radians(90.0)
        cos_a = math.cos(rad) * 1.0
        sin_a = math.sin(rad) * 1.0
        cx, cy = 50.0, 50.0
        tx = (1 - cos_a) * cx + sin_a * cy
        ty = -sin_a * cx + (1 - cos_a) * cy
        expected = [cos_a, -sin_a, tx, sin_a, cos_a, ty]
        for actual, exp in zip(matrix, expected):
            assert abs(actual - exp) < 1e-10

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
        matrix = [m["value"] for m in pipe._ops[0].params["matrix"].value]
        # At angle=0 and scale=2: matrix should be [2, 0, -50, 0, 2, -50]
        assert abs(matrix[0] - 2.0) < 1e-10
        assert abs(matrix[4] - 2.0) < 1e-10


class TestAffineFusion:
    """Tests for consecutive affine operation fusion."""

    def test_two_translations_fuse(self) -> None:
        """Two consecutive translations fuse into one."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 10.0, 0.0, 1.0, 20.0],
                output_size=(100, 100),
            )
            .warp_affine(
                matrix=[1.0, 0.0, 30.0, 0.0, 1.0, 40.0],
                output_size=(200, 200),
            )
        )
        # Before fusion, there are 2 ops
        assert len(pipe._ops) == 2

        # After fusion (via _to_spec_dict), there should be 1 op
        spec = pipe._to_spec_dict()
        assert len(spec["ops"]) == 1
        fused = spec["ops"][0]
        assert fused["op"] == "warp_affine"
        # OpSpec.to_dict() serializes params at top-level; matrix elements are
        # per-element ParamValue dicts.
        fused_matrix = [m["value"] for m in fused["matrix"]["value"]]
        assert abs(fused_matrix[2] - 40.0) < 1e-10
        assert abs(fused_matrix[5] - 60.0) < 1e-10

    def test_non_affine_breaks_fusion(self) -> None:
        """A non-affine op between two affines prevents fusion."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 10.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
            )
            .normalize()
            .warp_affine(
                matrix=[1.0, 0.0, 20.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
            )
        )
        spec = pipe._to_spec_dict()
        # Should NOT fuse: affine, normalize, affine = 3 ops
        assert len(spec["ops"]) == 3

    def test_three_affines_fuse(self) -> None:
        """Three consecutive affines fuse into one."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .shear(sx=0.1, sy=0.0, output_size=(100, 100))
            .warp_affine(
                matrix=[1.0, 0.0, 10.0, 0.0, 1.0, 10.0],
                output_size=(100, 100),
            )
            .warp_affine(
                matrix=[2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                output_size=(200, 200),
            )
        )
        spec = pipe._to_spec_dict()
        assert len(spec["ops"]) == 1

    def test_fusion_preserves_output_dims(self) -> None:
        """Fused op uses the last affine's output dimensions."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(100, 100),
            )
            .warp_affine(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                output_size=(224, 224),
            )
        )
        spec = pipe._to_spec_dict()
        fused = spec["ops"][0]
        assert fused["output_height"]["value"] == 224
        assert fused["output_width"]["value"] == 224


def _expected_rotate_affine(
    angle_deg: float, ih: int, iw: int, *, expand: bool = False
) -> tuple[list[float], int, int]:
    """Reference rotate->warp_affine conversion for a known input shape.

    Mirrors the documented conversion: rotation about the input center,
    translated to the output center, with the output size either kept or
    expanded to the rotated bounding box.
    """
    import math

    rad = math.radians(angle_deg % 360)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = iw / 2.0, ih / 2.0
    if expand:
        new_w = round(iw * abs(cos_a) + ih * abs(sin_a))
        new_h = round(ih * abs(cos_a) + iw * abs(sin_a))
    else:
        new_w, new_h = iw, ih
    new_cx, new_cy = new_w / 2.0, new_h / 2.0
    tx = -cx * cos_a - cy * (-sin_a) + new_cx
    ty = -cx * sin_a - cy * cos_a + new_cy
    return [cos_a, -sin_a, tx, sin_a, cos_a, ty], new_h, new_w


def _compose(first: list[float], second: list[float]) -> list[float]:
    """Compose two 2x3 affine matrices (second applied after first)."""
    a1, b1, tx1, c1, d1, ty1 = first
    a2, b2, tx2, c2, d2, ty2 = second
    return [
        a2 * a1 + b2 * c1,
        a2 * b1 + b2 * d1,
        a2 * tx1 + b2 * ty1 + tx2,
        c2 * a1 + d2 * c1,
        c2 * b1 + d2 * d1,
        c2 * tx1 + d2 * ty1 + ty2,
    ]


IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class TestAffineFusionPositionalShape:
    """Rotate->affine conversion must use the shape ENTERING the rotate op.

    Regression tests for the bug where ``_try_convert_rotate_to_affine``
    read the pipeline's final ``_shape_hints`` (the shape after ALL ops)
    instead of the shape at the rotate's position, silently producing a
    wrong rotation center / output size whenever a shape-changing op
    followed the rotate.
    """

    def test_rotate_fused_with_following_affine_uses_input_shape(self) -> None:
        """Rotate before a warp_affine converts with its own input dims."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(45)
            .warp_affine(matrix=IDENTITY, output_size=(50, 50))
        )
        spec = pipe._to_spec_dict()
        # resize stays, rotate+warp_affine fuse into one
        assert [op["op"] for op in spec["ops"]] == ["resize", "warp_affine"]
        fused = spec["ops"][1]
        expected_matrix, _, _ = _expected_rotate_affine(45, ih=100, iw=100)
        expected = _compose(expected_matrix, IDENTITY)
        for got, want in zip([m["value"] for m in fused["matrix"]["value"]], expected):
            assert abs(got - want) < 1e-9
        # Output dims come from the LAST op in the fused run
        assert fused["output_height"]["value"] == 50
        assert fused["output_width"]["value"] == 50

    def test_expand_rotate_fused_uses_input_shape(self) -> None:
        """rotate(expand=True) converts with pre-rotate dims, not post."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(45, expand=True)
            .warp_affine(matrix=IDENTITY, output_size=(200, 200))
        )
        spec = pipe._to_spec_dict()
        assert [op["op"] for op in spec["ops"]] == ["resize", "warp_affine"]
        fused = spec["ops"][1]
        expected_matrix, _, _ = _expected_rotate_affine(45, ih=100, iw=100, expand=True)
        expected = _compose(expected_matrix, IDENTITY)
        for got, want in zip([m["value"] for m in fused["matrix"]["value"]], expected):
            assert abs(got - want) < 1e-9
        assert fused["output_height"]["value"] == 200
        assert fused["output_width"]["value"] == 200

    def test_two_rotates_fuse_with_positional_shapes(self) -> None:
        """Adjacent static rotates still fuse, each at its own input shape."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(30)
            .rotate(15)
        )
        spec = pipe._to_spec_dict()
        assert [op["op"] for op in spec["ops"]] == ["resize", "warp_affine"]
        fused = spec["ops"][1]
        m1, _, _ = _expected_rotate_affine(30, ih=100, iw=100)
        m2, _, _ = _expected_rotate_affine(15, ih=100, iw=100)
        expected = _compose(m1, m2)
        for got, want in zip([m["value"] for m in fused["matrix"]["value"]], expected):
            assert abs(got - want) < 1e-9

    def test_mid_chain_assert_shape_feeds_conversion(self) -> None:
        """assert_shape() hints at the rotate's position drive conversion."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(height=80, width=60)
            .rotate(45)
            .warp_affine(matrix=IDENTITY, output_size=(50, 50))
        )
        spec = pipe._to_spec_dict()
        assert [op["op"] for op in spec["ops"]] == ["warp_affine"]
        fused = spec["ops"][0]
        expected_matrix, _, _ = _expected_rotate_affine(45, ih=80, iw=60)
        expected = _compose(expected_matrix, IDENTITY)
        for got, want in zip([m["value"] for m in fused["matrix"]["value"]], expected):
            assert abs(got - want) < 1e-9

    def test_lone_rotate_is_not_converted(self) -> None:
        """A rotate with no adjacent affine op stays a runtime rotate.

        The runtime rotate computes its matrix from actual buffer
        dimensions; converting a lone rotate trades that for plan-time
        hints with zero fusion benefit.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(45)
            .grayscale()
        )
        spec = pipe._to_spec_dict()
        assert [op["op"] for op in spec["ops"]] == ["resize", "rotate", "grayscale"]

    def test_rotate_followed_by_resize_is_not_converted(self) -> None:
        """The original corruption trigger: shape-changing op after rotate."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(45)
            .resize(height=50, width=50)
        )
        spec = pipe._to_spec_dict()
        assert [op["op"] for op in spec["ops"]] == ["resize", "rotate", "resize"]

    def test_unknown_input_shape_blocks_conversion(self) -> None:
        """Rotate with unknown input dims must not convert, even when the
        FINAL pipeline shape is known (that was the bug's other face)."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize_scale(scale=0.5)
            .rotate(45)
            .warp_affine(matrix=IDENTITY, output_size=(50, 50))
        )
        spec = pipe._to_spec_dict()
        assert [op["op"] for op in spec["ops"]] == [
            "resize_scale",
            "rotate",
            "warp_affine",
        ]


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
        assert pipe._shape_hints.height is None
        assert pipe._shape_hints.width is None

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
        assert pipe._shape_hints.height is None
        assert pipe._shape_hints.width is None

    def test_expr_angle_square_non_expand_keeps_hints(self) -> None:
        """Square image, no expand: any angle keeps HxW."""
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=100)
            .rotate(pl.col("angle"))
        )
        assert pipe._shape_hints.height is not None
        assert pipe._shape_hints.height.value == 100
        assert pipe._shape_hints.width is not None
        assert pipe._shape_hints.width.value == 100


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
        assert pipe._ops[-1].op == "warp_affine"
        # Matrix is serialized as a list of 6 per-element ParamValue dicts.
        matrix_param = pipe._ops[-1].params["matrix"]
        assert isinstance(matrix_param.value, list)
        assert len(matrix_param.value) == 6

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
        assert len(pipe._ops[-1].params["matrix"].value) == 6

    def test_warp_affine_still_rejects_wrong_length(self) -> None:
        import polars as pl

        pipe = Pipeline().source("image_bytes")
        with pytest.raises(ValueError, match="6 elements"):
            pipe.warp_affine(matrix=[pl.col("a"), 1.0], output_size=(10, 10))

    def test_shear_accepts_expr_factors(self) -> None:
        import polars as pl

        pipe = (
            Pipeline()
            .source("image_bytes")
            .shear(sx=pl.col("shear_x"), output_size=(64, 64))
        )
        # shear delegates to warp_affine; the matrix carries the expr element.
        assert pipe._ops[-1].op == "warp_affine"
        assert len(pipe._ops[-1].params["matrix"].value) == 6


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
    """An affine warp's output rank must match its declared `infer_shape`.

    `ComputeOp::Affine::infer_shape` replaces H and W and leaves the rest of the
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
