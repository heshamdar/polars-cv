"""
Tests for affine transform pipeline builder operations.

Tests the Python-side pipeline construction for warp_affine, shear,
and rotate_and_scale. These tests do NOT require the compiled plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from polars_cv import Pipeline

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
        assert op.params["matrix"].value == [1.0, 0.0, 50.0, 0.0, 1.0, 30.0]
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
        assert matrix == [1.0, 0.2, 0.0, 0.0, 1.0, 0.0]

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
        matrix = pipe._ops[0].params["matrix"].value
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
        matrix = pipe._ops[0].params["matrix"].value
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
        matrix = pipe._ops[0].params["matrix"].value
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
        # OpSpec.to_dict() serializes params at top-level
        fused_matrix = fused["matrix"]["value"]
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


class TestAffineContract:
    """Tests for the warp_affine operation contract."""

    def test_contract_exists(self) -> None:
        """warp_affine has an entry in OPERATION_CONTRACTS."""
        from polars_cv._types import OPERATION_CONTRACTS

        assert "warp_affine" in OPERATION_CONTRACTS

    def test_contract_preserves_dtype(self) -> None:
        """warp_affine contract preserves dtype."""
        from polars_cv._types import OPERATION_CONTRACTS, DTypeEffect

        contract = OPERATION_CONTRACTS["warp_affine"]
        assert contract.dtype_effect is DTypeEffect.PRESERVE

    def test_contract_preserves_ndim(self) -> None:
        """warp_affine contract preserves ndim."""
        from polars_cv._types import OPERATION_CONTRACTS, NdimEffect

        contract = OPERATION_CONTRACTS["warp_affine"]
        assert contract.ndim_effect is NdimEffect.PRESERVE

    def test_contract_alpha_passthrough(self) -> None:
        """warp_affine uses PASSTHROUGH alpha mode (all channels transformed)."""
        from polars_cv._types import OPERATION_CONTRACTS, AlphaMode

        contract = OPERATION_CONTRACTS["warp_affine"]
        assert contract.alpha_mode is AlphaMode.PASSTHROUGH
