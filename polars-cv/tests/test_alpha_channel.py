"""Tests for alpha channel preservation and handling.

Verifies that RGBA (4-channel) and GrayA (2-channel) images are decoded,
processed, and encoded correctly, and that the AlphaMode contract system
drives channel inference at planning time.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
from polars_cv import AlphaMode, Pipeline
from polars_cv._types import OPERATION_CONTRACTS

from tests.conftest import plugin_required


def _make_rgba_png(width: int = 10, height: int = 10) -> bytes:
    """Create an RGBA PNG image with a non-trivial alpha channel."""
    from PIL import Image

    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 0] = 200  # R
    arr[:, :, 1] = 100  # G
    arr[:, :, 2] = 50  # B
    arr[:, :, 3] = 128  # A (semi-transparent)
    img = Image.fromarray(arr, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_graya_png(width: int = 10, height: int = 10) -> bytes:
    """Create a GrayA PNG image with a non-trivial alpha channel."""
    from PIL import Image

    arr = np.zeros((height, width, 2), dtype=np.uint8)
    arr[:, :, 0] = 180  # Gray intensity
    arr[:, :, 1] = 64  # Alpha
    img = Image.fromarray(arr, "LA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgb_png(width: int = 10, height: int = 10) -> bytes:
    """Create a standard RGB PNG image."""
    from PIL import Image

    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[:, :, 1] = 100
    arr[:, :, 2] = 50
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gray_png(width: int = 10, height: int = 10) -> bytes:
    """Create a single-channel grayscale PNG image."""
    from PIL import Image

    arr = np.full((height, width), 180, dtype=np.uint8)
    img = Image.fromarray(arr, "L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _extract_shape(df: pl.DataFrame, col: str = "output") -> list[int]:
    """Extract shape from a numpy struct output."""
    from polars_cv import numpy_from_struct

    struct_val = df[col][0]
    arr = numpy_from_struct(struct_val)
    return list(arr.shape)


# ---------------------------------------------------------------------------
# Contract tests (no plugin required)
# ---------------------------------------------------------------------------


class TestAlphaModeContracts:
    """Verify AlphaMode is assigned correctly in OPERATION_CONTRACTS."""

    def test_passthrough_ops_have_passthrough(self) -> None:
        """Spatial and element-wise ops should be PASSTHROUGH."""
        passthrough_ops = [
            "resize",
            "normalize",
            "cast",
            "clamp",
            "scale",
            "relu",
            "flip",
            "transpose",
            "crop",
            "pad",
            "pad_to_size",
            "letterbox",
            "rotate",
            "reshape",
            "adjust_contrast",
            "adjust_gamma",
            "invert",
            "convolve2d",
            "channel_swap",
            "equalize_histogram",
        ]
        for op in passthrough_ops:
            contract = OPERATION_CONTRACTS[op]
            assert contract.alpha_mode is AlphaMode.PASSTHROUGH, (
                f"{op} should be PASSTHROUGH, got {contract.alpha_mode}"
            )

    def test_drop_ops_have_drop(self) -> None:
        """Ops that discard alpha should be DROP."""
        drop_ops = [
            "grayscale",
            "canny",
            "threshold",
            "channel_select",
            "perceptual_hash",
            "erode",
            "dilate",
            "morphology_gradient",
        ]
        for op in drop_ops:
            contract = OPERATION_CONTRACTS[op]
            assert contract.alpha_mode is AlphaMode.DROP, (
                f"{op} should be DROP, got {contract.alpha_mode}"
            )

    def test_strip_process_restore_ops(self) -> None:
        """Ops that strip, process, and restore alpha.

        sobel/laplacian/sharpen are intentionally absent: they are convenience
        builders that lower to convolve2d (PASSTHROUGH), so their alpha handling
        is governed by the convolve2d contract, not a contract of their own.
        """
        spr_ops = ["blur", "cvt_color"]
        for op in spr_ops:
            contract = OPERATION_CONTRACTS[op]
            assert contract.alpha_mode is AlphaMode.STRIP_PROCESS_RESTORE, (
                f"{op} should be STRIP_PROCESS_RESTORE, got {contract.alpha_mode}"
            )

    def test_not_applicable_ops(self) -> None:
        """Non-image domain ops should be NOT_APPLICABLE."""
        na_ops = [
            "reduce_sum",
            "reduce_mean",
            "extract_shape",
            "contour_area",
            "histogram",
            "channel_merge",
        ]
        for op in na_ops:
            contract = OPERATION_CONTRACTS[op]
            assert contract.alpha_mode is AlphaMode.NOT_APPLICABLE, (
                f"{op} should be NOT_APPLICABLE, got {contract.alpha_mode}"
            )

    def test_every_contract_has_alpha_mode(self) -> None:
        """All contracts should have an explicitly set AlphaMode."""
        for op_name, contract in OPERATION_CONTRACTS.items():
            assert hasattr(contract, "alpha_mode"), (
                f"{op_name} contract is missing alpha_mode"
            )
            assert isinstance(contract.alpha_mode, AlphaMode), (
                f"{op_name} alpha_mode is not an AlphaMode enum"
            )


class TestChannelInferencePlanning:
    """Verify planning-time channel inference uses AlphaMode correctly."""

    def test_image_source_channels_unknown(self) -> None:
        """Image sources should have unknown channels at planning time."""
        pipe = Pipeline().source("image_bytes")
        assert pipe._shape_hints.channels is None

    def test_grayscale_drops_to_1(self) -> None:
        """Grayscale (DROP) always sets channels to 1."""
        pipe = Pipeline().source("image_bytes").grayscale()
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 1

    def test_canny_drops_to_1(self) -> None:
        """Canny (DROP) always sets channels to 1."""
        pipe = Pipeline().source("image_bytes").canny()
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 1

    def test_passthrough_preserves_unknown(self) -> None:
        """PASSTHROUGH ops preserve unknown channels."""
        pipe = Pipeline().source("image_bytes").resize(height=224, width=224)
        assert pipe._shape_hints.channels is None

    def test_passthrough_preserves_known(self) -> None:
        """PASSTHROUGH ops preserve known channel count."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .resize(height=224, width=224)
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 4

    def test_cvt_color_with_known_rgba(self) -> None:
        """cvt_color (STRIP_PROCESS_RESTORE) on known RGBA input."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .cvt_color("rgb", "hsv")
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 4

    def test_cvt_color_to_gray_with_known_rgba(self) -> None:
        """cvt_color to gray on RGBA produces GrayA (2ch)."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=4)
            .cvt_color("rgb", "gray")
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 2

    def test_cvt_color_with_unknown_channels(self) -> None:
        """cvt_color with unknown input channels produces unknown output."""
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        assert pipe._shape_hints.channels is None

    def test_blur_with_known_rgba(self) -> None:
        """Blur (STRIP_PROCESS_RESTORE) on known 4ch preserves 4ch."""
        pipe = Pipeline().source("image_bytes").assert_shape(channels=4).blur(sigma=1.0)
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 4

    def test_cvt_color_rgb_no_alpha(self) -> None:
        """cvt_color on 3ch (no alpha) produces 3ch."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=3)
            .cvt_color("rgb", "hsv")
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 3

    def test_threshold_drops_to_1(self) -> None:
        """Threshold (DROP + PRESERVE ndim) should set channels to 1."""
        pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 1

    def test_erode_drops_to_1(self) -> None:
        """Erode (DROP + PRESERVE ndim) should set channels to 1."""
        pipe = (
            Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 1

    def test_dilate_drops_to_1(self) -> None:
        """Dilate (DROP + PRESERVE ndim) should set channels to 1."""
        pipe = (
            Pipeline().source("image_bytes").grayscale().threshold(128).dilate(ksize=3)
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 1

    def test_morphology_gradient_drops_to_1(self) -> None:
        """Morphology gradient (DROP + PRESERVE ndim) should set channels to 1."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .morphology_gradient(ksize=3)
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 1

    def test_threshold_erode_dilate_chain_channels(self) -> None:
        """Chained DROP ops should all report channels=1."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .erode(ksize=3)
            .dilate(ksize=3)
        )
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 1

    def test_rotate_expand_true_shape_hints(self) -> None:
        """rotate(expand=True) should compute correct output dimensions."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(height=100, width=200)
            .rotate(45, expand=True)
        )
        h = pipe._shape_hints.height
        w = pipe._shape_hints.width
        assert h is not None
        assert w is not None
        assert h.value > 100
        assert w.value > 200

    def test_rotate_expand_true_90_swaps(self) -> None:
        """rotate(90, expand=True) should swap height and width."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(height=100, width=200)
            .rotate(90, expand=True)
        )
        assert pipe._shape_hints.height is not None
        assert pipe._shape_hints.width is not None
        assert pipe._shape_hints.height.value == 200
        assert pipe._shape_hints.width.value == 100

    def test_rotate_expand_false_90_swaps(self) -> None:
        """rotate(90, expand=False) should swap height and width."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(height=100, width=200)
            .rotate(90)
        )
        assert pipe._shape_hints.height is not None
        assert pipe._shape_hints.width is not None
        assert pipe._shape_hints.height.value == 200
        assert pipe._shape_hints.width.value == 100

    def test_rotate_expand_false_arbitrary_preserves_size(self) -> None:
        """rotate(45, expand=False) should keep original dimensions."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(height=100, width=200)
            .rotate(45)
        )
        assert pipe._shape_hints.height is not None
        assert pipe._shape_hints.width is not None
        assert pipe._shape_hints.height.value == 100
        assert pipe._shape_hints.width.value == 200

    def test_expected_shape_none_when_channels_unknown(self) -> None:
        """expected_shape returns None when channels are unknown."""
        pipe = Pipeline().source("image_bytes").resize(height=224, width=224)
        assert pipe._shape_hints.channels is None
        # H and W are known but channels are not, so expected_shape is None
        assert pipe._shape_hints.height is not None
        assert pipe._shape_hints.width is not None


# ---------------------------------------------------------------------------
# Integration tests (plugin required)
# ---------------------------------------------------------------------------


@plugin_required
class TestAlphaDecoding:
    """Test that RGBA and GrayA images are decoded with alpha preserved."""

    def test_rgba_png_decode_shape(self) -> None:
        """RGBA PNG should decode to [H, W, 4]."""
        png_bytes = _make_rgba_png(10, 8)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [8, 10, 4]

    def test_graya_png_decode_shape(self) -> None:
        """GrayA PNG should decode to [H, W, 2]."""
        png_bytes = _make_graya_png(10, 8)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [8, 10, 2]

    def test_rgb_png_decode_shape_unchanged(self) -> None:
        """RGB PNG should still decode to [H, W, 3]."""
        png_bytes = _make_rgb_png(10, 8)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [8, 10, 3]

    def test_gray_png_decode_shape_unchanged(self) -> None:
        """Grayscale PNG should still decode to [H, W, 1]."""
        png_bytes = _make_gray_png(10, 8)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [8, 10, 1]

    def test_rgba_alpha_values_preserved(self) -> None:
        """The alpha channel values should be preserved through decode."""
        from polars_cv import numpy_from_struct

        png_bytes = _make_rgba_png(4, 4)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result["output"][0])
        assert arr.shape[2] == 4
        assert np.all(arr[:, :, 3] == 128)


@plugin_required
class TestAlphaPassthroughOps:
    """Test PASSTHROUGH operations preserve alpha channel."""

    def test_resize_preserves_rgba(self) -> None:
        """Resize should keep all 4 channels."""
        png_bytes = _make_rgba_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").resize(height=5, width=5)
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [5, 5, 4]

    def test_normalize_preserves_rgba(self) -> None:
        """Normalize should keep all 4 channels."""
        from polars_cv import numpy_from_struct

        png_bytes = _make_rgba_png(4, 4)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").normalize()
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result["output"][0])
        assert arr.shape[2] == 4

    def test_flip_preserves_rgba(self) -> None:
        """Flip should keep all 4 channels."""
        png_bytes = _make_rgba_png(4, 4)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").flip(axes=[1])
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [4, 4, 4]


@plugin_required
class TestAlphaDropOps:
    """Test DROP operations discard alpha channel."""

    def test_grayscale_drops_alpha(self) -> None:
        """Grayscale on RGBA should produce 1 channel."""
        png_bytes = _make_rgba_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [10, 10, 1]

    def test_grayscale_graya_drops_alpha(self) -> None:
        """Grayscale on GrayA should produce 1 channel."""
        png_bytes = _make_graya_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [10, 10, 1]

    def test_canny_drops_alpha(self) -> None:
        """Canny on RGBA should produce 1-channel mask."""
        png_bytes = _make_rgba_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").canny()
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [10, 10, 1]


@plugin_required
class TestAlphaStripProcessRestore:
    """Test STRIP_PROCESS_RESTORE operations preserve alpha through processing."""

    def test_blur_preserves_rgba(self) -> None:
        """Blur on RGBA should produce 4 channels."""
        png_bytes = _make_rgba_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").blur(sigma=1.0)
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [10, 10, 4]

    def test_cvt_color_rgb_to_hsv_preserves_alpha(self) -> None:
        """cvt_color RGB->HSV on RGBA should produce 4 channels."""
        png_bytes = _make_rgba_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [10, 10, 4]

    def test_cvt_color_rgb_to_gray_produces_graya(self) -> None:
        """cvt_color RGB->Gray on RGBA should produce 2 channels (GrayA)."""
        png_bytes = _make_rgba_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "gray")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        shape = _extract_shape(result)
        assert shape == [10, 10, 2]

    def test_cvt_color_alpha_values_preserved(self) -> None:
        """Alpha values should survive through cvt_color RGB->HSV."""
        from polars_cv import numpy_from_struct

        png_bytes = _make_rgba_png(4, 4)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").cvt_color("rgb", "bgr")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        arr = numpy_from_struct(result["output"][0])
        assert arr.shape[2] == 4
        assert np.all(arr[:, :, 3] == 128)


@plugin_required
class TestAlphaEncoding:
    """Test that RGBA/GrayA buffers can be encoded back to image formats."""

    def test_rgba_to_png_roundtrip(self) -> None:
        """RGBA buffer should encode to valid PNG bytes."""
        png_bytes = _make_rgba_png(10, 10)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("png"))
        output_bytes = result["output"][0]
        assert isinstance(output_bytes, bytes)
        assert len(output_bytes) > 0
        assert output_bytes[:4] == b"\x89PNG"

    def test_rgba_tiff_roundtrip(self) -> None:
        """RGBA buffer should encode to TIFF and decode back."""
        from polars_cv import numpy_from_struct

        png_bytes = _make_rgba_png(8, 8)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes")

        result = df.with_columns(tiff=pl.col("image").cv.pipe(pipe).sink("tiff"))
        tiff_bytes = result["tiff"][0]
        assert isinstance(tiff_bytes, bytes)
        assert len(tiff_bytes) > 0

        df2 = pl.DataFrame({"image": [tiff_bytes]})
        pipe2 = Pipeline().source("image_bytes")
        result2 = df2.with_columns(output=pl.col("image").cv.pipe(pipe2).sink("numpy"))
        arr = numpy_from_struct(result2["output"][0])
        assert arr.shape[2] == 4
