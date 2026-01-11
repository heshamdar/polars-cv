"""
Tests for padding operations.

These tests validate:
- pad(): Add padding with specified amounts and mode
- pad_to_size(): Pad to exact dimensions with positioning
- letterbox(): Resize maintaining aspect ratio and pad to exact size
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    pass


# Check if plugin is available
def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    from pathlib import Path

    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


# Mark tests with plugin_required marker
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


# --- Fixtures ---


@pytest.fixture
def small_image() -> np.ndarray:
    """
    Small test image (50x100 = 1:2 aspect ratio).

    Returns:
        np.ndarray: RGB image with shape (50, 100, 3) and dtype uint8.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (50, 100, 3), dtype=np.uint8)


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    """
    Encode a numpy array as PNG bytes.

    Returns:
        A callable that encodes a numpy array as PNG bytes.
    """

    def _encode(arr: np.ndarray) -> bytes:
        from io import BytesIO

        from PIL import Image

        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return _encode


# --- Reference Tests (NumPy baseline) ---


class TestPadReference:
    """Establish expected behavior for padding using NumPy as reference."""

    def test_pad_constant_reference(self, small_image: np.ndarray) -> None:
        """
        Pad with constant value.

        pad(top=10, bottom=10, left=20, right=20) on 50x100x3 -> 70x140x3
        """
        padded = np.pad(
            small_image,
            ((10, 10), (20, 20), (0, 0)),  # (top, bottom), (left, right), (channels)
            mode="constant",
            constant_values=0,
        )

        assert padded.shape == (70, 140, 3)
        # Verify padding is zeros
        assert np.all(padded[:10, :, :] == 0)  # Top
        assert np.all(padded[-10:, :, :] == 0)  # Bottom
        assert np.all(padded[:, :20, :] == 0)  # Left
        assert np.all(padded[:, -20:, :] == 0)  # Right
        # Verify original content preserved
        np.testing.assert_array_equal(padded[10:60, 20:120, :], small_image)

    def test_pad_edge_reference(self, small_image: np.ndarray) -> None:
        """
        Pad by replicating edge values.
        """
        padded = np.pad(
            small_image,
            ((5, 5), (10, 10), (0, 0)),
            mode="edge",
        )

        assert padded.shape == (60, 120, 3)
        # Edge values should match original edges
        np.testing.assert_array_equal(padded[0, 10:110, :], small_image[0, :, :])  # Top
        np.testing.assert_array_equal(
            padded[-1, 10:110, :], small_image[-1, :, :]
        )  # Bottom

    def test_pad_reflect_reference(self, small_image: np.ndarray) -> None:
        """
        Pad with reflection (not including edge).
        """
        padded = np.pad(
            small_image,
            ((5, 5), (10, 10), (0, 0)),
            mode="reflect",
        )

        assert padded.shape == (60, 120, 3)

    def test_pad_symmetric_reference(self, small_image: np.ndarray) -> None:
        """
        Pad with symmetric reflection (including edge).
        """
        padded = np.pad(
            small_image,
            ((5, 5), (10, 10), (0, 0)),
            mode="symmetric",
        )

        assert padded.shape == (60, 120, 3)

    def test_pad_with_value_reference(self, small_image: np.ndarray) -> None:
        """
        Pad with non-zero constant value.
        """
        fill_value = 128
        padded = np.pad(
            small_image,
            ((10, 10), (10, 10), (0, 0)),
            mode="constant",
            constant_values=fill_value,
        )

        assert np.all(padded[:10, :, :] == fill_value)


class TestPadToSizeReference:
    """Establish expected behavior for padding to exact size."""

    def test_pad_to_size_center_reference(self, small_image: np.ndarray) -> None:
        """
        Pad to target size with centered positioning.

        50x100 padded to 100x200, centered.
        """
        target_h, target_w = 100, 200
        current_h, current_w = small_image.shape[:2]

        pad_h = target_h - current_h  # 50
        pad_w = target_w - current_w  # 100

        top = pad_h // 2  # 25
        bottom = pad_h - top  # 25
        left = pad_w // 2  # 50
        right = pad_w - left  # 50

        padded = np.pad(
            small_image,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        assert padded.shape == (target_h, target_w, 3)
        # Verify content is centered
        np.testing.assert_array_equal(padded[25:75, 50:150, :], small_image)

    def test_pad_to_size_topleft_reference(self, small_image: np.ndarray) -> None:
        """
        Pad to target size with top-left positioning (padding on right and bottom).
        """
        target_h, target_w = 100, 200
        current_h, current_w = small_image.shape[:2]

        bottom = target_h - current_h
        right = target_w - current_w

        padded = np.pad(
            small_image,
            ((0, bottom), (0, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        assert padded.shape == (target_h, target_w, 3)
        np.testing.assert_array_equal(padded[:50, :100, :], small_image)


class TestLetterboxReference:
    """Establish expected behavior for letterboxing."""

    def test_letterbox_wide_image_reference(self) -> None:
        """
        Letterbox a wide image (200x100) to 100x100 target.

        Expected: resize to 100x50, then pad top/bottom by 25.
        """
        from PIL import Image

        rng = np.random.default_rng(42)
        wide_img = rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)

        # Target: 100x100
        target_h, _ = 100, 100

        # First resize preserving aspect ratio
        # 200x100 (2:1), to fit in 100x100: max dimension is 200 (width)
        # Scale = 100/200 = 0.5 -> 100x50
        pil_img = Image.fromarray(wide_img)
        resized = pil_img.resize((100, 50), Image.LANCZOS)
        resized_arr = np.array(resized)

        # Then pad vertically to reach 100x100
        pad_h = target_h - 50  # 50
        top = pad_h // 2  # 25
        bottom = pad_h - top  # 25

        letterboxed = np.pad(
            resized_arr,
            ((top, bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        assert letterboxed.shape == (100, 100, 3)

    def test_letterbox_tall_image_reference(self) -> None:
        """
        Letterbox a tall image (100x200) to 100x100 target.

        Expected: resize to 50x100, then pad left/right by 25.
        """
        from PIL import Image

        rng = np.random.default_rng(42)
        tall_img = rng.integers(0, 256, (200, 100, 3), dtype=np.uint8)

        _, target_w = 100, 100

        # Resize preserving aspect ratio: 100x200 (1:2) to fit 100x100
        # Scale = 100/200 = 0.5 -> 50x100
        pil_img = Image.fromarray(tall_img)
        resized = pil_img.resize((50, 100), Image.LANCZOS)
        resized_arr = np.array(resized)

        # Pad horizontally
        pad_w = target_w - 50  # 50
        left = pad_w // 2  # 25
        right = pad_w - left  # 25

        letterboxed = np.pad(
            resized_arr,
            ((0, 0), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        assert letterboxed.shape == (100, 100, 3)


# --- polars-cv Integration Tests ---


@plugin_required
class TestPadPolarsCV:
    """Tests for polars-cv pad() method."""

    def test_pad_constant(
        self,
        small_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Pad with constant zero fill."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(small_image)]})

        pipe = (
            Pipeline().source("image_bytes").pad(top=10, bottom=10, left=20, right=20)
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (70, 140, 3)
        # Verify padding
        assert np.all(actual[:10, :, :] == 0)
        assert np.all(actual[-10:, :, :] == 0)
        assert np.all(actual[:, :20, :] == 0)
        assert np.all(actual[:, -20:, :] == 0)

    def test_pad_with_value(
        self,
        small_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Pad with non-zero constant value."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(small_image)]})

        pipe = Pipeline().source("image_bytes").pad(top=10, bottom=10, value=128)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (70, 100, 3)
        assert np.all(actual[:10, :, :] == 128)
        assert np.all(actual[-10:, :, :] == 128)

    def test_pad_edge_mode(
        self,
        small_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Pad with edge replication."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(small_image)]})

        pipe = Pipeline().source("image_bytes").pad(top=5, bottom=5, mode="edge")

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (60, 100, 3)


@plugin_required
class TestPadToSizePolarsCV:
    """Tests for polars-cv pad_to_size() method."""

    def test_pad_to_size_center(
        self,
        small_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Pad to target size with centered positioning."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(small_image)]})

        pipe = Pipeline().source("image_bytes").pad_to_size(height=100, width=200)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (100, 200, 3)
        # Verify content is centered
        np.testing.assert_array_equal(actual[25:75, 50:150, :], small_image)

    def test_flip_then_pad(
        self, small_image: np.ndarray, encode_png: Callable[[np.ndarray], bytes]
    ) -> None:
        """Flip then pad."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(small_image)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=50, width=50)
            .flip_v()
            .pad_to_size(height=100, width=200)
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (100, 200, 3)

    def test_pad_to_size_topleft(
        self,
        small_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Pad to target size with top-left positioning."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(small_image)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .pad_to_size(height=100, width=200, position="top-left")
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (100, 200, 3)
        np.testing.assert_array_equal(actual[:50, :100, :], small_image)


@plugin_required
class TestLetterboxPolarsCV:
    """Tests for polars-cv letterbox() method."""

    def test_letterbox_wide_image(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Letterbox a wide image to square target."""
        from polars_cv import Pipeline, numpy_from_struct

        rng = np.random.default_rng(42)
        wide_img = rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)

        df = pl.DataFrame({"img": [encode_png(wide_img)]})

        # Letterbox to 100x100
        pipe = Pipeline().source("image_bytes").letterbox(height=100, width=100)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (100, 100, 3)
        # Padding should be on top and bottom (resized to 100x50, padded to 100x100)
        # Top 25 rows should be padding (all same value)
        # Bottom 25 rows should be padding

    def test_letterbox_tall_image(
        self,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Letterbox a tall image to square target."""
        from polars_cv import Pipeline, numpy_from_struct

        rng = np.random.default_rng(42)
        tall_img = rng.integers(0, 256, (200, 100, 3), dtype=np.uint8)

        df = pl.DataFrame({"img": [encode_png(tall_img)]})

        pipe = Pipeline().source("image_bytes").letterbox(height=100, width=100)

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (100, 100, 3)

    def test_letterbox_with_fill_value(
        self,
        small_image: np.ndarray,
        encode_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Letterbox with non-zero fill value."""
        from polars_cv import Pipeline, numpy_from_struct

        df = pl.DataFrame({"img": [encode_png(small_image)]})

        # 50x100 to 100x100: resize to 50x100, pad height by 50 (25 top, 25 bottom)
        pipe = (
            Pipeline().source("image_bytes").letterbox(height=100, width=100, value=128)
        )

        result = df.select(output=pl.col("img").cv.pipe(pipe).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])

        assert actual.shape == (100, 100, 3)
        # Top padding should be fill value
        assert np.all(actual[:25, :, :] == 128)
        # Bottom padding should be fill value
        assert np.all(actual[75:, :, :] == 128)
