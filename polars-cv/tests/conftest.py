"""
Pytest configuration and fixtures for polars-cv tests.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest

# Add the python source to the path for testing without installation
python_src = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(python_src))

if TYPE_CHECKING:
    pass


def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


# Mark tests with plugin_required marker for easy filtering
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@pytest.fixture
def create_test_png() -> Callable[[int, int, tuple[int, int, int]], bytes]:
    """
    Factory for creating test PNG images.

    Returns:
        A callable that creates PNG bytes for a given width, height, and color.
    """

    def _create(
        width: int = 100,
        height: int = 100,
        color: tuple[int, int, int] = (128, 128, 128),
    ) -> bytes:
        """
        Create a test PNG image.

        Args:
            width: Image width.
            height: Image height.
            color: RGB color tuple.

        Returns:
            PNG bytes.
        """
        try:
            from PIL import Image

            img = Image.new("RGB", (width, height), color)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pytest.skip("PIL/Pillow required for this test")
            return b""

    return _create


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    """
    Encode a numpy array as PNG bytes.

    Returns:
        A callable that encodes a numpy array as PNG bytes.
    """

    def _encode(arr: np.ndarray) -> bytes:
        """
        Encode numpy array as PNG bytes.

        Args:
            arr: NumPy array with shape (H, W, 3) or (H, W) and dtype uint8.

        Returns:
            PNG bytes.
        """
        try:
            from PIL import Image

            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pytest.skip("PIL/Pillow required for this test")
            return b""

    return _encode


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create minimal valid PNG bytes for testing."""
    # Minimal 1x1 red PNG
    # This is a valid PNG that can be decoded by image libraries
    return bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,  # 1x1
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,  # 8-bit RGB
            0x90,
            0x77,
            0x53,
            0xDE,  # CRC
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,
            0x54,  # IDAT chunk
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xCF,
            0xC0,
            0x00,
            0x00,  # Compressed data
            0x00,
            0x03,
            0x00,
            0x01,  # Compressed data cont.
            0x00,
            0x18,
            0xDD,
            0x8D,
            0xB4,  # CRC
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,  # IEND chunk
            0xAE,
            0x42,
            0x60,
            0x82,  # CRC
        ]
    )
