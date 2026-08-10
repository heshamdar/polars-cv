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


# Skip, rather than fail, when the compiled extension is absent. This is a
# `skipif` and not a named marker, so it cannot be selected with `-k`/`-m`;
# tests carrying it drop out on their own when the plugin is not built.
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


def make_test_png(
    width: int = 10, height: int = 10, color: tuple[int, int, int] = (255, 0, 0)
) -> bytes:
    """
    Create a test PNG image (module-level; importable by test files that
    build images outside a fixture context).

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


#: PIL mode per channel count. 2 channels is grayscale+alpha, which is what the
#: ``StripProcessRestore`` channel rule produces from RGBA and which nothing
#: fed through a sink before the schema-parity matrix existed.
_MODE_FOR_CHANNELS = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}


def make_image_png(
    height: int = 8,
    width: int = 8,
    channels: int = 3,
    *,
    sixteen_bit: bool = False,
    seed: int = 0,
) -> bytes:
    """Encode a deterministic PNG with an exact channel count.

    ``create_test_png``/``make_test_png`` only make flat RGB images. The schema
    matrix needs every channel count the alpha rules distinguish (1, 2, 3, 4)
    and a 16-bit path for the ``u16`` decode, at sizes it chooses, with varying
    pixel content so operations like ``equalize_histogram`` and ``canny`` have
    something to act on.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""

    buf = io.BytesIO()
    if sixteen_bit:
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 65535, size=(height, width), dtype=np.uint16)
        Image.fromarray(arr, mode="I;16").save(buf, format="PNG")
        return buf.getvalue()

    mode = _MODE_FOR_CHANNELS.get(channels)
    if mode is None:
        raise ValueError(f"unsupported channel count for a PNG: {channels}")

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    if channels == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def make_rect_png(height: int = 100, width: int = 200, channels: int = 3) -> bytes:
    """A black image with one white filled rectangle.

    Contour pipelines need an image that thresholds into a small, predictable
    number of regions. Noise thresholds into hundreds of one-pixel contours,
    which is slow and makes any downstream assertion depend on the RNG.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""

    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
    if channels == 1:
        arr = arr[:, :, 0]
    buf = io.BytesIO()
    Image.fromarray(arr, mode=_MODE_FOR_CHANNELS[channels]).save(buf, format="PNG")
    return buf.getvalue()


def make_ring_png(height: int = 100, width: int = 200, channels: int = 3) -> bytes:
    """A black image with one white rectangle that has a rectangular hole.

    ``make_rect_png``'s solid block cannot tell ``extract_contours(mode=)``
    apart: with no enclosed background region, "external" and "all" find the
    same single border. A ring has a second border to find, so the mode
    genuinely changes the result.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""

    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[height // 8 : 7 * height // 8, width // 8 : 7 * width // 8] = 255
    arr[3 * height // 8 : 5 * height // 8, 3 * width // 8 : 5 * width // 8] = 0
    if channels == 1:
        arr = arr[:, :, 0]
    buf = io.BytesIO()
    Image.fromarray(arr, mode=_MODE_FOR_CHANNELS[channels]).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def image_png() -> Callable[..., bytes]:
    """Fixture form of :func:`make_image_png`."""
    return make_image_png


@pytest.fixture
def ring_png() -> Callable[..., bytes]:
    """Fixture form of :func:`make_ring_png`."""
    return make_ring_png


@pytest.fixture
def rect_png() -> Callable[..., bytes]:
    """Fixture form of :func:`make_rect_png`."""
    return make_rect_png


@pytest.fixture
def create_test_png() -> Callable[[int, int, tuple[int, int, int]], bytes]:
    """
    Factory fixture for creating test PNG images.

    Returns:
        A callable that creates PNG bytes for a given width, height, and color
        (default 100x100 gray, kept for existing fixture users).
    """

    def _create(
        width: int = 100,
        height: int = 100,
        color: tuple[int, int, int] = (128, 128, 128),
    ) -> bytes:
        return make_test_png(width, height, color)

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
