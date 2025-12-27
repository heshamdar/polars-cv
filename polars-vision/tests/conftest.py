"""
Pytest configuration and fixtures for polars-vision tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Add the python source to the path for testing without installation
python_src = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(python_src))

if TYPE_CHECKING:
    pass


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
