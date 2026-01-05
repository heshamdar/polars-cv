"""
Shared fixtures for reference tests.

These fixtures provide standard test data for validating polars-cv
operations against OpenCV and NumPy reference implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


# --- Image Fixtures ---


@pytest.fixture(scope="session")
def test_image_rgb() -> np.ndarray:
    """
    Standard RGB test image (256x256x3).

    Uses a fixed seed for reproducibility across test runs.

    Returns:
        np.ndarray: RGB image with shape (256, 256, 3) and dtype uint8.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def test_image_gray() -> np.ndarray:
    """
    Standard grayscale test image (256x256).

    Uses a fixed seed for reproducibility across test runs.

    Returns:
        np.ndarray: Grayscale image with shape (256, 256) and dtype uint8.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def sample_images() -> tuple[np.ndarray, np.ndarray]:
    """
    Two sample images for binary operations testing.

    Returns:
        tuple: Two RGB images with shape (100, 100, 3) and dtype uint8.
    """
    rng = np.random.default_rng(123)
    img1 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    return img1, img2


@pytest.fixture
def binary_mask() -> np.ndarray:
    """
    Binary mask for masking operations.

    Creates a 100x100 mask with a center square (25:75, 25:75) set to 1.

    Returns:
        np.ndarray: Binary mask with shape (100, 100) and dtype uint8.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1  # Center square
    return mask


# --- Contour Fixtures ---


def _make_circle(cx: float, cy: float, r: float, n: int = 64) -> np.ndarray:
    """
    Create a circular contour.

    Args:
        cx: Center x coordinate.
        cy: Center y coordinate.
        r: Radius.
        n: Number of points.

    Returns:
        np.ndarray: Contour points with shape (n, 2).
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


@pytest.fixture
def standard_contours() -> dict[str, np.ndarray]:
    """
    Collection of standard contours for testing.

    Provides common shapes: square, triangle, circle.
    All use CCW winding (positive area).

    Returns:
        dict: Named contours as float64 arrays with shape (n, 2).
    """
    return {
        "square": np.array([[10, 10], [10, 90], [90, 90], [90, 10]], dtype=np.float64),
        "triangle": np.array([[50, 10], [10, 90], [90, 90]], dtype=np.float64),
        "circle": _make_circle(50, 50, 40, 64).astype(np.float64),
    }


@pytest.fixture
def simple_contour() -> np.ndarray:
    """
    Simple square contour (CCW winding).

    Returns:
        np.ndarray: Square contour with shape (4, 2) and dtype float32.
    """
    return np.array([[10, 10], [10, 90], [90, 90], [90, 10]], dtype=np.float32)


@pytest.fixture
def triangle_contour() -> np.ndarray:
    """
    Simple triangle contour for rasterization tests.

    Returns:
        np.ndarray: Triangle contour with shape (3, 2) and dtype int32.
    """
    return np.array([[50, 10], [10, 90], [90, 90]], dtype=np.int32)


@pytest.fixture
def contour_with_hole() -> tuple[np.ndarray, np.ndarray]:
    """
    Square contour with a square hole inside.

    The outer contour is CCW (positive area), the hole is CW (negative area).

    Returns:
        tuple: (outer_contour, hole_contour) as float32 arrays.
    """
    outer = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=np.float32)
    # Create circular hole (CW winding for hole - reversed)
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    hole = np.column_stack([50 + 20 * np.cos(angles), 50 + 20 * np.sin(angles)]).astype(
        np.float32
    )[::-1]  # Reverse for CW
    return outer, hole


@pytest.fixture
def overlapping_contours() -> tuple[np.ndarray, np.ndarray]:
    """
    Two overlapping square contours for IoU/Dice tests.

    c1: Square from (10,10) to (60,60)
    c2: Square from (40,40) to (90,90)
    Overlap region: (40,40) to (60,60) = 20x20 = 400 pixels

    Returns:
        tuple: (contour1, contour2) as int32 arrays.
    """
    c1 = np.array([[10, 10], [10, 60], [60, 60], [60, 10]], dtype=np.int32)
    c2 = np.array([[40, 40], [40, 90], [90, 90], [90, 40]], dtype=np.int32)
    return c1, c2


@pytest.fixture
def non_overlapping_contours() -> tuple[np.ndarray, np.ndarray]:
    """
    Two non-overlapping square contours for IoU = 0 test.

    Returns:
        tuple: (contour1, contour2) as int32 arrays.
    """
    c1 = np.array([[10, 10], [10, 40], [40, 40], [40, 10]], dtype=np.int32)
    c2 = np.array([[60, 60], [60, 90], [90, 90], [90, 60]], dtype=np.int32)
    return c1, c2


# --- Mask Fixtures ---


@pytest.fixture
def binary_mask_with_shape() -> np.ndarray:
    """
    Binary mask with known rectangular shape.

    Rectangle from (20,20) to (80,80).

    Returns:
        np.ndarray: Binary mask with shape (100, 100) and dtype uint8.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    return mask


@pytest.fixture
def mask_with_hole() -> np.ndarray:
    """
    Mask with outer region and inner hole.

    Outer: (10,10) to (90,90)
    Hole: (30,30) to (70,70)

    Returns:
        np.ndarray: Binary mask with shape (100, 100) and dtype uint8.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    mask[30:70, 30:70] = 0  # Hole
    return mask
