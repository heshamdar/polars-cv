"""
Validation framework for geometry operations.

This module provides exception classes and validation functions for
geometric operations. Errors are raised when operations are performed
on invalid or incompatible geometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class GeometryValidationError(ValueError):
    """
    Base class for geometry validation errors.

    All geometry-related validation errors inherit from this class,
    allowing users to catch all geometry errors with a single except clause.
    """

    pass


class OpenContourError(GeometryValidationError):
    """
    Raised when an operation requires a closed contour.

    Some operations (like area computation, point-in-polygon tests,
    and rasterization) require the contour to be closed.

    Attributes:
        operation: The operation that was attempted.
    """

    def __init__(self, operation: str) -> None:
        """
        Initialize the error.

        Args:
            operation: Name of the operation that required a closed contour.
        """
        self.operation = operation
        super().__init__(
            f"Operation '{operation}' requires a closed contour. "
            f"Check the is_closed field or close the contour first."
        )


class CoordinateRangeError(GeometryValidationError):
    """
    Raised when coordinates are out of expected range.

    This typically occurs during normalization or absolute coordinate
    conversion when coordinates exceed the reference dimensions.

    Attributes:
        coord_type: Type of coordinate ("x" or "y").
        max_val: The problematic coordinate value.
        ref_dim: The reference dimension that was exceeded.
    """

    def __init__(
        self,
        coord_type: str,
        max_val: float,
        ref_dim: float,
    ) -> None:
        """
        Initialize the error.

        Args:
            coord_type: Which coordinate exceeded bounds ("x" or "y").
            max_val: The coordinate value that exceeded bounds.
            ref_dim: The reference dimension limit.
        """
        self.coord_type = coord_type
        self.max_val = max_val
        self.ref_dim = ref_dim
        super().__init__(
            f"{coord_type} coordinate {max_val} exceeds reference dimension {ref_dim}. "
            f"Ensure coordinates match the reference dimensions."
        )


class InvalidContourError(GeometryValidationError):
    """
    Raised when a contour is geometrically invalid.

    Examples include:
    - Fewer than 3 points (not a valid polygon)
    - Self-intersecting contour
    - Hole outside exterior boundary

    Attributes:
        reason: Description of why the contour is invalid.
    """

    def __init__(self, reason: str) -> None:
        """
        Initialize the error.

        Args:
            reason: Description of the invalidity.
        """
        self.reason = reason
        super().__init__(f"Invalid contour: {reason}")


class InsufficientPointsError(InvalidContourError):
    """
    Raised when a contour has too few points.

    A valid polygon requires at least 3 points.

    Attributes:
        got: Number of points found.
        min_required: Minimum required points.
    """

    def __init__(self, got: int, min_required: int = 3) -> None:
        """
        Initialize the error.

        Args:
            got: Number of points found.
            min_required: Minimum required points (default 3).
        """
        self.got = got
        self.min_required = min_required
        super().__init__(
            f"Contour has {got} points, but at least {min_required} are required"
        )


# --- Validation Functions ---


def validate_for_area(is_closed: bool) -> None:
    """
    Validate that a contour can be used for area computation.

    Args:
        is_closed: Whether the contour is closed.

    Raises:
        OpenContourError: If the contour is not closed.
    """
    if not is_closed:
        raise OpenContourError("area")


def validate_for_rasterize(is_closed: bool) -> None:
    """
    Validate that a contour can be rasterized.

    Args:
        is_closed: Whether the contour is closed.

    Raises:
        OpenContourError: If the contour is not closed.
    """
    if not is_closed:
        raise OpenContourError("rasterize")


def validate_point_count(n_points: int, min_required: int = 3) -> None:
    """
    Validate that a contour has enough points.

    Args:
        n_points: Number of points in the contour.
        min_required: Minimum required points.

    Raises:
        InsufficientPointsError: If there are too few points.
    """
    if n_points < min_required:
        raise InsufficientPointsError(n_points, min_required)


def validate_normalize_coords(
    max_x: float,
    max_y: float,
    ref_width: float,
    ref_height: float,
) -> None:
    """
    Validate that coordinates are within reference dimensions.

    Args:
        max_x: Maximum X coordinate in the contour.
        max_y: Maximum Y coordinate in the contour.
        ref_width: Reference width for normalization.
        ref_height: Reference height for normalization.

    Raises:
        CoordinateRangeError: If any coordinate exceeds reference dimensions.
    """
    if max_x > ref_width:
        raise CoordinateRangeError("x", max_x, ref_width)
    if max_y > ref_height:
        raise CoordinateRangeError("y", max_y, ref_height)
