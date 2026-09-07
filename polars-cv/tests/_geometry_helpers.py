"""Test-only geometry constructors.

``bbox_from_corners`` / ``bbox_from_center`` build ``BBOX_SCHEMA`` dicts and are
used solely by ``test_geometry_schemas.py``. They lived in
``polars_cv.geometry.schemas`` but had no production caller and were not part of
the public API (no ``__all__``, no re-export), so they moved here beside the one
suite that uses them -- mirroring the ``_metric_refs`` / ``_op_cases``
shared-helper convention. ``contour_from_points`` stayed in ``schemas`` because
it is advertised in ``docs/api/geometry.md`` and used by production-facing tests.
"""

from __future__ import annotations


def bbox_from_corners(x1: float, y1: float, x2: float, y2: float) -> dict:
    """
    Create a bounding box from corner coordinates.

    Args:
        x1: Left X coordinate.
        y1: Top Y coordinate.
        x2: Right X coordinate.
        y2: Bottom Y coordinate.

    Returns:
        Dictionary matching BBOX_SCHEMA.
    """
    return {
        "x": min(x1, x2),
        "y": min(y1, y2),
        "width": abs(x2 - x1),
        "height": abs(y2 - y1),
    }


def bbox_from_center(cx: float, cy: float, width: float, height: float) -> dict:
    """
    Create a bounding box from center and dimensions.

    Args:
        cx: Center X coordinate.
        cy: Center Y coordinate.
        width: Width of the box.
        height: Height of the box.

    Returns:
        Dictionary matching BBOX_SCHEMA.
    """
    return {
        "x": cx - width / 2,
        "y": cy - height / 2,
        "width": width,
        "height": height,
    }
