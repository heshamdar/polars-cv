"""
Polars Struct schemas for geometric entities.

This module defines the standard schemas for points, contours, and
bounding boxes used throughout polars-cv.

Coordinate System:
- All coordinates are in pixel space with top-left origin.
- X increases to the right, Y increases downward.
- For normalized coordinates, values are in [0, 1] range.

Holes:
- The `holes` field of `CONTOUR_SCHEMA` is the ONLY thing that makes a ring a hole.
- Point order is never interpreted as a hole signal, and no operation requires a
  particular winding: a ring listed in `holes` is a hole whichever way it is wound.

Winding Direction:
- Winding is COMPUTED from point order, not stored explicitly.
- Counter-clockwise (CCW) = positive signed area; clockwise (CW) = negative.
- In image coordinates (Y-down), CCW appears as CW visually.
- It is reported by `.contour.winding()` and set by `.contour.ensure_winding()`.
  Nothing else consults it.
"""

from __future__ import annotations

import polars as pl

# --- Point Schemas ---

POINT_SCHEMA = pl.Struct(
    [
        pl.Field("x", pl.Float64),
        pl.Field("y", pl.Float64),
    ]
)
"""
Single 2D point schema.

Fields:
    x: X coordinate (horizontal, increases right).
    y: Y coordinate (vertical, increases down).

Example:
    ```python
    >>> point = {"x": 50.0, "y": 100.0}
    ```
"""

ANNOTATED_POINT_SCHEMA = pl.Struct(
    [
        pl.Field("x", pl.Float64),
        pl.Field("y", pl.Float64),
        pl.Field("label", pl.String),
        pl.Field("confidence", pl.Float64),
    ]
)
"""
Point with optional metadata.

Fields:
    x: X coordinate.
    y: Y coordinate.
    label: Optional label/class string.
    confidence: Optional confidence score [0, 1].

Useful for detection keypoints with class labels.
"""

POINT_SET_SCHEMA = pl.List(POINT_SCHEMA)
"""
Multiple points (e.g., keypoints, landmarks).

A list of POINT_SCHEMA structs.
"""

# --- Contour Schemas ---

RING_SCHEMA = pl.List(POINT_SCHEMA)
"""
Base ring schema - ordered list of points forming a closed ring.

The ring is implicitly closed; do not repeat the first point at the end.

A ring carries no indication of whether it is an exterior or a hole — that is
decided by which field of `CONTOUR_SCHEMA` it appears in. Point order only
determines the ring's winding, which `.contour.winding()` reports.
"""

CONTOUR_SCHEMA = pl.Struct(
    [
        pl.Field("exterior", RING_SCHEMA),
        pl.Field("holes", pl.List(RING_SCHEMA)),
        pl.Field("is_closed", pl.Boolean),
    ]
)
"""
Contour with exterior boundary and optional holes.

Fields:
    exterior: Outer boundary as a ring of points, in either winding direction.
    holes: List of interior holes, each in either winding direction. This field
           is the sole carrier of hole-ness — see below.
    is_closed: Reserved. Always written as True and never read back; rings are
               implicitly closed. Kept for wire compatibility.

Hole-ness is structural, not directional. A ring is a hole because it appears in
`holes`, not because of how it is wound, and every operation (area, centroid,
`contains_point`, IoU, Dice, rasterization) treats it that way. Flipping a ring
with `.contour.flip()` or `.contour.ensure_winding()` therefore never changes what
region the contour describes; it only changes what `.contour.winding()` reports.

The region is the exterior minus the **union** of the hole rings. Rings that
overlap are not subtracted twice, and a ring nested inside another hole is still a
hole — it simply lies in a part that has already been removed.

Example:
    ```python
    >>> contour = {
    ...     "exterior": [
    ...         {"x": 0, "y": 0},
    ...         {"x": 0, "y": 100},
    ...         {"x": 100, "y": 100},
    ...         {"x": 100, "y": 0}
    ...     ],
    ...     "holes": [],
    ...     "is_closed": True
    ... }
    ```
"""

CONTOUR_SET_SCHEMA = pl.List(CONTOUR_SCHEMA)
"""
Multiple contours (e.g., multiple detected objects).

A list of CONTOUR_SCHEMA structs.
"""

MATCH_RESULT_SCHEMA = pl.Struct(
    [
        pl.Field("pred_idx", pl.List(pl.UInt32)),
        pl.Field("gt_idx", pl.List(pl.UInt32)),
        pl.Field("iou", pl.List(pl.Float64)),
        pl.Field("n_preds", pl.UInt32),
        pl.Field("n_gts", pl.UInt32),
        pl.Field("n_tp", pl.UInt32),
        pl.Field("n_fp", pl.UInt32),
        pl.Field("n_fn", pl.UInt32),
    ]
)
"""
Detection matching output schema.

Fields:
    pred_idx: Prediction indices in evaluation order.
    gt_idx: Matched GT indices for each prediction; null indicates unmatched FP.
    iou: IoU value for each prediction's chosen match (0.0 for unmatched).
    n_preds: Number of predictions in the row.
    n_gts: Number of ground-truth contours in the row.
    n_tp: Number of matched true positives.
    n_fp: Number of unmatched false positives.
    n_fn: Number of unmatched false negatives.
"""

# --- Bounding Box Schema ---

BBOX_SCHEMA = pl.Struct(
    [
        pl.Field("x", pl.Float64),
        pl.Field("y", pl.Float64),
        pl.Field("width", pl.Float64),
        pl.Field("height", pl.Float64),
    ]
)
"""
Axis-aligned bounding box.

Fields:
    x: Top-left X coordinate.
    y: Top-left Y coordinate.
    width: Width of the box.
    height: Height of the box.

Alternative representations (center, corner pairs) can be
converted using utility functions.
"""

# --- Schema Validation Helpers ---


def validate_point(point: dict) -> bool:
    """
    Validate that a dictionary matches POINT_SCHEMA.

    Args:
        point: Dictionary to validate.

    Returns:
        True if valid, False otherwise.
    """
    return (
        isinstance(point, dict)
        and "x" in point
        and "y" in point
        and isinstance(point["x"], (int, float))
        and isinstance(point["y"], (int, float))
    )


def validate_contour(contour: dict) -> bool:
    """
    Validate that a dictionary matches CONTOUR_SCHEMA.

    Args:
        contour: Dictionary to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(contour, dict):
        return False

    if "exterior" not in contour:
        return False

    exterior = contour["exterior"]
    if not isinstance(exterior, list):
        return False

    if len(exterior) < 3:
        return False  # Need at least 3 points for a valid polygon

    if not all(validate_point(p) for p in exterior):
        return False

    holes = contour.get("holes", [])
    if not isinstance(holes, list):
        return False

    for hole in holes:
        if not isinstance(hole, list):
            return False
        if len(hole) < 3:
            return False
        if not all(validate_point(p) for p in hole):
            return False

    return True


def contour_from_points(
    points: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
    is_closed: bool = True,
) -> dict:
    """
    Create a contour dictionary from a list of (x, y) tuples.

    Args:
        points: List of (x, y) tuples for the exterior ring.
        holes: Optional list of holes, each as a list of (x, y) tuples.
        is_closed: Whether the contour is closed.

    Returns:
        Dictionary matching CONTOUR_SCHEMA.

    Example:
        >>> contour = contour_from_points([(0, 0), (100, 0), (100, 100), (0, 100)])
    """
    exterior = [{"x": float(x), "y": float(y)} for x, y in points]

    hole_rings = []
    if holes:
        for hole in holes:
            hole_ring = [{"x": float(x), "y": float(y)} for x, y in hole]
            hole_rings.append(hole_ring)

    return {
        "exterior": exterior,
        "holes": hole_rings,
        "is_closed": is_closed,
    }


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
