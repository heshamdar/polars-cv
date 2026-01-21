"""
Geometry module for polars-cv.

This module provides geometric entity schemas (points, contours, bounding boxes)
and operations for working with them in Polars DataFrames.

Key schemas:
- `POINT_SCHEMA`: Single 2D point with x, y coordinates
- `CONTOUR_SCHEMA`: Contour with exterior ring and optional holes
- `BBOX_SCHEMA`: Axis-aligned bounding box

Key namespaces:
- `.contour`: Operations on contour columns (area, perimeter, winding, etc.)
- `.point`: Operations on point columns (normalize, translate, scale, etc.)

Example:
```python
    >>> import polars as pl
    >>> from polars_cv.geometry import CONTOUR_SCHEMA
    >>>
    >>> # Create contour data
    >>> df = pl.DataFrame({
    ...     "contour": [
    ...         {"exterior": [{"x": 0, "y": 0}, {"x": 100, "y": 0},
    ...                       {"x": 100, "y": 100}, {"x": 0, "y": 100}],
    ...          "holes": [],
    ...          "is_closed": True}
    ...     ]
    ... }).cast({"contour": CONTOUR_SCHEMA})
    >>>
    >>> # Compute area
    >>> df.with_columns(area=pl.col("contour").contour.area())
```
"""

from .schemas import (
    ANNOTATED_POINT_SCHEMA,
    BBOX_SCHEMA,
    CONTOUR_SCHEMA,
    CONTOUR_SET_SCHEMA,
    POINT_SCHEMA,
    POINT_SET_SCHEMA,
    RING_SCHEMA,
)
from .validation import (
    CoordinateRangeError,
    GeometryValidationError,
    InvalidContourError,
    OpenContourError,
)

__all__ = [
    # Schemas
    "POINT_SCHEMA",
    "ANNOTATED_POINT_SCHEMA",
    "POINT_SET_SCHEMA",
    "RING_SCHEMA",
    "CONTOUR_SCHEMA",
    "CONTOUR_SET_SCHEMA",
    "BBOX_SCHEMA",
    # Validation errors
    "GeometryValidationError",
    "OpenContourError",
    "CoordinateRangeError",
    "InvalidContourError",
]
