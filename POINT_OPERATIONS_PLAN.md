# Point Operations Implementation Plan

This document outlines a comprehensive plan for implementing point-level operations in polars-cv, ensuring cohesion with the existing architecture.

## Current State

### Python API (Defined but Not Implemented)

The `PointNamespace` in `polars_cv/geometry/points.py` defines these operations, but **none have Rust implementations**:

| Method | Rust Function | Type | Status |
|--------|---------------|------|--------|
| `normalize(ref_width, ref_height)` | `point_normalize` | Transform | Not implemented |
| `to_absolute(ref_width, ref_height)` | `point_to_absolute` | Transform | Not implemented |
| `translate(dx, dy)` | `point_translate` | Transform | Not implemented |
| `scale(sx, sy)` | `point_scale` | Transform | Not implemented |
| `distance(other)` | `point_distance` | Pairwise | Not implemented |
| `manhattan_distance(other)` | `point_manhattan_distance` | Pairwise | Not implemented |
| `x()` | N/A (uses Polars built-in) | Extraction | **Working** |
| `y()` | N/A (uses Polars built-in) | Extraction | **Working** |

### Schemas (Complete)

```python
POINT_SCHEMA = pl.Struct({"x": pl.Float64, "y": pl.Float64})
POINT_SET_SCHEMA = pl.List(POINT_SCHEMA)  # Multiple points/keypoints
ANNOTATED_POINT_SCHEMA = pl.Struct({"x": pl.Float64, "y": pl.Float64, "label": pl.Utf8, "confidence": pl.Float64})
```

### view-buffer Library (Available Primitives)

The `view-buffer` geometry module provides:

- `Point::distance_to(&self, other: &Point) -> f64` - Euclidean distance
- `Point::distance_squared_to(&self, other: &Point) -> f64` - Squared distance (faster)
- `perpendicular_distance(point, line_start, line_end) -> f64` - Point to line segment
- `point_in_polygon(point, polygon) -> i32` - Returns 1 (inside), 0 (on boundary), -1 (outside)

---

## Proposed Implementation

### Phase 1: Core Point Operations (Rust Backend)

Create `polars-cv/src/point.rs` implementing the already-defined Python API:

#### 1.1 Kwargs Structure

```rust
#[derive(Debug, Deserialize)]
pub struct PointKwargs {
    #[serde(default)]
    pub ref_width: Option<f64>,
    #[serde(default)]
    pub ref_height: Option<f64>,
    #[serde(default)]
    pub dx: Option<f64>,
    #[serde(default)]
    pub dy: Option<f64>,
    #[serde(default)]
    pub sx: Option<f64>,
    #[serde(default)]
    pub sy: Option<f64>,
}
```

#### 1.2 Point Parsing Helper

```rust
fn parse_point(value: &AnyValue) -> PolarsResult<(f64, f64)> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();
            let mut x = None;
            let mut y = None;
            for (i, field) in fields.iter().enumerate() {
                match field.name().as_str() {
                    "x" => x = extract_f64(&values[i]),
                    "y" => y = extract_f64(&values[i]),
                    _ => {}
                }
            }
            Ok((x.ok_or_else(...)?, y.ok_or_else(...)?))
        }
        AnyValue::Struct(row_idx, struct_arr, fields) => { ... }
        _ => Err(polars_err!(ComputeError: "Expected Point struct"))
    }
}
```

#### 1.3 Transform Operations

| Function | Input | Output | Implementation |
|----------|-------|--------|----------------|
| `point_normalize` | Point | Point | `(x / ref_width, y / ref_height)` |
| `point_to_absolute` | Point | Point | `(x * ref_width, y * ref_height)` |
| `point_translate` | Point | Point | `(x + dx, y + dy)` |
| `point_scale` | Point | Point | `(x * sx, y * sy)` |

#### 1.4 Pairwise Operations

| Function | Inputs | Output | Implementation |
|----------|--------|--------|----------------|
| `point_distance` | Point, Point | Float64 | `sqrt((x2-x1)^2 + (y2-y1)^2)` |
| `point_manhattan_distance` | Point, Point | Float64 | `abs(x2-x1) + abs(y2-y1)` |

---

### Phase 2: Point-to-Contour Operations (New)

These operations compute relationships between points and contours.

#### 2.1 Python API Additions (`points.py`)

```python
def distance_to_contour(self, contour: pl.Expr) -> pl.Expr:
    """
    Compute minimum distance from point to contour boundary.

    Returns 0 if point is on boundary, negative if inside (optional).
    """
    return register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="point_distance_to_contour",
        args=[self._expr, contour],
        is_elementwise=True,
    )

def signed_distance_to_contour(self, contour: pl.Expr) -> pl.Expr:
    """
    Signed distance: negative if inside, positive if outside.
    """
    return register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="point_signed_distance_to_contour",
        args=[self._expr, contour],
        is_elementwise=True,
    )

def nearest_point_on_contour(self, contour: pl.Expr) -> pl.Expr:
    """
    Find the nearest point on the contour boundary.

    Returns: Point struct with x, y coordinates.
    """
    return register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="point_nearest_on_contour",
        args=[self._expr, contour],
        is_elementwise=True,
    )
```

#### 2.2 Rust Implementation

```rust
/// Distance from point to contour boundary (minimum distance to any edge)
#[polars_expr(output_type=Float64)]
fn point_distance_to_contour(inputs: &[Series]) -> PolarsResult<Series> {
    let points = &inputs[0];
    let contours = &inputs[1];

    for i in 0..len {
        let (px, py) = parse_point(&points.get(i)?)?;
        let contour = parse_contour(&contours.get(i)?)?;

        // Compute min distance to all edges
        let mut min_dist = f64::INFINITY;
        for edge in contour.exterior.windows(2) {
            let dist = perpendicular_distance(
                &Point::new(px, py),
                &edge[0],
                &edge[1]
            );
            min_dist = min_dist.min(dist);
        }
        // Close the polygon (last to first)
        let last = contour.exterior.last().unwrap();
        let first = contour.exterior.first().unwrap();
        let dist = perpendicular_distance(&Point::new(px, py), last, first);
        min_dist = min_dist.min(dist);

        results.push(Some(min_dist));
    }
}

/// Signed distance: negative if inside, positive if outside
fn point_signed_distance_to_contour(inputs: &[Series]) -> PolarsResult<Series> {
    // Same as above, but multiply by -1 if point_in_polygon returns 1 (inside)
}
```

#### 2.3 view-buffer Enhancement

Add to `view-buffer/src/geometry/measures.rs`:

```rust
/// Compute distance from point to polygon boundary.
pub fn distance_to_polygon(point: &Point, polygon: &[Point]) -> f64 {
    if polygon.len() < 2 {
        return f64::INFINITY;
    }

    let mut min_dist = f64::INFINITY;
    let n = polygon.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let dist = perpendicular_distance(point, &polygon[i], &polygon[j]);
        min_dist = min_dist.min(dist);
    }

    min_dist
}

/// Find nearest point on polygon boundary.
pub fn nearest_point_on_polygon(point: &Point, polygon: &[Point]) -> Point {
    // Similar, but track the nearest point, not just distance
}
```

---

### Phase 3: Geometric Point Operations (New)

Additional useful point operations.

#### 3.1 Python API Additions

```python
def angle_to(self, other: pl.Expr) -> pl.Expr:
    """
    Compute angle from this point to another in radians.

    Returns angle in [-π, π] using atan2.
    """
    return register_plugin_function(...)

def rotate(self, angle: float, *, origin: pl.Expr | None = None) -> pl.Expr:
    """
    Rotate point around origin by angle (radians).

    Args:
        angle: Rotation angle in radians (counter-clockwise positive)
        origin: Center of rotation (default: coordinate origin 0,0)
    """
    return register_plugin_function(...)

def midpoint(self, other: pl.Expr) -> pl.Expr:
    """
    Compute midpoint between this point and another.
    """
    return register_plugin_function(...)

def interpolate(self, other: pl.Expr, t: float) -> pl.Expr:
    """
    Linear interpolation between two points.

    Args:
        other: Target point
        t: Interpolation parameter (0 = self, 1 = other)
    """
    return register_plugin_function(...)

def within_bbox(self, bbox: pl.Expr) -> pl.Expr:
    """
    Check if point is within bounding box.

    Args:
        bbox: BBox struct with x, y, width, height
    """
    return register_plugin_function(...)
```

#### 3.2 Implementation

| Function | Formula |
|----------|---------|
| `angle_to` | `atan2(other.y - self.y, other.x - self.x)` |
| `rotate` | `x' = cos(θ)(x-ox) - sin(θ)(y-oy) + ox`, `y' = sin(θ)(x-ox) + cos(θ)(y-oy) + oy` |
| `midpoint` | `((x1+x2)/2, (y1+y2)/2)` |
| `interpolate` | `(x1 + t*(x2-x1), y1 + t*(y2-y1))` |
| `within_bbox` | `x >= bbox.x && x <= bbox.x+w && y >= bbox.y && y <= bbox.y+h` |

---

### Phase 4: Point Set Operations (New Namespace)

For working with collections of points (keypoints, landmarks).

#### 4.1 Python API

Create new namespace or extend PointNamespace to handle `POINT_SET_SCHEMA`:

```python
@pl.api.register_expr_namespace("points")
class PointSetNamespace:
    """Operations on point set columns (List of Points)."""

    def centroid(self) -> pl.Expr:
        """Compute centroid of point set."""

    def bounding_box(self) -> pl.Expr:
        """Compute bounding box of point set."""

    def count(self) -> pl.Expr:
        """Count number of points."""

    def convex_hull(self) -> pl.Expr:
        """Compute convex hull as a contour."""

    def get(self, index: int) -> pl.Expr:
        """Get point at index."""

    def translate(self, dx, dy) -> pl.Expr:
        """Translate all points."""

    def scale(self, sx, sy) -> pl.Expr:
        """Scale all points."""
```

#### 4.2 Rust Implementation

Point set operations iterate over the list and apply operations to each point or aggregate:

```rust
#[polars_expr(output_type_func=point_output_type)]
fn points_centroid(inputs: &[Series]) -> PolarsResult<Series> {
    // Parse list of points, compute mean x and mean y
}

#[polars_expr(output_type_func=bbox_output_type)]
fn points_bounding_box(inputs: &[Series]) -> PolarsResult<Series> {
    // Parse list of points, find min/max x/y
}
```

---

### Phase 5: Contour Namespace Extensions (Cross-Namespace)

Add point-centric operations to ContourNamespace for discoverability:

```python
# In contours.py
def distance_to_point(self, point: pl.Expr) -> pl.Expr:
    """Distance from contour boundary to point (symmetric to point.distance_to_contour)."""
    # Internally calls point_distance_to_contour with swapped args
```

---

## File Structure

```
polars-cv/
├── src/
│   ├── lib.rs           # Add: mod point;
│   ├── contour.rs       # Existing
│   └── point.rs         # NEW: Point operations
├── python/polars_cv/
│   └── geometry/
│       ├── points.py    # Update: Add new operations
│       ├── pointset.py  # NEW: Point set namespace
│       └── contours.py  # Update: Add distance_to_point
└── tests/
    ├── test_point_plugin.py     # NEW: Point operation tests
    └── reference/
        └── test_point_ops_ref.py # NEW: Reference implementation tests
```

---

## Implementation Order

### Milestone 1: Core Operations (Enable Existing API)
1. Create `point.rs` with basic structure
2. Implement `point_normalize`, `point_to_absolute`
3. Implement `point_translate`, `point_scale`
4. Implement `point_distance`, `point_manhattan_distance`
5. Add `mod point;` to `lib.rs`
6. Write tests for all Phase 1 operations

### Milestone 2: Point-to-Contour Distance
1. Add `distance_to_polygon` to view-buffer
2. Implement `point_distance_to_contour`
3. Implement `point_signed_distance_to_contour`
4. Implement `point_nearest_on_contour`
5. Write tests

### Milestone 3: Geometric Operations
1. Implement `angle_to`, `midpoint`, `interpolate`
2. Implement `rotate` with origin parameter
3. Implement `within_bbox`
4. Write tests

### Milestone 4: Point Set Operations
1. Create `pointset.py` namespace
2. Implement `points_centroid`, `points_bounding_box`
3. Implement `points_convex_hull` (returns Contour)
4. Implement transform operations on sets
5. Write tests

---

## Testing Strategy

### Unit Tests (`test_point_plugin.py`)

```python
class TestPointTransforms:
    def test_normalize_basic(self):
        df = pl.DataFrame({"pt": [{"x": 50.0, "y": 100.0}]})
        result = df.with_columns(
            normalized=pl.col("pt").point.normalize(100, 200)
        )
        assert result["normalized"][0] == {"x": 0.5, "y": 0.5}

    def test_translate_basic(self):
        df = pl.DataFrame({"pt": [{"x": 10.0, "y": 20.0}]})
        result = df.with_columns(
            moved=pl.col("pt").point.translate(5.0, -10.0)
        )
        assert result["moved"][0] == {"x": 15.0, "y": 10.0}

class TestPointDistances:
    def test_euclidean_distance(self):
        df = pl.DataFrame({
            "p1": [{"x": 0.0, "y": 0.0}],
            "p2": [{"x": 3.0, "y": 4.0}]
        })
        result = df.with_columns(
            dist=pl.col("p1").point.distance(pl.col("p2"))
        )
        assert abs(result["dist"][0] - 5.0) < 1e-10

    def test_manhattan_distance(self):
        df = pl.DataFrame({
            "p1": [{"x": 0.0, "y": 0.0}],
            "p2": [{"x": 3.0, "y": 4.0}]
        })
        result = df.with_columns(
            dist=pl.col("p1").point.manhattan_distance(pl.col("p2"))
        )
        assert abs(result["dist"][0] - 7.0) < 1e-10

class TestPointToContour:
    def test_distance_to_square(self):
        # Point outside square
        df = pl.DataFrame({
            "pt": [{"x": 15.0, "y": 5.0}],
            "contour": [{"exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0}
            ], "holes": []}]
        })
        result = df.with_columns(
            dist=pl.col("pt").point.distance_to_contour(pl.col("contour"))
        )
        assert abs(result["dist"][0] - 5.0) < 1e-10  # 5 units from right edge
```

### Reference Tests (`test_point_ops_ref.py`)

Compare against numpy/scipy implementations:

```python
import numpy as np
from scipy.spatial.distance import cdist

def test_distance_reference():
    """Verify point distance matches scipy."""
    points_a = np.random.rand(100, 2) * 100
    points_b = np.random.rand(100, 2) * 100

    # Reference: scipy
    expected = np.sqrt(np.sum((points_a - points_b)**2, axis=1))

    # Our implementation
    df = pl.DataFrame({
        "p1": [{"x": p[0], "y": p[1]} for p in points_a],
        "p2": [{"x": p[0], "y": p[1]} for p in points_b]
    })
    result = df.with_columns(dist=pl.col("p1").point.distance(pl.col("p2")))

    np.testing.assert_allclose(result["dist"].to_numpy(), expected, rtol=1e-10)
```

---

## Architecture Alignment

### Consistency with ContourNamespace

1. **Same kwargs pattern**: Use `PointKwargs` struct with `#[serde(default)]`
2. **Same error handling**: Use `polars_err!(ComputeError: ...)` macro
3. **Same output types**: `output_type=Float64` for scalars, `output_type_func=` for structs
4. **Same null handling**: Check `value.is_null()` before processing
5. **Same naming**: `point_` prefix for all Rust functions

### Cross-Namespace Compatibility

Operations that work between types should be accessible from either namespace:
- `point.distance_to_contour(contour)` - from point's perspective
- `contour.distance_to_point(point)` - from contour's perspective (calls same Rust function)

### Expression Chaining

All operations return `pl.Expr` enabling chaining:
```python
df.with_columns(
    result=pl.col("pt")
        .point.translate(10, 20)
        .point.scale(2, 2)
        .point.normalize(100, 100)
)
```

---

## Edge Cases to Handle

1. **Null values**: Return null for null inputs
2. **Empty point sets**: Return null for aggregations on empty sets
3. **Degenerate contours**: Handle single-point or two-point "contours"
4. **Zero dimensions**: Handle ref_width=0 or ref_height=0 gracefully
5. **Normalized coordinates out of range**: Allow but don't clamp (user decision)

---

## Performance Considerations

1. **Use `distance_squared_to`** when only comparing distances (avoid sqrt)
2. **Bounding box pre-check** for point-to-contour distance (skip if point is far)
3. **SIMD opportunities** for batch point operations (future optimization)
4. **Parallel iteration** using Polars chunked arrays

---

## Summary

| Phase | Operations | Priority |
|-------|------------|----------|
| 1 | normalize, to_absolute, translate, scale, distance, manhattan_distance | **High** (enables existing API) |
| 2 | distance_to_contour, signed_distance, nearest_point | **High** (user requested) |
| 3 | angle_to, rotate, midpoint, interpolate, within_bbox | Medium |
| 4 | Point set operations (centroid, bbox, convex_hull) | Medium |

Total new Rust functions: **~15**
Total Python API additions: **~12 methods**
Estimated tests: **~30-40 test cases**
