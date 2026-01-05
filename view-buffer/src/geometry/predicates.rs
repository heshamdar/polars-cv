//! Geometric predicates for contours.
//!
//! Implements convexity check and point-in-polygon tests.

use super::contour::{Contour, Point};

/// Checks if a polygon is convex.
///
/// A polygon is convex if all cross products of consecutive edge pairs
/// have the same sign.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// true if the polygon is convex
pub fn is_convex(points: &[Point]) -> bool {
    if points.len() < 3 {
        return true; // Degenerate cases are considered convex
    }

    let n = points.len();
    let mut sign: Option<bool> = None;

    for i in 0..n {
        let p0 = &points[i];
        let p1 = &points[(i + 1) % n];
        let p2 = &points[(i + 2) % n];

        let cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);

        // Skip collinear points (cross ≈ 0)
        if cross.abs() < 1e-10 {
            continue;
        }

        let current_sign = cross > 0.0;

        match sign {
            None => sign = Some(current_sign),
            Some(s) if s != current_sign => return false,
            _ => {}
        }
    }

    true
}

/// Checks if a contour is convex.
///
/// Only checks the exterior ring; contours with holes are not convex.
///
/// # Arguments
/// * `contour` - The contour to check
///
/// # Returns
/// true if the contour is convex and has no holes
pub fn contour_is_convex(contour: &Contour) -> bool {
    if contour.has_holes() {
        return false;
    }
    is_convex(&contour.exterior)
}

/// Tests if a point is inside a polygon using the ray casting algorithm.
///
/// # Arguments
/// * `point` - The point to test
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// * `1` if point is inside
/// * `0` if point is on the boundary
/// * `-1` if point is outside
pub fn point_in_polygon(point: &Point, polygon: &[Point]) -> i32 {
    if polygon.len() < 3 {
        return -1;
    }

    let n = polygon.len();
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let pi = &polygon[i];
        let pj = &polygon[j];

        // Check if point is on the edge
        if is_point_on_segment(point, pi, pj) {
            return 0;
        }

        // Ray casting
        if ((pi.y > point.y) != (pj.y > point.y))
            && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }

        j = i;
    }

    if inside {
        1
    } else {
        -1
    }
}

/// Checks if a point lies on a line segment.
fn is_point_on_segment(point: &Point, seg_start: &Point, seg_end: &Point) -> bool {
    // Check collinearity using cross product
    let cross = (point.y - seg_start.y) * (seg_end.x - seg_start.x)
        - (point.x - seg_start.x) * (seg_end.y - seg_start.y);

    if cross.abs() > 1e-10 {
        return false;
    }

    // Check if point is within bounding box of segment
    let min_x = seg_start.x.min(seg_end.x);
    let max_x = seg_start.x.max(seg_end.x);
    let min_y = seg_start.y.min(seg_end.y);
    let max_y = seg_start.y.max(seg_end.y);

    point.x >= min_x - 1e-10
        && point.x <= max_x + 1e-10
        && point.y >= min_y - 1e-10
        && point.y <= max_y + 1e-10
}

/// Tests if a point is inside a contour (including holes).
///
/// # Arguments
/// * `point` - The point to test
/// * `contour` - The contour to test against
///
/// # Returns
/// * `1` if point is inside (not in a hole)
/// * `0` if point is on the boundary
/// * `-1` if point is outside or inside a hole
pub fn point_in_contour(point: &Point, contour: &Contour) -> i32 {
    let exterior_result = point_in_polygon(point, &contour.exterior);

    if exterior_result == 0 {
        return 0; // On exterior boundary
    }

    if exterior_result < 0 {
        return -1; // Outside exterior
    }

    // Check if point is inside any hole
    for hole in &contour.holes {
        let hole_result = point_in_polygon(point, hole);
        if hole_result == 0 {
            return 0; // On hole boundary
        }
        if hole_result > 0 {
            return -1; // Inside hole = outside contour
        }
    }

    1 // Inside exterior and not in any hole
}

/// Checks if a contour contains a specific point.
///
/// Convenience wrapper around `point_in_contour`.
///
/// # Arguments
/// * `contour` - The contour to check
/// * `x` - X coordinate of the point
/// * `y` - Y coordinate of the point
///
/// # Returns
/// true if the point is inside the contour
pub fn contains_point(contour: &Contour, x: f64, y: f64) -> bool {
    point_in_contour(&Point::new(x, y), contour) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_contour() -> Contour {
        Contour::from_tuples(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    }

    #[test]
    fn test_is_convex_square() {
        let contour = square_contour();
        assert!(contour_is_convex(&contour));
    }

    #[test]
    fn test_is_convex_l_shape() {
        // L-shaped (concave)
        let contour = Contour::from_tuples(&[
            (0.0, 0.0),
            (0.0, 10.0),
            (5.0, 10.0),
            (5.0, 5.0),
            (10.0, 5.0),
            (10.0, 0.0),
        ]);
        assert!(!contour_is_convex(&contour));
    }

    #[test]
    fn test_is_convex_with_holes() {
        let exterior = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let hole = vec![
            Point::new(2.0, 2.0),
            Point::new(8.0, 2.0),
            Point::new(8.0, 8.0),
            Point::new(2.0, 8.0),
        ];
        let contour = Contour::with_holes(exterior, vec![hole]);
        assert!(!contour_is_convex(&contour)); // Has holes = not convex
    }

    #[test]
    fn test_point_inside() {
        let contour = square_contour();
        assert!(contains_point(&contour, 5.0, 5.0));
    }

    #[test]
    fn test_point_outside() {
        let contour = square_contour();
        assert!(!contains_point(&contour, 15.0, 5.0));
    }

    #[test]
    fn test_point_on_boundary() {
        let contour = square_contour();
        let result = point_in_contour(&Point::new(5.0, 0.0), &contour);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_point_in_hole() {
        let exterior = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let hole = vec![
            Point::new(3.0, 3.0),
            Point::new(7.0, 3.0),
            Point::new(7.0, 7.0),
            Point::new(3.0, 7.0),
        ];
        let contour = Contour::with_holes(exterior, vec![hole]);

        assert!(!contains_point(&contour, 5.0, 5.0)); // Inside hole
        assert!(contains_point(&contour, 1.0, 1.0)); // Inside exterior, not in hole
    }
}
