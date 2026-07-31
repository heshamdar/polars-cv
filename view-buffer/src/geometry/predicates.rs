//! Geometric predicates for contours.
//!
//! Implements convexity check and point-in-polygon tests.
//!
//! Both delegate to `geo`, whose implementations use exact (robust) orientation
//! predicates rather than epsilon comparisons.

use super::contour::{Contour, Point};
use geo::coordinate_position::{CoordPos, CoordinatePosition};
use geo::{coord, IsConvex};

/// Checks if a polygon is convex.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// true if the polygon is convex. Degenerate rings (fewer than 3 points) are
/// considered convex.
pub fn is_convex(points: &[Point]) -> bool {
    if points.len() < 3 {
        return true;
    }
    Contour::new(points.to_vec())
        .to_geo()
        .exterior()
        .is_convex()
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

/// Tests if a point is inside a polygon.
///
/// # Arguments
/// * `point` - The point to test
/// * `polygon` - Slice of points forming a closed polygon
///
/// # Returns
/// * `1` if point is inside
/// * `0` if point is on the boundary
/// * `-1` if point is outside
pub fn point_in_polygon(point: &Point, polygon: &[Point]) -> i32 {
    if polygon.len() < 3 {
        return -1;
    }
    point_in_contour(point, &Contour::new(polygon.to_vec()))
}

/// Tests if a point is inside a contour, treating holes as outside.
///
/// # Arguments
/// * `point` - The point to test
/// * `contour` - The contour to test against
///
/// # Returns
/// * `1` if point is inside (not in a hole)
/// * `0` if point is on the boundary of the exterior or of a hole
/// * `-1` if point is outside or inside a hole
pub fn point_in_contour(point: &Point, contour: &Contour) -> i32 {
    if contour.exterior.len() < 3 {
        return -1;
    }

    match contour
        .to_geo()
        .coordinate_position(&coord! { x: point.x, y: point.y })
    {
        CoordPos::Inside => 1,
        CoordPos::OnBoundary => 0,
        CoordPos::Outside => -1,
    }
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
