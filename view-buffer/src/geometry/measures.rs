//! Geometric measures for contours.
//!
//! Implements area, perimeter, centroid, bounding box, and winding direction.

use super::contour::{BoundingBox, Contour, Point, Winding};

/// Computes the signed area of a polygon using the Shoelace formula.
///
/// Positive area indicates counter-clockwise winding (in standard math coords).
/// Negative area indicates clockwise winding.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// Signed area value
pub fn signed_area(points: &[Point]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let n = points.len();
    let mut area = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }

    area / 2.0
}

/// Computes the area of a contour.
///
/// For contours with holes, the hole areas are subtracted.
///
/// # Arguments
/// * `contour` - The contour to measure
/// * `signed` - If true, returns signed area (negative for CW winding)
///
/// # Returns
/// Area value (absolute if `signed` is false)
pub fn area(contour: &Contour, signed: bool) -> f64 {
    let exterior_area = signed_area(&contour.exterior);

    let holes_area: f64 = contour
        .holes
        .iter()
        .map(|hole| signed_area(hole).abs())
        .sum();

    let net_area = exterior_area.abs() - holes_area;

    if signed {
        if exterior_area >= 0.0 {
            net_area
        } else {
            -net_area
        }
    } else {
        net_area.abs()
    }
}

/// Computes the perimeter (arc length) of a polygon.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// Total perimeter length
pub fn perimeter_of_ring(points: &[Point]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    let n = points.len();
    let mut perimeter = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        perimeter += points[i].distance_to(&points[j]);
    }

    perimeter
}

/// Computes the perimeter of a contour including holes.
///
/// # Arguments
/// * `contour` - The contour to measure
///
/// # Returns
/// Total perimeter length
pub fn perimeter(contour: &Contour) -> f64 {
    let exterior_perimeter = perimeter_of_ring(&contour.exterior);

    let holes_perimeter: f64 = contour
        .holes
        .iter()
        .map(|hole| perimeter_of_ring(hole))
        .sum();

    exterior_perimeter + holes_perimeter
}

/// Computes the centroid (center of mass) of a polygon.
///
/// Uses the formula for centroid of a simple polygon.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// Centroid point, or (0, 0) if polygon has less than 3 points
pub fn centroid_of_ring(points: &[Point]) -> Point {
    if points.len() < 3 {
        if points.is_empty() {
            return Point::new(0.0, 0.0);
        }
        // For degenerate cases, return mean of points
        let sum_x: f64 = points.iter().map(|p| p.x).sum();
        let sum_y: f64 = points.iter().map(|p| p.y).sum();
        return Point::new(sum_x / points.len() as f64, sum_y / points.len() as f64);
    }

    let n = points.len();
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut signed_area = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let cross = points[i].x * points[j].y - points[j].x * points[i].y;
        signed_area += cross;
        cx += (points[i].x + points[j].x) * cross;
        cy += (points[i].y + points[j].y) * cross;
    }

    signed_area /= 2.0;

    if signed_area.abs() < 1e-10 {
        // Degenerate polygon, return mean of points
        let sum_x: f64 = points.iter().map(|p| p.x).sum();
        let sum_y: f64 = points.iter().map(|p| p.y).sum();
        return Point::new(sum_x / points.len() as f64, sum_y / points.len() as f64);
    }

    cx /= 6.0 * signed_area;
    cy /= 6.0 * signed_area;

    Point::new(cx, cy)
}

/// Computes the centroid of a contour.
///
/// For contours with holes, uses weighted average based on area.
///
/// # Arguments
/// * `contour` - The contour to measure
///
/// # Returns
/// Centroid point
pub fn centroid(contour: &Contour) -> Point {
    if contour.holes.is_empty() {
        return centroid_of_ring(&contour.exterior);
    }

    // For contours with holes, use weighted centroid
    let ext_area = signed_area(&contour.exterior).abs();
    let ext_centroid = centroid_of_ring(&contour.exterior);

    let mut total_area = ext_area;
    let mut weighted_x = ext_centroid.x * ext_area;
    let mut weighted_y = ext_centroid.y * ext_area;

    for hole in &contour.holes {
        let hole_area = signed_area(hole).abs();
        let hole_centroid = centroid_of_ring(hole);
        // Subtract hole contribution
        total_area -= hole_area;
        weighted_x -= hole_centroid.x * hole_area;
        weighted_y -= hole_centroid.y * hole_area;
    }

    if total_area.abs() < 1e-10 {
        return ext_centroid;
    }

    Point::new(weighted_x / total_area, weighted_y / total_area)
}

/// Computes the bounding box of a contour.
///
/// # Arguments
/// * `contour` - The contour to measure
///
/// # Returns
/// Bounding box, or None if contour is empty
pub fn bounding_box(contour: &Contour) -> Option<BoundingBox> {
    contour.bounding_box()
}

/// Determines the winding direction of a polygon.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// Winding direction
pub fn winding(points: &[Point]) -> Winding {
    let area = signed_area(points);
    if area >= 0.0 {
        Winding::CounterClockwise
    } else {
        Winding::Clockwise
    }
}

/// Determines the winding direction of a contour's exterior.
///
/// # Arguments
/// * `contour` - The contour to check
///
/// # Returns
/// Winding direction of the exterior ring
pub fn contour_winding(contour: &Contour) -> Winding {
    winding(&contour.exterior)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_contour() -> Contour {
        Contour::from_tuples(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    }

    fn ccw_square() -> Contour {
        // CCW in standard math coordinates (y-up)
        Contour::from_tuples(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    }

    #[test]
    fn test_signed_area_ccw() {
        let contour = ccw_square();
        let area = signed_area(&contour.exterior);
        // CCW in standard coords = positive area
        assert!(area > 0.0);
        assert!((area - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_signed_area_cw() {
        let contour = Contour::from_tuples(&[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]);
        let area = signed_area(&contour.exterior);
        // CW in standard coords = negative area
        assert!(area < 0.0);
        assert!((area.abs() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_area_unsigned() {
        let contour = square_contour();
        let a = area(&contour, false);
        assert!((a - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_area_with_hole() {
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
        let a = area(&contour, false);
        // 100 - 36 = 64
        assert!((a - 64.0).abs() < 0.01);
    }

    #[test]
    fn test_perimeter() {
        let contour = square_contour();
        let p = perimeter(&contour);
        assert!((p - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_centroid() {
        let contour = square_contour();
        let c = centroid(&contour);
        assert!((c.x - 5.0).abs() < 0.01);
        assert!((c.y - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_winding_ccw() {
        let contour = ccw_square();
        assert_eq!(contour_winding(&contour), Winding::CounterClockwise);
    }

    #[test]
    fn test_winding_cw() {
        let contour = Contour::from_tuples(&[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]);
        assert_eq!(contour_winding(&contour), Winding::Clockwise);
    }
}
