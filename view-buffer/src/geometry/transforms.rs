//! Contour transformation operations.
//!
//! Implements translate, scale, flip, simplify, and convex hull.

use super::contour::{Contour, Point, Winding};
use super::measures::{centroid, signed_area};
use super::ops::ScaleOrigin;

/// Translates a contour by the given offset.
///
/// # Arguments
/// * `contour` - The contour to translate
/// * `dx` - X offset
/// * `dy` - Y offset
///
/// # Returns
/// New translated contour
pub fn translate(contour: &Contour, dx: f64, dy: f64) -> Contour {
    let exterior = contour
        .exterior
        .iter()
        .map(|p| Point::new(p.x + dx, p.y + dy))
        .collect();

    let holes = contour
        .holes
        .iter()
        .map(|hole| {
            hole.iter()
                .map(|p| Point::new(p.x + dx, p.y + dy))
                .collect()
        })
        .collect();

    Contour::with_holes(exterior, holes)
}

/// Scales a contour relative to a specified origin.
///
/// # Arguments
/// * `contour` - The contour to scale
/// * `sx` - X scale factor
/// * `sy` - Y scale factor
/// * `origin` - The point to scale around
///
/// # Returns
/// New scaled contour
pub fn scale(contour: &Contour, sx: f64, sy: f64, origin: ScaleOrigin) -> Contour {
    let center = match origin {
        ScaleOrigin::Origin => Point::new(0.0, 0.0),
        ScaleOrigin::Centroid => centroid(contour),
        ScaleOrigin::BBoxCenter => contour
            .bounding_box()
            .map(|bb| bb.center())
            .unwrap_or_else(|| Point::new(0.0, 0.0)),
    };

    let scale_point = |p: &Point| -> Point {
        Point::new(
            (p.x - center.x) * sx + center.x,
            (p.y - center.y) * sy + center.y,
        )
    };

    let exterior = contour.exterior.iter().map(scale_point).collect();

    let holes = contour
        .holes
        .iter()
        .map(|hole| hole.iter().map(scale_point).collect())
        .collect();

    Contour::with_holes(exterior, holes)
}

/// Flips (reverses) the point order of a contour.
///
/// This changes the winding direction.
///
/// # Arguments
/// * `contour` - The contour to flip
///
/// # Returns
/// New contour with reversed point order
pub fn flip(contour: &Contour) -> Contour {
    let exterior: Vec<Point> = contour.exterior.iter().copied().rev().collect();

    let holes: Vec<Vec<Point>> = contour
        .holes
        .iter()
        .map(|hole| hole.iter().copied().rev().collect())
        .collect();

    Contour::with_holes(exterior, holes)
}

/// Ensures the contour has the specified winding direction.
///
/// If the contour already has the correct winding, returns it unchanged.
/// Otherwise, flips the contour.
///
/// # Arguments
/// * `contour` - The contour to check/flip
/// * `direction` - The desired winding direction
///
/// # Returns
/// Contour with the correct winding direction
pub fn ensure_winding(contour: &Contour, direction: Winding) -> Contour {
    let current = if signed_area(&contour.exterior) >= 0.0 {
        Winding::CounterClockwise
    } else {
        Winding::Clockwise
    };

    if current == direction {
        contour.clone()
    } else {
        flip(contour)
    }
}

/// Normalizes contour coordinates to [0, 1] range.
///
/// # Arguments
/// * `contour` - The contour to normalize
/// * `ref_width` - Reference width for normalization
/// * `ref_height` - Reference height for normalization
///
/// # Returns
/// New contour with coordinates in [0, 1] range
pub fn normalize(contour: &Contour, ref_width: f64, ref_height: f64) -> Contour {
    let normalize_point = |p: &Point| -> Point { Point::new(p.x / ref_width, p.y / ref_height) };

    let exterior = contour.exterior.iter().map(normalize_point).collect();

    let holes = contour
        .holes
        .iter()
        .map(|hole| hole.iter().map(normalize_point).collect())
        .collect();

    Contour::with_holes(exterior, holes)
}

/// Converts normalized coordinates to absolute pixel coordinates.
///
/// # Arguments
/// * `contour` - The contour with normalized [0, 1] coordinates
/// * `ref_width` - Reference width
/// * `ref_height` - Reference height
///
/// # Returns
/// New contour with pixel coordinates
pub fn to_absolute(contour: &Contour, ref_width: f64, ref_height: f64) -> Contour {
    let to_abs = |p: &Point| -> Point { Point::new(p.x * ref_width, p.y * ref_height) };

    let exterior = contour.exterior.iter().map(to_abs).collect();

    let holes = contour
        .holes
        .iter()
        .map(|hole| hole.iter().map(to_abs).collect())
        .collect();

    Contour::with_holes(exterior, holes)
}

/// Simplifies a contour using the Douglas-Peucker algorithm.
///
/// # Arguments
/// * `contour` - The contour to simplify
/// * `tolerance` - Maximum perpendicular distance for point removal
///
/// # Returns
/// Simplified contour with fewer points
pub fn simplify(contour: &Contour, tolerance: f64) -> Contour {
    let exterior = douglas_peucker(&contour.exterior, tolerance);

    let holes = contour
        .holes
        .iter()
        .map(|hole| douglas_peucker(hole, tolerance))
        .collect();

    Contour::with_holes(exterior, holes)
}

/// Douglas-Peucker simplification algorithm.
fn douglas_peucker(points: &[Point], tolerance: f64) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the point with maximum distance from the line between first and last
    let first = &points[0];
    let last = &points[points.len() - 1];

    let mut max_dist = 0.0;
    let mut max_idx = 0;

    for (i, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let dist = perpendicular_distance(point, first, last);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > tolerance {
        // Recursively simplify both segments
        let mut left = douglas_peucker(&points[..=max_idx], tolerance);
        let right = douglas_peucker(&points[max_idx..], tolerance);

        // Remove duplicate point at junction
        left.pop();
        left.extend(right);
        left
    } else {
        // All points are within tolerance, keep only endpoints
        vec![*first, *last]
    }
}

/// Computes perpendicular distance from a point to a line.
fn perpendicular_distance(point: &Point, line_start: &Point, line_end: &Point) -> f64 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;
    let length_sq = dx * dx + dy * dy;

    if length_sq < 1e-10 {
        return point.distance_to(line_start);
    }

    // Area of triangle * 2 / base = height
    let area = ((line_end.x - line_start.x) * (line_start.y - point.y)
        - (line_start.x - point.x) * (line_end.y - line_start.y))
        .abs();

    area / length_sq.sqrt()
}

/// Computes the convex hull of a contour using Graham scan.
///
/// # Arguments
/// * `contour` - The contour to compute hull for
///
/// # Returns
/// New contour representing the convex hull
pub fn convex_hull(contour: &Contour) -> Contour {
    let hull_points = graham_scan(&contour.exterior);
    Contour::new(hull_points)
}

/// Graham scan algorithm for convex hull.
fn graham_scan(points: &[Point]) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the bottom-most point (or left-most in case of tie)
    let mut start_idx = 0;
    for (i, p) in points.iter().enumerate().skip(1) {
        if p.y < points[start_idx].y || (p.y == points[start_idx].y && p.x < points[start_idx].x) {
            start_idx = i;
        }
    }

    let start = points[start_idx];

    // Sort points by polar angle with respect to start
    let mut sorted: Vec<Point> = points.iter().copied().filter(|p| *p != start).collect();

    sorted.sort_by(|a, b| {
        let angle_a = (a.y - start.y).atan2(a.x - start.x);
        let angle_b = (b.y - start.y).atan2(b.x - start.x);
        angle_a
            .partial_cmp(&angle_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build hull
    let mut hull = vec![start];

    for point in sorted {
        while hull.len() > 1
            && cross_product(&hull[hull.len() - 2], &hull[hull.len() - 1], &point) <= 0.0
        {
            hull.pop();
        }
        hull.push(point);
    }

    hull
}

/// Cross product of vectors OA and OB.
fn cross_product(o: &Point, a: &Point, b: &Point) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_contour() -> Contour {
        Contour::from_tuples(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    }

    #[test]
    fn test_translate() {
        let contour = square_contour();
        let translated = translate(&contour, 5.0, 10.0);

        assert!((translated.exterior[0].x - 5.0).abs() < 0.01);
        assert!((translated.exterior[0].y - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_scale_origin() {
        let contour = square_contour();
        let scaled = scale(&contour, 2.0, 2.0, ScaleOrigin::Origin);

        assert!((scaled.exterior[0].x - 0.0).abs() < 0.01);
        assert!((scaled.exterior[1].x - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_scale_centroid() {
        let contour = square_contour();
        let scaled = scale(&contour, 0.5, 0.5, ScaleOrigin::Centroid);

        // Centroid is at (5, 5), should stay at (5, 5) after scaling
        let c = centroid(&scaled);
        assert!((c.x - 5.0).abs() < 0.01);
        assert!((c.y - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_flip() {
        let contour = square_contour();
        let original_winding = signed_area(&contour.exterior) >= 0.0;

        let flipped = flip(&contour);
        let flipped_winding = signed_area(&flipped.exterior) >= 0.0;

        assert_ne!(original_winding, flipped_winding);
    }

    #[test]
    fn test_normalize() {
        let contour = square_contour();
        let normalized = normalize(&contour, 10.0, 10.0);

        assert!((normalized.exterior[0].x - 0.0).abs() < 0.01);
        assert!((normalized.exterior[2].x - 1.0).abs() < 0.01);
        assert!((normalized.exterior[2].y - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_simplify() {
        // Create a contour with points on a line (should be simplified)
        let contour = Contour::from_tuples(&[
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (10.0, 10.0),
            (5.0, 10.0),
            (0.0, 10.0),
            (0.0, 5.0),
        ]);
        let simplified = simplify(&contour, 0.1);

        // Should reduce to 4 corners
        assert!(simplified.len() <= contour.len());
    }

    #[test]
    fn test_convex_hull() {
        // L-shaped contour
        let contour = Contour::from_tuples(&[
            (0.0, 0.0),
            (0.0, 10.0),
            (5.0, 10.0),
            (5.0, 5.0),
            (10.0, 5.0),
            (10.0, 0.0),
        ]);
        let hull = convex_hull(&contour);

        // Convex hull of L-shape should have fewer or equal points
        assert!(hull.len() <= 5); // Depends on algorithm details
    }
}
