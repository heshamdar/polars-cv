//! Geometric measures for contours.
//!
//! Implements area, perimeter, centroid, bounding box, winding direction, and
//! point-to-contour distance. The maths is `geo`'s; this module only maps between
//! [`Contour`] and [`geo::Polygon`] and keeps the crate's degenerate-input
//! conventions.

use super::contour::{BoundingBox, Contour, Point, Winding};
use geo::{Area, Centroid, Closest, ClosestPoint, Distance, Euclidean, Length, LineString};

fn ring_polygon(points: &[Point]) -> geo::Polygon<f64> {
    Contour::new(points.to_vec()).to_geo()
}

fn ring_line_string(points: &[Point]) -> LineString<f64> {
    ring_polygon(points).into_inner().0
}

/// Computes the signed area of a single closed ring.
///
/// Positive area indicates counter-clockwise winding (in standard math coords).
/// Negative area indicates clockwise winding.
///
/// This measures the ring alone, which is why it takes a `&[Point]` rather than a
/// [`Contour`]: its callers ([`winding`] and [`super::transforms::ensure_winding`])
/// ask about point order, and [`area`] uses only its *sign*. For the area of the
/// region a contour describes — holes removed — use [`area`].
///
/// Measured on a hole-free `geo::Polygon` rather than the ring's `LineString`: to
/// `geo` a line string is one-dimensional and its `signed_area` is zero, so the
/// ring has to be handed over as the polygon it bounds.
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

    ring_polygon(points).signed_area()
}

/// Computes the area of a contour.
///
/// This is the area of the region the contour describes — the exterior minus the
/// union of every hole ring, whichever way each ring is wound. Subtracting each
/// hole's area in turn would double-count wherever two holes overlap, and would
/// disagree with `contains_point`, `rasterize` and [`super::pairwise::iou`]; see
/// [`Contour::to_geo_region`].
///
/// # Arguments
/// * `contour` - The contour to measure
/// * `signed` - If true, returns signed area (negative for CW exterior winding)
///
/// # Returns
/// Area value (absolute if `signed` is false)
pub fn area(contour: &Contour, signed: bool) -> f64 {
    let magnitude = contour.to_geo_region().unsigned_area();

    if signed && signed_area(&contour.exterior) < 0.0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Computes the perimeter (arc length) of a closed ring.
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

    Euclidean.length(&ring_line_string(points))
}

/// Computes the perimeter of a contour including holes.
///
/// # Arguments
/// * `contour` - The contour to measure
///
/// # Returns
/// Total perimeter length
pub fn perimeter(contour: &Contour) -> f64 {
    perimeter_of_ring(&contour.exterior)
        + contour
            .holes
            .iter()
            .map(|hole| perimeter_of_ring(hole))
            .sum::<f64>()
}

/// Mean of a set of points, used where a centroid is undefined.
fn vertex_mean(points: &[Point]) -> Point {
    if points.is_empty() {
        return Point::new(0.0, 0.0);
    }

    let n = points.len() as f64;
    Point::new(
        points.iter().map(|p| p.x).sum::<f64>() / n,
        points.iter().map(|p| p.y).sum::<f64>() / n,
    )
}

/// Computes the centroid (center of mass) of a polygon.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// Centroid point, or the mean of the points when fewer than 3 are given
pub fn centroid_of_ring(points: &[Point]) -> Point {
    if points.len() < 3 {
        return vertex_mean(points);
    }

    ring_polygon(points)
        .centroid()
        .map_or_else(|| vertex_mean(points), |c| Point::new(c.x(), c.y()))
}

/// Computes the centroid of a contour.
///
/// Measured on the same region as [`area`] — the exterior minus the union of the
/// hole rings. Handing `geo` a polygon whose holes are interior rings would instead
/// subtract each hole's moment in turn, double-counting wherever two hole rings
/// overlap and subtracting a nested ring that lies in an already-removed part; the
/// centroid would then belong to a different shape than the one `area`,
/// `contains_point` and [`super::pairwise::iou`] report on.
///
/// # Arguments
/// * `contour` - The contour to measure
///
/// # Returns
/// Centroid point
pub fn centroid(contour: &Contour) -> Point {
    contour.to_geo_region().centroid().map_or_else(
        || centroid_of_ring(&contour.exterior),
        |c| Point::new(c.x(), c.y()),
    )
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
/// This *reports* point order; it is never used to decide whether a ring is a
/// hole. See [`Contour`] for the hole convention.
///
/// # Arguments
/// * `points` - Slice of points forming a closed polygon
///
/// # Returns
/// Winding direction
pub fn winding(points: &[Point]) -> Winding {
    if signed_area(points) >= 0.0 {
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

// ============================================================================
// Point Distance Operations
// ============================================================================

fn geo_point(point: &Point) -> geo::Point<f64> {
    geo::Point::new(point.x, point.y)
}

/// Computes the minimum distance from a point to a polygon boundary.
///
/// # Arguments
/// * `point` - The query point
/// * `polygon` - Slice of points forming a closed polygon
///
/// # Returns
/// Minimum distance to any edge of the polygon
pub fn distance_to_polygon(point: &Point, polygon: &[Point]) -> f64 {
    match polygon {
        [] => f64::INFINITY,
        [only] => point.distance_to(only),
        _ => Euclidean.distance(&geo_point(point), &ring_line_string(polygon)),
    }
}

/// Computes the minimum distance from a point to a contour boundary.
///
/// Considers both the exterior ring and any holes.
///
/// # Arguments
/// * `point` - The query point
/// * `contour` - The contour to measure distance to
///
/// # Returns
/// Minimum distance to the contour boundary
pub fn distance_to_contour(point: &Point, contour: &Contour) -> f64 {
    std::iter::once(&contour.exterior)
        .chain(&contour.holes)
        .map(|ring| distance_to_polygon(point, ring))
        .fold(f64::INFINITY, f64::min)
}

/// Finds the nearest point on a polygon boundary.
///
/// # Arguments
/// * `point` - The query point
/// * `polygon` - Slice of points forming a closed polygon
///
/// # Returns
/// The nearest point on the polygon boundary, or None if polygon is empty
pub fn nearest_point_on_polygon(point: &Point, polygon: &[Point]) -> Option<Point> {
    match polygon {
        [] => None,
        [only] => Some(*only),
        _ => match ring_line_string(polygon).closest_point(&geo_point(point)) {
            Closest::Intersection(p) | Closest::SinglePoint(p) => Some(Point::new(p.x(), p.y())),
            Closest::Indeterminate => None,
        },
    }
}

/// Finds the nearest point on a contour boundary.
///
/// Considers both the exterior ring and any holes.
///
/// # Arguments
/// * `point` - The query point
/// * `contour` - The contour to search
///
/// # Returns
/// The nearest point on the contour boundary
pub fn nearest_point_on_contour(point: &Point, contour: &Contour) -> Option<Point> {
    std::iter::once(&contour.exterior)
        .chain(&contour.holes)
        .filter_map(|ring| nearest_point_on_polygon(point, ring))
        .min_by(|a, b| point.distance_to(a).total_cmp(&point.distance_to(b)))
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

    /// A 100x100 square with two *overlapping* hole rings.
    ///
    /// Hole A is [10,50]^2 (area 1600, centroid 30,30), hole B is [30,70]^2 (area
    /// 1600, centroid 50,50); they share [30,50]^2 (area 400, centroid 40,40). The
    /// union is therefore area 2800 about (40,40), leaving a region of area 7200.
    ///
    /// Asymmetry is the point: with a symmetric arrangement, subtracting the holes
    /// one at a time lands on the right centroid by accident.
    fn overlapping_holes() -> Contour {
        let ring = |pts: &[(f64, f64)]| -> Vec<Point> {
            pts.iter().map(|&(x, y)| Point::new(x, y)).collect()
        };
        Contour::with_holes(
            ring(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]),
            vec![
                ring(&[(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]),
                ring(&[(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)]),
            ],
        )
    }

    #[test]
    fn test_area_of_overlapping_holes_removes_the_union_once() {
        // Not 10000 - 1600 - 1600: the shared [30,50]^2 belongs to both rings and
        // is only removed once.
        assert!((area(&overlapping_holes(), false) - 7200.0).abs() < 1e-9);
    }

    #[test]
    fn test_centroid_measures_the_same_region_as_area() {
        // Region moment: 10000*50 - 2800*40 = 388000, over an area of 7200.
        // Subtracting each hole's moment in turn instead gives 372000/6800 =
        // 54.7059..., the centroid of a shape that does not exist.
        let c = centroid(&overlapping_holes());
        let expected = 388000.0 / 7200.0;
        assert!((c.x - expected).abs() < 1e-6, "x was {}", c.x);
        assert!((c.y - expected).abs() < 1e-6, "y was {}", c.y);
    }

    #[test]
    fn test_centroid_of_hole_free_contour_is_unchanged() {
        let c = centroid(&square_contour());
        assert!((c.x - 5.0).abs() < 1e-9 && (c.y - 5.0).abs() < 1e-9);
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

    #[test]
    fn test_distance_to_polygon() {
        let polygon = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];

        // Point outside, closest to right edge
        let point = Point::new(15.0, 5.0);
        assert!((distance_to_polygon(&point, &polygon) - 5.0).abs() < 1e-10);

        // Point inside (still computes distance to boundary)
        let point = Point::new(5.0, 5.0);
        assert!((distance_to_polygon(&point, &polygon) - 5.0).abs() < 1e-10);

        // Point on boundary
        let point = Point::new(10.0, 5.0);
        assert!(distance_to_polygon(&point, &polygon) < 1e-10);
    }

    #[test]
    fn test_nearest_point_on_polygon() {
        let polygon = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];

        // Point outside, closest to right edge
        let point = Point::new(15.0, 5.0);
        let nearest = nearest_point_on_polygon(&point, &polygon).unwrap();
        assert!((nearest.x - 10.0).abs() < 1e-10);
        assert!((nearest.y - 5.0).abs() < 1e-10);
    }
}
