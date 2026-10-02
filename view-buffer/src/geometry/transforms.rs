//! Contour transformation operations.
//!
//! Implements translate, scale, flip, simplify, and convex hull.

use super::contour::{BorderArc, Contour, Outline, Point, Winding};
use super::measures::{centroid, contour_winding};
use super::ops::ScaleOrigin;
use geo::{ConvexHull, Simplify};

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
    if contour_winding(contour) == direction {
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
/// Each ring is simplified independently; holes are preserved.
///
/// # Arguments
/// * `contour` - The contour to simplify
/// * `tolerance` - Maximum perpendicular distance for point removal
///
/// # Returns
/// Simplified contour with fewer points
pub fn simplify(contour: &Contour, tolerance: f64) -> Contour {
    Contour::from_geo(&contour.to_geo().simplify(tolerance))
}

/// Simplifies an outline: a region as [`simplify`] does, an open polyline as
/// a line (Douglas-Peucker keeps its two endpoints; no closing edge is added).
pub fn simplify_outline(outline: &Outline, tolerance: f64) -> Outline {
    match outline {
        Outline::Closed(c) => Outline::Closed(simplify(c, tolerance)),
        Outline::Open(points) => {
            let line: geo::LineString<f64> = points
                .iter()
                .map(|p| geo::coord! { x: p.x, y: p.y })
                .collect();
            Outline::Open(
                line.simplify(tolerance)
                    .0
                    .iter()
                    .map(|c| Point::new(c.x, c.y))
                    .collect(),
            )
        }
    }
}

/// The convex hull of an outline's exterior vertices — the same region
/// whether the outline closes or not, since a hull reads only the points.
pub fn convex_hull_outline(outline: &Outline) -> Contour {
    match outline {
        Outline::Closed(c) => convex_hull(c),
        Outline::Open(points) => convex_hull(&Contour::new(points.clone())),
    }
}

/// The `k` largest contours by area ([`super::measures::area`], holes
/// removed): largest first, equal areas in their input order.
pub fn largest(contours: &[Contour], k: usize) -> Vec<Contour> {
    let mut ranked: Vec<(f64, &Contour)> = contours
        .iter()
        .map(|c| (super::measures::area(c, false), c))
        .collect();
    // Stable, so equal areas keep their input order.
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.into_iter().take(k).map(|(_, c)| c.clone()).collect()
}

/// Close an open line whose ends lie on the image frame `[0, width] x
/// [0, height]` into a region, along the frame.
///
/// Each endpoint is snapped onto its nearest frame edge (refused when farther
/// than `max_snap`: it does not lie on the frame, and joining it there would
/// invent geometry). The frame is then walked from the line's end back to its
/// start — `arc` says which way round — collecting the corners it passes, so
/// the ring is the line, the snapped end, those corners and the snapped start.
/// A chord between the endpoints is not enough: for a corner region the
/// endpoints sit on two edges and the region is the corner the chord cuts off.
pub fn close_along_border(
    line: &[Point],
    width: f64,
    height: f64,
    arc: BorderArc,
    max_snap: f64,
) -> Result<Contour, String> {
    if !(width > 0.0 && height > 0.0) {
        return Err(format!(
            "the frame must have a positive size, got {width} x {height}"
        ));
    }
    let (Some(&first), Some(&last)) = (line.first(), line.last()) else {
        return Err("an empty line has no ends to close".to_string());
    };
    if line.len() < 2 {
        return Err("a one-point line has no ends to close".to_string());
    }
    let (w, h) = (width, height);
    let perimeter = 2.0 * (w + h);
    // A point on the frame and its clockwise perimeter coordinate from (0, 0)
    // (y down: along the top rightward, down the right, ...).
    let snap = |p: Point, which: &str| -> Result<(Point, f64), String> {
        let (x, y) = (p.x.clamp(0.0, w), p.y.clamp(0.0, h));
        let candidates = [
            (Point::new(x, 0.0), x),                     // top
            (Point::new(w, y), w + y),                   // right
            (Point::new(x, h), w + h + (w - x)),         // bottom
            (Point::new(0.0, y), 2.0 * w + h + (h - y)), // left
        ];
        let (q, t) = candidates
            .into_iter()
            .min_by(|a, b| p.distance_to(&a.0).total_cmp(&p.distance_to(&b.0)))
            .expect("four candidates");
        let d = p.distance_to(&q);
        if d > max_snap {
            return Err(format!(
                "the line's {which} point ({}, {}) is {d} from the image frame, \
                 beyond max_snap = {max_snap}: it does not lie on the frame",
                p.x, p.y
            ));
        }
        Ok((q, t.rem_euclid(perimeter)))
    };
    let (start, t_start) = snap(first, "first")?;
    let (end, t_end) = snap(last, "last")?;
    let clockwise_span = (t_start - t_end).rem_euclid(perimeter);
    let clockwise = match arc {
        BorderArc::Clockwise => true,
        BorderArc::Counterclockwise => false,
        BorderArc::Shortest => clockwise_span <= perimeter - clockwise_span,
    };
    let span = if clockwise {
        clockwise_span
    } else {
        perimeter - clockwise_span
    };
    let corners = [
        (Point::new(0.0, 0.0), 0.0),
        (Point::new(w, 0.0), w),
        (Point::new(w, h), w + h),
        (Point::new(0.0, h), 2.0 * w + h),
    ];
    let mut passed: Vec<(f64, Point)> = corners
        .into_iter()
        .map(|(c, t)| {
            let along = if clockwise { t - t_end } else { t_end - t };
            (along.rem_euclid(perimeter), c)
        })
        .filter(|&(along, _)| along > 0.0 && along < span)
        .collect();
    passed.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut ring: Vec<Point> = line.to_vec();
    let mut push = |p: Point| {
        if ring.last() != Some(&p) {
            ring.push(p);
        }
    };
    push(end);
    passed.into_iter().for_each(|(_, c)| push(c));
    push(start);
    // The ring closes back on its first point implicitly; drop a repeat.
    if ring.len() > 1 && ring.last() == ring.first() {
        ring.pop();
    }
    Ok(Contour::new(ring))
}

/// Computes the convex hull of a contour's exterior ring.
///
/// # Arguments
/// * `contour` - The contour to compute hull for
///
/// # Returns
/// New contour representing the convex hull
pub fn convex_hull(contour: &Contour) -> Contour {
    Contour::from_geo(&contour.to_geo().convex_hull())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_keeps_the_k_biggest_largest_first_ties_in_order() {
        let sq =
            |x: f64, s: f64| Contour::from_tuples(&[(x, 0.0), (x + s, 0.0), (x + s, s), (x, s)]);
        let set = vec![sq(0.0, 1.0), sq(10.0, 3.0), sq(20.0, 2.0), sq(30.0, 3.0)];
        let xs = |cs: Vec<Contour>| cs.iter().map(|c| c.exterior[0].x).collect::<Vec<_>>();
        assert_eq!(xs(largest(&set, 3)), vec![10.0, 30.0, 20.0]);
        assert_eq!(xs(largest(&set, 10)).len(), 4);
        assert!(largest(&[], 2).is_empty());
    }

    #[test]
    fn an_open_outline_simplifies_as_a_line() {
        // A zig-zag whose middle vertices sit within the tolerance of the
        // chord: a line keeps its endpoints and no closing edge appears.
        let line = Outline::Open(vec![
            Point::new(0.0, 0.0),
            Point::new(5.0, 0.1),
            Point::new(10.0, 0.0),
        ]);
        assert_eq!(
            simplify_outline(&line, 1.0),
            Outline::Open(vec![Point::new(0.0, 0.0), Point::new(10.0, 0.0)])
        );
    }

    #[test]
    fn a_point_wise_transform_keeps_an_outline_open() {
        let line = Outline::Open(vec![Point::new(0.0, 0.0), Point::new(1.0, 2.0)]);
        assert_eq!(
            line.map_points(|c| translate(c, 1.0, 1.0)),
            Outline::Open(vec![Point::new(1.0, 1.0), Point::new(2.0, 3.0)])
        );
        assert_eq!(
            line.map_points(flip),
            Outline::Open(vec![Point::new(1.0, 2.0), Point::new(0.0, 0.0)])
        );
    }

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
        let flipped = flip(&contour);

        assert_ne!(contour_winding(&contour), contour_winding(&flipped));
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

#[cfg(test)]
mod close_along_border_tests {
    use super::*;
    use crate::geometry::contour::BorderArc;
    use crate::geometry::measures::area;

    fn pts(v: &[(f64, f64)]) -> Vec<Point> {
        v.iter().map(|&(x, y)| Point::new(x, y)).collect()
    }

    /// A pectoral-like edge from the top edge to the left edge of a 100x100
    /// image: its region is the corner triangle at (0, 0).
    fn pectoral() -> Vec<Point> {
        pts(&[(60.0, 0.0), (30.0, 20.0), (0.0, 40.0)])
    }

    #[test]
    fn the_shortest_arc_closes_through_the_corner_it_passes() {
        let c = close_along_border(&pectoral(), 100.0, 100.0, BorderArc::Shortest, 2.0).unwrap();
        assert_eq!(c.exterior.last(), Some(&Point::new(0.0, 0.0)));
        // (60,0) (30,20) (0,40) (0,0): shoelace area 1200.
        assert!((area(&c, false) - 1200.0).abs() < 1e-9, "{c:?}");
    }

    #[test]
    fn the_other_way_round_is_the_complement() {
        // From the end (left edge) back to the start (top edge) clockwise is
        // the short way here; counter-clockwise runs down, across the bottom
        // and up the right edge, through the other three corners.
        let cw = close_along_border(&pectoral(), 100.0, 100.0, BorderArc::Clockwise, 2.0).unwrap();
        let ccw = close_along_border(&pectoral(), 100.0, 100.0, BorderArc::Counterclockwise, 2.0)
            .unwrap();
        assert!((area(&cw, false) - 1200.0).abs() < 1e-9);
        assert!(
            (area(&ccw, false) - (10000.0 - 1200.0)).abs() < 1e-9,
            "{ccw:?}"
        );
        assert_eq!(ccw.exterior.len(), 3 + 3);
    }

    #[test]
    fn endpoints_near_the_frame_are_snapped_onto_it() {
        let line = pts(&[(60.0, 1.5), (30.0, 20.0), (1.0, 40.0)]);
        let c = close_along_border(&line, 100.0, 100.0, BorderArc::Shortest, 2.0).unwrap();
        // The snapped end (0, 40), the corner, the snapped start (60, 0).
        assert_eq!(
            &c.exterior[3..],
            &pts(&[(0.0, 40.0), (0.0, 0.0), (60.0, 0.0)])[..]
        );
    }

    #[test]
    fn an_endpoint_away_from_the_frame_is_refused() {
        let line = pts(&[(60.0, 5.0), (0.0, 40.0)]);
        let err = close_along_border(&line, 100.0, 100.0, BorderArc::Shortest, 2.0).unwrap_err();
        assert!(err.contains("5"), "{err}");
    }

    #[test]
    fn a_line_across_the_image_takes_the_shorter_side() {
        // Top edge to bottom edge, left of centre: the left side is shorter.
        let line = pts(&[(20.0, 0.0), (30.0, 100.0)]);
        let c = close_along_border(&line, 100.0, 100.0, BorderArc::Shortest, 2.0).unwrap();
        assert!((area(&c, false) - 2500.0).abs() < 1e-9, "{c:?}");
    }
}
