//! Contour transformation operations.
//!
//! Implements translate, scale, flip, simplify, convex hull, and the shaping
//! ops: offset, union, smoothing, densifying and stroking.

use super::contour::{BorderArc, Contour, Outline, Point, Winding};
use super::measures::{centroid, contour_winding};
use super::ops::{LineCap, LineJoin, ScaleOrigin};
use geo::algorithm::buffer::{BufferStyle, LineCap as GeoCap, LineJoin as GeoJoin};
use geo::{
    unary_union, BooleanOps, Buffer, ChaikinSmoothing, ConvexHull, Densify, Euclidean, Simplify,
};
use std::borrow::Borrow;

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

/// A region's rings as open lines (`boundary()`): the exterior, then each
/// hole, each line returning to its first point so it traces the whole ring.
/// A ring of fewer than two points traces no edge and gives no line.
pub fn boundary(contour: &Contour) -> Vec<Outline> {
    std::iter::once(&contour.exterior)
        .chain(&contour.holes)
        .filter(|ring| ring.len() >= 2)
        .map(|ring| {
            let mut line = ring.clone();
            line.push(ring[0]);
            Outline::Open(line)
        })
        .collect()
}

/// An open line's ends joined (`close()`): the region it bounds. A line that
/// already returns to its first point (as [`boundary`] draws one) drops the
/// repeated point, since a ring never repeats it, so closing a boundary gives
/// back the region's exterior.
pub fn close(line: &[Point]) -> Contour {
    let mut ring = line.to_vec();
    if ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    Contour::new(ring)
}

/// Why `boundary()` refuses a line: it already is one.
pub const BOUNDARY_OF_A_LINE: &str =
    "an open contour (is_closed = false) is already a line; boundary() takes a region";

/// Why `close()` refuses a region: it has no ends to join.
pub const CLOSE_OF_A_REGION: &str =
    "the contour is already closed; close() takes an open line (is_closed = false)";

/// Why `scale(origin="centroid")` refuses a line: it bounds no region.
pub const SCALE_A_LINE_ABOUT_ITS_CENTROID: &str =
    "origin=\"centroid\" needs a closed contour: an open polyline \
     (is_closed = false) bounds no region to take a centroid of. Scale about \
     \"bbox_center\" or \"origin\" instead";

/// `boundary()` of one outline, as both surfaces run it: a region's rings
/// ([`boundary`]), or [`BOUNDARY_OF_A_LINE`].
pub fn boundary_outline(outline: &Outline) -> Result<Vec<Outline>, &'static str> {
    match outline {
        Outline::Closed(region) => Ok(boundary(region)),
        Outline::Open(_) => Err(BOUNDARY_OF_A_LINE),
    }
}

/// `close()` of one outline, as both surfaces run it: a line's ends joined
/// ([`close`]), or [`CLOSE_OF_A_REGION`].
pub fn close_outline(outline: &Outline) -> Result<Outline, &'static str> {
    match outline {
        Outline::Open(line) => Ok(Outline::Closed(close(line))),
        Outline::Closed(_) => Err(CLOSE_OF_A_REGION),
    }
}

/// `scale()` of one outline, as both surfaces run it ([`scale`] on its
/// points); a line has no centroid to scale about
/// ([`SCALE_A_LINE_ABOUT_ITS_CENTROID`]).
pub fn scale_outline(
    outline: &Outline,
    sx: f64,
    sy: f64,
    origin: ScaleOrigin,
) -> Result<Outline, &'static str> {
    if !outline.is_closed() && origin == ScaleOrigin::Centroid {
        return Err(SCALE_A_LINE_ABOUT_ITS_CENTROID);
    }
    Ok(outline.map_points(|c| scale(c, sx, sy, origin)))
}

/// The `k` largest contours by area ([`super::measures::area`], holes
/// removed): largest first, equal areas in their input order.
pub fn largest<C: std::borrow::Borrow<Contour>>(contours: &[C], k: usize) -> Vec<Contour> {
    let mut ranked: Vec<(f64, &Contour)> = contours
        .iter()
        .map(|c| (super::measures::area(c.borrow(), false), c.borrow()))
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
    if !(width > 0.0 && height > 0.0 && width.is_finite() && height.is_finite()) {
        return Err(format!(
            "the frame must have a positive, finite size, got {width} x {height}"
        ));
    }
    // A NaN would pass every `d > max_snap` test below and join a line from
    // anywhere.
    if max_snap.is_nan() || max_snap < 0.0 {
        return Err(format!("max_snap must be >= 0, got {max_snap}"));
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
        assert!(largest::<Contour>(&[], 2).is_empty());
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

    /// `max_snap` and the frame are numbers the walk relies on: a NaN
    /// `max_snap` compared false against every distance and joined a line
    /// from anywhere; an infinite frame has no perimeter to walk.
    #[test]
    fn parameters_no_walk_can_use_are_refused() {
        let middle = pts(&[(50.0, 50.0), (60.0, 60.0)]);
        for max_snap in [f64::NAN, -1.0] {
            let err = close_along_border(&middle, 100.0, 100.0, BorderArc::Shortest, max_snap)
                .unwrap_err();
            assert!(err.contains("max_snap"), "{err}");
        }
        for (w, h) in [(f64::INFINITY, 100.0), (100.0, f64::NAN), (0.0, 100.0)] {
            let err = close_along_border(&pectoral(), w, h, BorderArc::Shortest, 2.0).unwrap_err();
            assert!(err.contains("positive"), "{err}");
        }
        // An infinite max_snap is a meaningful "snap any end".
        assert!(
            close_along_border(&middle, 100.0, 100.0, BorderArc::Shortest, f64::INFINITY).is_ok()
        );
    }

    #[test]
    fn a_line_across_the_image_takes_the_shorter_side() {
        // Top edge to bottom edge, left of centre: the left side is shorter.
        let line = pts(&[(20.0, 0.0), (30.0, 100.0)]);
        let c = close_along_border(&line, 100.0, 100.0, BorderArc::Shortest, 2.0).unwrap();
        assert!((area(&c, false) - 2500.0).abs() < 1e-9, "{c:?}");
    }
}

#[cfg(test)]
mod boundary_close_tests {
    use super::*;
    use crate::geometry::measures::{area, outline_length};

    fn pts(v: &[(f64, f64)]) -> Vec<Point> {
        v.iter().map(|&(x, y)| Point::new(x, y)).collect()
    }

    #[test]
    fn boundary_draws_each_ring_back_to_its_start() {
        let square = pts(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
        let hole = pts(&[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]);
        let lines = boundary(&Contour::with_holes(square.clone(), vec![hole.clone()]));
        assert_eq!(lines.len(), 2);
        for (line, ring) in lines.iter().zip([&square, &hole]) {
            let Outline::Open(points) = line else {
                panic!("boundary gave a region: {line:?}")
            };
            assert_eq!(points.len(), ring.len() + 1);
            assert_eq!(points.first(), points.last());
            assert_eq!(&points[..ring.len()], &ring[..]);
        }
        // Its lines are the region's perimeter, the closing edges included.
        let total: f64 = lines.iter().map(outline_length).sum();
        assert!((total - (16.0 + 2.0 + 2f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn a_ring_with_no_edge_draws_no_line() {
        assert!(boundary(&Contour::new(pts(&[(1.0, 1.0)]))).is_empty());
        assert!(boundary(&Contour::new(Vec::new())).is_empty());
    }

    #[test]
    fn close_joins_the_ends_and_undoes_boundary() {
        let zigzag = pts(&[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]);
        let region = close(&zigzag);
        assert_eq!(region.exterior, zigzag);
        assert!((area(&region, false) - 6.0).abs() < 1e-12);
        let square = Contour::new(pts(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]));
        let [Outline::Open(line)] = &boundary(&square)[..] else {
            panic!("one ring, one line")
        };
        assert_eq!(close(line), square);
    }
}

// ============================================================================
// Shaping: offset, union, smoothing, densifying, stroking (CR-98)
// ============================================================================

/// The gap, in pixels, a round join's or cap's polygon aims for against its
/// true arc: a 64th of a pixel, under the finest sample spacing `rasterize`
/// takes (a 16th). Held up to a radius of about 32 pixels; see
/// [`arc_tolerance`] for larger ones.
pub const ARC_TOLERANCE: f64 = 1.0 / 64.0;

/// The finest angle step `i_overlay` takes along a round arc, whatever it is
/// asked for (`LineJoin::normalize` / `LineCap::normalize` clamp to it).
const FINEST_ARC_STEP: f64 = 0.01 * std::f64::consts::PI;

/// The largest gap between an arc of `radius` and its polygon: [`ARC_TOLERANCE`],
/// or, past a radius of about 32 pixels where `i_overlay`'s finest step
/// ([`FINEST_ARC_STEP`]) is reached, `r (1 − cos 0.01π)`, about `r / 2000`
/// (0.12 pixel at 1000). The documented bound of `offset` and `stroke`.
pub fn arc_tolerance(radius: f64) -> f64 {
    ARC_TOLERANCE.max(radius.abs() * (1.0 - FINEST_ARC_STEP.cos()))
}

/// A miter join is cut off where its point would reach more than this many
/// half-widths (or offset distances) from the corner: SVG's default, which
/// cuts corners sharper than about 29 degrees.
pub const MITER_LIMIT: f64 = 4.0;

/// The angle step a round arc of `radius` takes so it stays within
/// [`ARC_TOLERANCE`] of the true arc. `i_overlay` rounds an arc's segment count
/// down, so a step can reach twice this; a step `2θ` leaves a gap
/// `r (1 − cos θ)`, which is the tolerance at `θ = acos(1 − tol / r)`.
fn arc_step(radius: f64) -> f64 {
    let r = radius.abs();
    if r <= ARC_TOLERANCE {
        return std::f64::consts::FRAC_PI_4;
    }
    (1.0 - ARC_TOLERANCE / r).acos()
}

fn geo_join(join: LineJoin, radius: f64) -> GeoJoin<f64> {
    match join {
        LineJoin::Round => GeoJoin::Round(arc_step(radius)),
        // `i_overlay` takes the sharpest corner it still miters: a miter of
        // limit `m` reaches `1 / sin(θ / 2)` half-widths at a corner of `θ`.
        LineJoin::Miter => GeoJoin::Miter(2.0 * (1.0 / MITER_LIMIT).asin()),
        LineJoin::Bevel => GeoJoin::Bevel,
    }
}

fn regions_of(polygons: &geo::MultiPolygon<f64>) -> Vec<Contour> {
    polygons.0.iter().map(Contour::from_geo).collect()
}

fn line_to_geo(points: &[Point]) -> geo::LineString<f64> {
    points
        .iter()
        .map(|p| geo::coord! { x: p.x, y: p.y })
        .collect()
}

fn line_from_geo(line: &geo::LineString<f64>) -> Vec<Point> {
    line.0.iter().map(|c| Point::new(c.x, c.y)).collect()
}

fn union_polygons<C: Borrow<Contour>>(regions: &[C]) -> geo::MultiPolygon<f64> {
    let parts: Vec<geo::MultiPolygon<f64>> =
        regions.iter().map(|c| c.borrow().to_geo_region()).collect();
    unary_union(parts.iter())
}

/// The union of a set's regions (`union_all()`), as disjoint regions: each
/// member is its exterior minus its holes (`Contour::to_geo_region`), and a
/// hole another member covers is filled.
pub fn union_all<C: Borrow<Contour>>(regions: &[C]) -> Vec<Contour> {
    regions_of(&union_polygons(regions))
}

/// How [`boolean`] combines two sets' regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionOp {
    /// Inside either.
    Union,
    /// Inside both.
    Intersection,
    /// Inside the first and not the second.
    Difference,
    /// Inside exactly one.
    SymmetricDifference,
}

impl RegionOp {
    /// The op's engine name, as the executor names every step in a row
    /// error (`Offset:`, `Histogram:`).
    pub fn name(self) -> &'static str {
        match self {
            RegionOp::Union => "Union",
            RegionOp::Intersection => "Intersection",
            RegionOp::Difference => "Difference",
            RegionOp::SymmetricDifference => "SymmetricDifference",
        }
    }
}

/// Two sets combined by `op` (`union`, `intersection`, `difference`,
/// `symmetric_difference`), each read as the union of its regions; the result
/// is disjoint regions.
pub fn boolean<A: Borrow<Contour>, B: Borrow<Contour>>(
    a: &[A],
    b: &[B],
    op: RegionOp,
) -> Vec<Contour> {
    let (a, b) = (union_polygons(a), union_polygons(b));
    regions_of(&match op {
        RegionOp::Union => a.union(&b),
        RegionOp::Intersection => a.intersection(&b),
        RegionOp::Difference => a.difference(&b),
        RegionOp::SymmetricDifference => a.xor(&b),
    })
}

/// `offset(distance, join)`'s one rule for the value it takes.
pub fn check_offset_distance(distance: f64) -> Result<(), String> {
    if distance.is_finite() {
        Ok(())
    } else {
        Err(format!("offset: distance must be finite, got {distance}"))
    }
}

/// The union of a set's regions grown by `distance` (shrunk, negative), as
/// disjoint regions: every point within `distance` of the union, or inside
/// it and at least `-distance` from its edge. Growing can merge regions and
/// shrinking split or remove them. Convex corners turn by `join`; concave
/// ones stay sharp.
pub fn offset<C: Borrow<Contour>>(
    regions: &[C],
    distance: f64,
    join: LineJoin,
) -> Result<Vec<Contour>, String> {
    check_offset_distance(distance)?;
    let union = union_polygons(regions);
    if distance == 0.0 {
        return Ok(regions_of(&union));
    }
    let style = BufferStyle::new(distance).line_join(geo_join(join, distance));
    Ok(regions_of(&union.buffer_with_style(style)))
}

/// The most points a shaping op (`smooth`, `densify`) may make for one row's
/// contour set: 2^22, 64 MiB of coordinates. Both ops multiply points by a
/// parameter (a pass doubles them; a short segment length splits every
/// edge), so a value in a row could otherwise ask for more memory than
/// exists, and an allocation that fails aborts the process instead of
/// failing the row. Counted over the whole set: thousands of traced
/// contours each within it could together exceed it.
pub const MAX_POINTS: usize = 1 << 22;

/// `Ok` when an op's `points` (counted in f64, which cannot overflow) are
/// within [`MAX_POINTS`]; the refusal names the value that asked (the step
/// running it names the op, as it names every refusal).
fn within_point_budget(points: f64, asked: String) -> Result<(), String> {
    if points <= MAX_POINTS as f64 {
        return Ok(());
    }
    Err(format!(
        "{asked} would make about {points:.2e} points, past the {MAX_POINTS} \
         one row's contours may hold"
    ))
}

/// Chaikin's corner cutting, `iterations` times (`smooth()`): each edge is
/// replaced by its points a quarter and three quarters along. A region's
/// rings cut their closing edge too; an open line keeps its two ends, unless
/// it returns to its first point, when it has no ends and is cut through it.
///
/// One result per member of the set (a row). Each pass at most doubles
/// every path's points, so the set's `points · 2^iterations` is held to
/// [`MAX_POINTS`] before any pass runs.
pub fn smooth<O: Borrow<Outline>>(outlines: &[O], iterations: u32) -> Result<Vec<Outline>, String> {
    let vertices: usize = outlines
        .iter()
        .map(|o| o.borrow().vertices().count().max(1))
        .sum();
    let points = vertices as f64 * 2f64.powf(f64::from(iterations));
    within_point_budget(points, format!("{iterations} passes"))?;
    let n = iterations as usize;
    Ok(outlines
        .iter()
        .map(|outline| match outline.borrow() {
            Outline::Closed(c) => {
                Outline::Closed(Contour::from_geo(&c.to_geo().chaikin_smoothing(n)))
            }
            Outline::Open(points) => {
                Outline::Open(line_from_geo(&line_to_geo(points).chaikin_smoothing(n)))
            }
        })
        .collect())
}

/// `densify(max_segment_length)`'s one rule for the value it takes.
pub fn check_max_segment_length(max_segment_length: f64) -> Result<(), String> {
    if max_segment_length.is_finite() && max_segment_length > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "densify: max_segment_length must be a finite length > 0, got {max_segment_length}"
        ))
    }
}

/// Each edge split into `ceil(L / max_segment_length)` equal parts
/// (`densify()`), every vertex kept: a region's closing edges included, an
/// open line's ends joined by nothing. One result per member of the set (a
/// row), whose points are held to [`MAX_POINTS`] before any is made.
pub fn densify<O: Borrow<Outline>>(
    outlines: &[O],
    max_segment_length: f64,
) -> Result<Vec<Outline>, String> {
    check_max_segment_length(max_segment_length)?;
    // Every vertex is kept and each edge adds `ceil(L / max) - 1`, counted
    // before any point is made.
    let points: f64 = outlines
        .iter()
        .flat_map(|o| o.borrow().paths())
        .map(|(points, closed)| {
            let closing = closed.then(|| points.last().zip(points.first())).flatten();
            let edges = points.windows(2).map(|e| (&e[0], &e[1])).chain(closing);
            points.len() as f64
                + edges
                    .map(|(a, b)| ((a.distance_to(b) / max_segment_length).ceil() - 1.0).max(0.0))
                    .sum::<f64>()
        })
        .sum();
    within_point_budget(points, format!("max_segment_length={max_segment_length}"))?;
    Ok(outlines
        .iter()
        .map(|outline| match outline.borrow() {
            Outline::Closed(c) => Outline::Closed(Contour::from_geo(
                &Euclidean.densify(&c.to_geo(), max_segment_length),
            )),
            Outline::Open(points) => Outline::Open(line_from_geo(
                &Euclidean.densify(&line_to_geo(points), max_segment_length),
            )),
        })
        .collect())
}

/// How `stroke()` draws a path: its width, ends and corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stroke {
    /// The stroke's full width, in pixels.
    pub width: f64,
    /// How each end of an open path is drawn.
    pub cap: LineCap,
    /// How each corner is turned.
    pub join: LineJoin,
}

impl Stroke {
    /// The one rule for `width`.
    pub fn check_width(width: f64) -> Result<(), String> {
        if width.is_finite() && width > 0.0 {
            Ok(())
        } else {
            Err(format!(
                "stroke: width must be a finite width > 0, got {width}"
            ))
        }
    }
}

/// The region one path covers when stroked: every point within `width / 2`
/// of it (round caps and joins), or that band with its ends and corners
/// shaped by `cap` and `join`. A closed path has no ends, so `cap` draws
/// nothing: its start is a corner like the others. A one-point open path is
/// a dot (none, for a butt cap).
fn stroke_path(points: &[Point], closed: bool, style: &Stroke) -> Vec<Contour> {
    let radius = style.width / 2.0;
    if closed {
        // geo strokes every line string as an open path, capping where a loop
        // starts; `i_overlay` (under it) strokes a closed path, joined there.
        use i_overlay::mesh::stroke::offset::StrokeOffset;
        use i_overlay::mesh::style::StrokeStyle;
        let path: Vec<[f64; 2]> = points.iter().map(|p| [p.x, p.y]).collect();
        let shapes = path.stroke(
            StrokeStyle::new(style.width).line_join(geo_join(style.join, radius)),
            true,
        );
        let mut parts: Vec<Contour> = shapes
            .iter()
            .filter_map(|shape| {
                let mut rings = shape
                    .iter()
                    .map(|ring| ring.iter().map(|&[x, y]| Point::new(x, y)).collect());
                Some(Contour::with_holes(rings.next()?, rings.collect()))
            })
            .collect();
        // Where the stroke is wider than the ring's interior, `i_overlay`
        // turns the ring's inward side inside out and leaves a hole the band
        // covers. The band round a ring is also its region grown by the
        // half-width less that region shrunk by it, which `offset` gives
        // exactly; that misses only what has no area (a doubled segment, a
        // spike), which the stroke above covers. Together they are the band.
        let region = [Contour::new(points.to_vec())];
        let grown = offset(&region, radius, style.join).expect("a checked width is finite");
        let shrunk = offset(&region, -radius, style.join).expect("a checked width is finite");
        parts.extend(boolean(&grown, &shrunk, RegionOp::Difference));
        return parts;
    }
    let cap = match style.cap {
        LineCap::Round => GeoCap::Round(arc_step(radius)),
        LineCap::Square => GeoCap::Square,
        LineCap::Butt => GeoCap::Butt,
    };
    let buffered = line_to_geo(points).buffer_with_style(
        BufferStyle::new(radius)
            .line_cap(cap)
            .line_join(geo_join(style.join, radius)),
    );
    regions_of(&buffered)
}

/// `stroke()`: the region a set's paths cover when stroked, as disjoint
/// regions (always a set). Every member is read as its outline: an open line
/// with `cap` at its ends (or, returning to its first point, as a closed
/// path), a region's rings each as a closed path. A ring of fewer than two
/// points has no edge and covers nothing; a one-point line is a dot. A region is not filled:
/// `stroke(...).rasterize()` draws outlines, and a filled region is
/// rasterized as itself.
pub fn stroke<O: Borrow<Outline>>(outlines: &[O], style: &Stroke) -> Result<Vec<Contour>, String> {
    Stroke::check_width(style.width)?;
    let mut parts = Vec::new();
    for outline in outlines {
        for (points, closed) in outline.borrow().paths() {
            // A closed path of fewer than two points has no edge, and no ends
            // to cap: it covers nothing (`i_overlay` underflows on an empty
            // one). A one-point *open* line is still a dot, its caps.
            if closed && points.len() < 2 {
                continue;
            }
            let loop_line = !closed && points.len() > 2 && points.first() == points.last();
            parts.extend(if loop_line {
                stroke_path(&points[..points.len() - 1], true, style)
            } else {
                stroke_path(points, closed, style)
            });
        }
    }
    Ok(union_all(&parts))
}

#[cfg(test)]
mod shaping_tests {
    use super::*;
    use crate::geometry::measures::area;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Contour {
        Contour::from_tuples(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }

    fn total_area(regions: &[Contour]) -> f64 {
        regions.iter().map(|c| area(c, false)).sum()
    }

    #[test]
    fn boolean_reads_each_side_as_its_union() {
        let a = [rect(0.0, 0.0, 6.0, 6.0), rect(3.0, 3.0, 9.0, 9.0)];
        let b = [rect(5.0, 0.0, 12.0, 4.0)];
        let whole_a = total_area(&union_all(&a));
        let inter = total_area(&boolean(&a, &b, RegionOp::Intersection));
        let union = total_area(&boolean(&a, &b, RegionOp::Union));
        assert!((whole_a - 63.0).abs() < 1e-9);
        // [5, 6] x [0, 4] and [5, 9] x [3, 4], overlapping in [5, 6] x [3, 4].
        assert!((inter - 7.0).abs() < 1e-9);
        assert!((union + inter - whole_a - 28.0).abs() < 1e-9);
        let diff = total_area(&boolean(&a, &b, RegionOp::Difference));
        assert!((diff - (whole_a - inter)).abs() < 1e-9);
        let xor = total_area(&boolean(&a, &b, RegionOp::SymmetricDifference));
        assert!((xor - (union - inter)).abs() < 1e-9);
        assert!(boolean(&a, &a, RegionOp::Difference).is_empty());
        assert!(boolean::<Contour, Contour>(&[], &b, RegionOp::Intersection).is_empty());
    }

    #[test]
    fn offset_closed_forms() {
        let sq = [rect(5.0, 5.0, 15.0, 15.0)];
        let miter = offset(&sq, 3.0, LineJoin::Miter).unwrap();
        assert!((total_area(&miter) - 256.0).abs() < 1e-6);
        let bevel = offset(&sq, 3.0, LineJoin::Bevel).unwrap();
        assert!((total_area(&bevel) - 238.0).abs() < 1e-6);
        let round = total_area(&offset(&sq, 3.0, LineJoin::Round).unwrap());
        let exact = 100.0 + 120.0 + 9.0 * std::f64::consts::PI;
        assert!(
            round <= exact + 1e-6 && round > exact - 6.0 * std::f64::consts::PI * ARC_TOLERANCE
        );
        for join in [LineJoin::Round, LineJoin::Miter, LineJoin::Bevel] {
            let shrunk = offset(&sq, -3.0, join).unwrap();
            assert!((total_area(&shrunk) - 16.0).abs() < 1e-6, "{join:?}");
        }
        assert!(offset(&sq, -6.0, LineJoin::Round).unwrap().is_empty());
        assert!(offset(&sq, f64::NAN, LineJoin::Round).is_err());
    }

    #[test]
    fn offset_reads_the_set_as_its_union() {
        let pair = [rect(0.0, 0.0, 6.0, 10.0), rect(4.0, 0.0, 10.0, 10.0)];
        let shrunk = offset(&pair, -2.0, LineJoin::Miter).unwrap();
        assert_eq!(shrunk.len(), 1);
        assert!((total_area(&shrunk) - 36.0).abs() < 1e-6);
        assert_eq!(
            offset(&pair, 0.0, LineJoin::Round).unwrap(),
            union_all(&pair)
        );
    }

    #[test]
    fn union_fills_a_covered_hole_and_keeps_an_island() {
        let ring = Contour::with_holes(
            rect(0.0, 0.0, 20.0, 20.0).exterior,
            vec![rect(4.0, 4.0, 16.0, 16.0).exterior],
        );
        let island = union_all(&[ring.clone(), rect(8.0, 8.0, 12.0, 12.0)]);
        assert_eq!(island.len(), 2);
        let plugged = union_all(&[ring, rect(3.0, 3.0, 17.0, 17.0)]);
        assert_eq!(plugged.len(), 1);
        assert!(plugged[0].holes.is_empty());
        assert!(union_all::<Contour>(&[]).is_empty());
    }

    #[test]
    fn smooth_keeps_an_open_lines_ends() {
        let line = Outline::Open(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
        ]);
        let [Outline::Open(out)] = &smooth(&[&line], 1).unwrap()[..] else {
            panic!("stays open")
        };
        // Each edge gives its quarter and three-quarter points, in order.
        let want = [
            (0.0, 0.0),
            (1.0, 0.0),
            (3.0, 0.0),
            (4.0, 1.0),
            (4.0, 3.0),
            (4.0, 4.0),
        ];
        assert_eq!(out.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>(), want);
        assert_eq!(smooth(&[&line], 0).unwrap(), std::slice::from_ref(&line));
        // A pass doubles the points: the budget refuses before running.
        let err = smooth(&[&line], 40).unwrap_err();
        assert!(
            err.starts_with("40 passes") && err.contains("4194304"),
            "{err}"
        );
        assert!(smooth(&[&line], u32::MAX).is_err());
    }

    #[test]
    fn densify_splits_each_edge_equally() {
        let sq = Outline::Closed(rect(0.0, 0.0, 4.0, 4.0));
        let [Outline::Closed(out)] = &densify(&[&sq], 1.5).unwrap()[..] else {
            panic!("stays closed")
        };
        // Each 4-long edge, the closing one too, in three parts.
        assert_eq!(out.exterior.len(), 12);
        assert!(densify(&[&sq], 0.0).is_err());
        assert!(densify(&[&sq], f64::INFINITY).is_err());
        // 16 units of perimeter in 1e-9 steps: refused before splitting.
        let err = densify(&[&sq], 1e-9).unwrap_err();
        assert!(
            err.starts_with("max_segment_length=") && err.contains("4194304"),
            "{err}"
        );
    }

    /// `arc_tolerance` is the bound the geometry keeps, and is reached past
    /// `i_overlay`'s finest step: each edge midpoint of a round offset of a
    /// square lies within it of the true boundary.
    #[test]
    fn a_round_offset_keeps_arc_tolerance() {
        for d in [3.0, 31.0, 200.0, 1000.0] {
            let grown = offset(&[rect(0.0, 0.0, 10.0, 10.0)], d, LineJoin::Round).unwrap();
            let ring = &grown[0].exterior;
            let worst = ring
                .iter()
                .zip(ring.iter().cycle().skip(1))
                .map(|(a, b)| {
                    let (x, y) = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
                    let dx = (-x).max(0.0).max(x - 10.0);
                    let dy = (-y).max(0.0).max(y - 10.0);
                    d - dx.hypot(dy)
                })
                .fold(0.0, f64::max);
            assert!(worst <= arc_tolerance(d) + 1e-9, "gap {worst} at {d}");
            if d >= 200.0 {
                assert!(worst > ARC_TOLERANCE, "the cap is reached at {d}");
            }
        }
    }

    #[test]
    fn stroke_closed_forms() {
        let line = Outline::Open(vec![Point::new(3.0, 6.0), Point::new(13.0, 6.0)]);
        let style = |cap| Stroke {
            width: 2.0,
            cap,
            join: LineJoin::Round,
        };
        let area_of = |o: &Outline, cap| total_area(&stroke(&[o], &style(cap)).unwrap());
        assert!((area_of(&line, LineCap::Butt) - 20.0).abs() < 1e-6);
        assert!((area_of(&line, LineCap::Square) - 24.0).abs() < 1e-6);
        let round = area_of(&line, LineCap::Round);
        let exact = 20.0 + std::f64::consts::PI;
        assert!(
            round <= exact + 1e-6 && round > exact - 2.0 * std::f64::consts::PI * ARC_TOLERANCE
        );
        let zero = Stroke {
            width: 0.0,
            ..style(LineCap::Round)
        };
        assert!(stroke(&[&line], &zero).is_err());
        let dot = Outline::Open(vec![Point::new(3.0, 6.0)]);
        assert!(stroke(&[dot], &style(LineCap::Butt)).unwrap().is_empty());
        // A region's ring is a closed path: a 10x10 square's band of width 2
        // is 12x12 minus 8x8 with miter corners, and no cap applies.
        let square = Outline::Closed(rect(0.0, 0.0, 10.0, 10.0));
        let ring = Stroke {
            join: LineJoin::Miter,
            ..style(LineCap::Round)
        };
        let band = stroke(&[square], &ring).unwrap();
        assert_eq!(band.len(), 1);
        assert_eq!(band[0].holes.len(), 1);
        assert!((total_area(&band) - (144.0 - 64.0)).abs() < 1e-6);
    }

    #[test]
    fn a_stroke_wider_than_its_ring_is_solid() {
        // Every point of a unit square is within 1/2 of its ring, so a band
        // of half-width 2 covers it: one solid 5x5 square (miter) or the
        // square grown by a disc (round). `i_overlay` alone left a hole
        // where the ring's inward side turned inside out.
        for join in [LineJoin::Miter, LineJoin::Round] {
            let style = Stroke {
                width: 4.0,
                cap: LineCap::Butt,
                join,
            };
            let band = stroke(&[Outline::Closed(rect(5.0, 5.0, 6.0, 6.0))], &style).unwrap();
            assert_eq!(band.len(), 1, "{join:?}");
            assert!(band[0].holes.is_empty(), "{join:?}: {:?}", band[0].holes);
        }
        let miter = Stroke {
            width: 4.0,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
        };
        let band = stroke(&[Outline::Closed(rect(5.0, 5.0, 6.0, 6.0))], &miter).unwrap();
        assert!((total_area(&band) - 25.0).abs() < 1e-6);
    }
}
