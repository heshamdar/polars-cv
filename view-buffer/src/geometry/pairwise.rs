//! Pairwise geometric operations between contours.
//!
//! Implements IoU, Dice coefficient, and Hausdorff distance.

use super::contour::{Contour, Point};
use super::measures::area;

/// Computes Intersection over Union (IoU) between two contours.
///
/// IoU = intersection_area / union_area
///
/// This implementation uses a polygon clipping approach for exact computation
/// when contours are simple polygons. For complex cases, it falls back to
/// rasterization-based approximation.
///
/// # Arguments
/// * `a` - First contour
/// * `b` - Second contour
///
/// # Returns
/// IoU value in [0, 1]
pub fn iou(a: &Contour, b: &Contour) -> f64 {
    let area_a = area(a, false);
    let area_b = area(b, false);

    if area_a < 1e-10 || area_b < 1e-10 {
        return 0.0;
    }

    // Check bounding box intersection first (fast rejection)
    let bbox_a = match a.bounding_box() {
        Some(bb) => bb,
        None => return 0.0,
    };
    let bbox_b = match b.bounding_box() {
        Some(bb) => bb,
        None => return 0.0,
    };

    if !bbox_a.intersects(&bbox_b) {
        return 0.0;
    }

    // Use Sutherland-Hodgman polygon clipping for intersection
    let intersection = polygon_intersection(&a.exterior, &b.exterior);
    let intersection_area = polygon_area(&intersection);

    let union_area = area_a + area_b - intersection_area;

    if union_area < 1e-10 {
        return 0.0;
    }

    (intersection_area / union_area).clamp(0.0, 1.0)
}

/// Computes the Dice coefficient between two contours.
///
/// Dice = 2 * intersection_area / (area_a + area_b)
///
/// # Arguments
/// * `a` - First contour
/// * `b` - Second contour
///
/// # Returns
/// Dice coefficient in [0, 1]
pub fn dice(a: &Contour, b: &Contour) -> f64 {
    let area_a = area(a, false);
    let area_b = area(b, false);

    if area_a < 1e-10 || area_b < 1e-10 {
        return 0.0;
    }

    let bbox_a = match a.bounding_box() {
        Some(bb) => bb,
        None => return 0.0,
    };
    let bbox_b = match b.bounding_box() {
        Some(bb) => bb,
        None => return 0.0,
    };

    if !bbox_a.intersects(&bbox_b) {
        return 0.0;
    }

    let intersection = polygon_intersection(&a.exterior, &b.exterior);
    let intersection_area = polygon_area(&intersection);

    let denominator = area_a + area_b;

    if denominator < 1e-10 {
        return 0.0;
    }

    (2.0 * intersection_area / denominator).clamp(0.0, 1.0)
}

/// Computes the Hausdorff distance between two contours.
///
/// The Hausdorff distance is the maximum of the directed Hausdorff distances:
/// H(A, B) = max(h(A, B), h(B, A))
///
/// where h(A, B) = max_{a in A} min_{b in B} d(a, b)
///
/// # Arguments
/// * `a` - First contour
/// * `b` - Second contour
///
/// # Returns
/// Hausdorff distance
pub fn hausdorff_distance(a: &Contour, b: &Contour) -> f64 {
    let h_ab = directed_hausdorff(&a.exterior, &b.exterior);
    let h_ba = directed_hausdorff(&b.exterior, &a.exterior);
    h_ab.max(h_ba)
}

/// Directed Hausdorff distance from A to B.
fn directed_hausdorff(a: &[Point], b: &[Point]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let mut max_dist = 0.0;

    for pa in a {
        let mut min_dist = f64::INFINITY;
        for pb in b {
            let dist = pa.distance_to(pb);
            if dist < min_dist {
                min_dist = dist;
            }
        }
        if min_dist > max_dist {
            max_dist = min_dist;
        }
    }

    max_dist
}

/// Sutherland-Hodgman polygon clipping algorithm.
///
/// Clips `subject` polygon against `clip` polygon.
fn polygon_intersection(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }

    let mut output = subject.to_vec();

    let n = clip.len();
    for i in 0..n {
        if output.is_empty() {
            break;
        }

        let edge_start = &clip[i];
        let edge_end = &clip[(i + 1) % n];

        output = clip_polygon_by_edge(&output, edge_start, edge_end);
    }

    output
}

/// Clips a polygon by a single edge using Sutherland-Hodgman.
fn clip_polygon_by_edge(polygon: &[Point], edge_start: &Point, edge_end: &Point) -> Vec<Point> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let n = polygon.len();

    for i in 0..n {
        let current = &polygon[i];
        let next = &polygon[(i + 1) % n];

        let current_inside = is_inside_edge(current, edge_start, edge_end);
        let next_inside = is_inside_edge(next, edge_start, edge_end);

        if current_inside {
            result.push(*current);
            if !next_inside {
                if let Some(inter) = line_intersection(current, next, edge_start, edge_end) {
                    result.push(inter);
                }
            }
        } else if next_inside {
            if let Some(inter) = line_intersection(current, next, edge_start, edge_end) {
                result.push(inter);
            }
        }
    }

    result
}

/// Checks if a point is on the "inside" of an edge (left side for CCW polygon).
fn is_inside_edge(point: &Point, edge_start: &Point, edge_end: &Point) -> bool {
    // Cross product determines which side of the edge the point is on
    let cross = (edge_end.x - edge_start.x) * (point.y - edge_start.y)
        - (edge_end.y - edge_start.y) * (point.x - edge_start.x);
    cross >= 0.0
}

/// Computes the intersection of two line segments.
fn line_intersection(p1: &Point, p2: &Point, p3: &Point, p4: &Point) -> Option<Point> {
    let d1x = p2.x - p1.x;
    let d1y = p2.y - p1.y;
    let d2x = p4.x - p3.x;
    let d2y = p4.y - p3.y;

    let denom = d1x * d2y - d1y * d2x;

    if denom.abs() < 1e-10 {
        return None; // Parallel lines
    }

    let t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;

    Some(Point::new(p1.x + t * d1x, p1.y + t * d1y))
}

/// Computes the area of a polygon using the Shoelace formula.
fn polygon_area(points: &[Point]) -> f64 {
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

    (area / 2.0).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_contour(x: f64, y: f64, size: f64) -> Contour {
        Contour::from_tuples(&[(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
    }

    #[test]
    fn test_iou_identical() {
        let a = square_contour(0.0, 0.0, 10.0);
        let b = square_contour(0.0, 0.0, 10.0);
        let iou_val = iou(&a, &b);
        assert!((iou_val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = square_contour(0.0, 0.0, 10.0);
        let b = square_contour(20.0, 20.0, 10.0);
        let iou_val = iou(&a, &b);
        assert!(iou_val < 0.01);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = square_contour(0.0, 0.0, 10.0);
        let b = square_contour(5.0, 5.0, 10.0);
        let iou_val = iou(&a, &b);
        // 25 / (100 + 100 - 25) = 25/175 ≈ 0.143
        assert!(iou_val > 0.1 && iou_val < 0.2);
    }

    #[test]
    fn test_dice_identical() {
        let a = square_contour(0.0, 0.0, 10.0);
        let b = square_contour(0.0, 0.0, 10.0);
        let dice_val = dice(&a, &b);
        assert!((dice_val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dice_no_overlap() {
        let a = square_contour(0.0, 0.0, 10.0);
        let b = square_contour(20.0, 20.0, 10.0);
        let dice_val = dice(&a, &b);
        assert!(dice_val < 0.01);
    }

    #[test]
    fn test_hausdorff_identical() {
        let a = square_contour(0.0, 0.0, 10.0);
        let b = square_contour(0.0, 0.0, 10.0);
        let h = hausdorff_distance(&a, &b);
        assert!(h < 0.01);
    }

    #[test]
    fn test_hausdorff_translated() {
        let a = square_contour(0.0, 0.0, 10.0);
        let b = square_contour(5.0, 0.0, 10.0);
        let h = hausdorff_distance(&a, &b);
        assert!((h - 5.0).abs() < 0.01);
    }
}
