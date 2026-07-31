//! Pairwise geometric operations between contours.
//!
//! Implements IoU, Dice coefficient, and Hausdorff distance.
//!
//! Overlap is computed exactly, by handing both regions to `geo`'s boolean
//! operations. Each contour is first reduced to its filled region — exterior minus
//! the union of its hole rings, see [`Contour::to_geo_region`] — so area and
//! intersection are measured on the same thing. Results are winding-independent
//! and hole-aware, and a contour always scores 1.0 against itself regardless of
//! point order, concavity, or holes.

use super::contour::{BoundingBox, Contour};
use geo::{Area, BooleanOps, BoundingRect, HausdorffDistance, Intersects, MultiPolygon};

/// Areas below this are treated as degenerate.
const EPSILON: f64 = 1e-10;

/// Pairwise matching result for a set of predictions and ground truths.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionMatchResult {
    /// Prediction indices in processing order.
    pub pred_idx: Vec<usize>,
    /// Matched GT index for each prediction, or None when unmatched.
    pub gt_idx: Vec<Option<usize>>,
    /// IoU value associated with each prediction match decision.
    pub iou: Vec<f64>,
    /// Total number of predictions.
    pub n_preds: usize,
    /// Total number of ground truths.
    pub n_gts: usize,
    /// Number of true positives.
    pub n_tp: usize,
    /// Number of false positives.
    pub n_fp: usize,
    /// Number of false negatives.
    pub n_fn: usize,
}

/// Areas of both regions and of their exact intersection.
///
/// Returns `None` when either region is degenerate or the two cannot possibly
/// overlap, letting the callers short-circuit to zero. The bounding-box test is
/// what keeps `match_detections` cheap: most pairs are disjoint and never reach
/// the boolean op.
///
/// Both inputs must come from [`Contour::to_geo_region`], whose output is already
/// a canonically-oriented region. That makes the result independent of the fill
/// rule — the choice that silently decided how nested rings were treated when the
/// holes were still carried as interior rings.
fn intersection_areas(a: &MultiPolygon<f64>, b: &MultiPolygon<f64>) -> Option<(f64, f64, f64)> {
    let (area_a, area_b) = (a.unsigned_area(), b.unsigned_area());

    if area_a < EPSILON || area_b < EPSILON {
        return None;
    }

    if !a.bounding_rect()?.intersects(&b.bounding_rect()?) {
        return None;
    }

    Some((area_a, area_b, a.intersection(b).unsigned_area()))
}

/// Computes Intersection over Union (IoU) between two contours.
///
/// IoU = intersection_area / union_area, computed exactly for arbitrary simple
/// polygons — concave shapes and holes included, in either winding direction.
///
/// # Arguments
/// * `a` - First contour
/// * `b` - Second contour
///
/// # Returns
/// IoU value in [0, 1]
pub fn iou(a: &Contour, b: &Contour) -> f64 {
    iou_regions(&a.to_geo_region(), &b.to_geo_region())
}

fn iou_regions(a: &MultiPolygon<f64>, b: &MultiPolygon<f64>) -> f64 {
    let Some((area_a, area_b, intersection_area)) = intersection_areas(a, b) else {
        return 0.0;
    };

    // Inclusion-exclusion is exact only because both sides are measured on the
    // same region: `to_geo_region` has already differenced the holes out, so
    // `unsigned_area` and the boolean op cannot disagree about what is filled.
    let union_area = area_a + area_b - intersection_area;

    if union_area < EPSILON {
        return 0.0;
    }

    (intersection_area / union_area).clamp(0.0, 1.0)
}

/// Computes a full pairwise IoU matrix between two contour sets.
///
/// Returns an `N x M` matrix where `N = a.len()` and `M = b.len()`.
/// Matrix element `(i, j)` is `iou(&a[i], &b[j])`.
pub fn iou_matrix(a: &[Contour], b: &[Contour]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }

    if b.is_empty() {
        return vec![Vec::new(); a.len()];
    }

    // Convert once per contour rather than once per pair.
    let a_regions: Vec<MultiPolygon<f64>> = a.iter().map(Contour::to_geo_region).collect();
    let b_regions: Vec<MultiPolygon<f64>> = b.iter().map(Contour::to_geo_region).collect();

    a_regions
        .iter()
        .map(|ra| b_regions.iter().map(|rb| iou_regions(ra, rb)).collect())
        .collect()
}

/// Greedy one-to-one detection matching using IoU thresholding.
///
/// Predictions are processed in `pred_order` if provided, otherwise in natural
/// order. For each prediction, the unmatched GT with highest IoU is selected.
/// Ties are broken by choosing the smallest GT index for determinism.
pub fn match_detections(
    preds: &[Contour],
    gts: &[Contour],
    threshold: f64,
    pred_order: Option<&[usize]>,
) -> DetectionMatchResult {
    match_from_matrix(
        iou_matrix(preds, gts),
        preds.len(),
        gts.len(),
        threshold,
        pred_order,
    )
}

/// The greedy matcher itself, over a precomputed IoU matrix.
///
/// Shared by the contour and bounding-box entry points so the matching policy
/// lives in exactly one place.
fn match_from_matrix(
    matrix: Vec<Vec<f64>>,
    n_preds: usize,
    n_gts: usize,
    threshold: f64,
    pred_order: Option<&[usize]>,
) -> DetectionMatchResult {
    let order: Vec<usize> = match pred_order {
        Some(indices) => indices.to_vec(),
        None => (0..n_preds).collect(),
    };

    let mut gt_taken = vec![false; n_gts];
    let mut gt_by_pred: Vec<Option<usize>> = vec![None; n_preds];
    let mut iou_by_pred: Vec<f64> = vec![0.0; n_preds];

    for pred_idx in order {
        if pred_idx >= n_preds {
            continue;
        }

        let mut best_gt: Option<usize> = None;
        let mut best_iou = -1.0_f64;
        for (gt_idx, is_taken) in gt_taken.iter().enumerate().take(n_gts) {
            if *is_taken {
                continue;
            }
            let cand_iou = matrix[pred_idx][gt_idx];
            if cand_iou > best_iou {
                best_iou = cand_iou;
                best_gt = Some(gt_idx);
            } else if (cand_iou - best_iou).abs() < 1e-12
                && matches!(best_gt, Some(current_best) if gt_idx < current_best)
            {
                best_gt = Some(gt_idx);
            }
        }

        match best_gt {
            Some(gt_idx) if best_iou >= threshold => {
                gt_taken[gt_idx] = true;
                gt_by_pred[pred_idx] = Some(gt_idx);
                iou_by_pred[pred_idx] = best_iou;
            }
            _ => {
                gt_by_pred[pred_idx] = None;
                iou_by_pred[pred_idx] = 0.0;
            }
        }
    }

    let n_tp = gt_by_pred.iter().filter(|v| v.is_some()).count();
    let n_fp = n_preds.saturating_sub(n_tp);
    let n_fn = n_gts.saturating_sub(n_tp);

    DetectionMatchResult {
        pred_idx: (0..n_preds).collect(),
        gt_idx: gt_by_pred,
        iou: iou_by_pred,
        n_preds,
        n_gts,
        n_tp,
        n_fp,
        n_fn,
    }
}

/// Convert an axis-aligned bounding box to a 4-point rectangular contour.
pub fn bbox_to_contour(bbox: &BoundingBox) -> Contour {
    Contour::from_tuples(&[
        (bbox.x, bbox.y),
        (bbox.x + bbox.width, bbox.y),
        (bbox.x + bbox.width, bbox.y + bbox.height),
        (bbox.x, bbox.y + bbox.height),
    ])
}

/// IoU between two axis-aligned bounding boxes.
///
/// Rectangle overlap is a two-interval intersection, so this stays analytic rather
/// than going through general polygon boolean ops.
pub fn bbox_iou(a: &BoundingBox, b: &BoundingBox) -> f64 {
    let intersection = a.intersection(b).map_or(0.0, |r| r.area());

    if intersection < EPSILON {
        return 0.0;
    }

    let union = a.area() + b.area() - intersection;

    if union < EPSILON {
        return 0.0;
    }

    (intersection / union).clamp(0.0, 1.0)
}

/// Pairwise IoU matrix between two sets of bounding boxes.
pub fn bbox_iou_matrix(a: &[BoundingBox], b: &[BoundingBox]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|ba| b.iter().map(|bb| bbox_iou(ba, bb)).collect())
        .collect()
}

/// Greedy one-to-one detection matching on bounding boxes.
pub fn bbox_match_detections(
    preds: &[BoundingBox],
    gts: &[BoundingBox],
    threshold: f64,
    pred_order: Option<&[usize]>,
) -> DetectionMatchResult {
    match_from_matrix(
        bbox_iou_matrix(preds, gts),
        preds.len(),
        gts.len(),
        threshold,
        pred_order,
    )
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
    let Some((area_a, area_b, intersection_area)) =
        intersection_areas(&a.to_geo_region(), &b.to_geo_region())
    else {
        return 0.0;
    };

    let denominator = area_a + area_b;

    if denominator < EPSILON {
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
    // `geo` folds with `Bounded::min_value()`, so an empty coordinate set yields
    // -f64::MAX rather than propagating emptiness. A distance is never negative.
    if a.exterior.is_empty() || b.exterior.is_empty() {
        return f64::INFINITY;
    }

    // Only the coordinates are read, so pass the rings rather than building
    // polygons whose closing points would be walked twice by the O(n*m) loop.
    a.to_geo()
        .exterior()
        .hausdorff_distance(b.to_geo().exterior())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::contour::{Point, Winding};
    use crate::geometry::transforms;

    fn square_contour(x: f64, y: f64, size: f64) -> Contour {
        Contour::from_tuples(&[(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
    }

    /// The same square as [`square_contour`], wound the other way.
    fn cw_square_contour(x: f64, y: f64, size: f64) -> Contour {
        Contour::from_tuples(&[(x, y), (x, y + size), (x + size, y + size), (x + size, y)])
    }

    /// Concave: a 100x100 square with the top-right quadrant removed. Area 7500.
    fn l_shape(dx: f64, dy: f64) -> Contour {
        Contour::from_tuples(&[
            (dx, dy),
            (dx + 100.0, dy),
            (dx + 100.0, dy + 50.0),
            (dx + 50.0, dy + 50.0),
            (dx + 50.0, dy + 100.0),
            (dx, dy + 100.0),
        ])
    }

    /// Concave with a reflex notch cut into one side. Area 100x100 - 40x60 = 7600.
    fn u_shape() -> Contour {
        Contour::from_tuples(&[
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (70.0, 100.0),
            (70.0, 40.0),
            (30.0, 40.0),
            (30.0, 100.0),
            (0.0, 100.0),
        ])
    }

    /// A 100x100 square with a 50x50 hole. Net area 10000 - 2500 = 7500.
    fn holed_square(hole_winding: Winding) -> Contour {
        let hole = match hole_winding {
            Winding::Clockwise => vec![(25.0, 25.0), (25.0, 75.0), (75.0, 75.0), (75.0, 25.0)],
            Winding::CounterClockwise => {
                vec![(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)]
            }
        };
        Contour::with_holes(
            vec![
                Point::new(0.0, 0.0),
                Point::new(100.0, 0.0),
                Point::new(100.0, 100.0),
                Point::new(0.0, 100.0),
            ],
            vec![hole.into_iter().map(|(x, y)| Point::new(x, y)).collect()],
        )
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

    // --- Winding independence ---
    //
    // `holes` is the only carrier of hole-ness in the contour spec; point order is
    // never interpreted as a hole signal. These pin that as behaviour.

    #[test]
    fn test_iou_cw_identical() {
        let a = cw_square_contour(0.0, 0.0, 10.0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_mixed_winding_identical() {
        let ccw = square_contour(0.0, 0.0, 10.0);
        let cw = cw_square_contour(0.0, 0.0, 10.0);
        assert!((iou(&ccw, &cw) - 1.0).abs() < 1e-9);
        assert!((iou(&cw, &ccw) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_is_symmetric() {
        let a = l_shape(0.0, 0.0);
        let b = l_shape(25.0, 25.0);
        assert!((iou(&a, &b) - iou(&b, &a)).abs() < 1e-12);
    }

    #[test]
    fn test_iou_winding_does_not_change_result() {
        let a = l_shape(0.0, 0.0);
        let b = l_shape(25.0, 25.0);
        let flipped_b = transforms::flip(&b);
        assert!((iou(&a, &b) - iou(&a, &flipped_b)).abs() < 1e-9);
    }

    // --- Concave polygons ---
    //
    // Sutherland-Hodgman clipping is only valid for a convex clip polygon, so these
    // are the cases that silently under-reported. Real segmentation contours are
    // essentially never convex.

    #[test]
    fn test_iou_l_shape_identical() {
        let a = l_shape(0.0, 0.0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_u_shape_identical() {
        let a = u_shape();
        assert!((iou(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_concave_partial_overlap() {
        // `a` covers x:[0,100] y:[0,50] plus x:[0,50] y:[50,100].
        // `b` is the same shape offset by (25,25): x:[25,125] y:[25,75] plus
        // x:[25,75] y:[75,125]. Overlapping rectangles:
        //   x:[25,100] y:[25,50] = 1875
        //   x:[25,50]  y:[50,75] =  625
        //   x:[25,50]  y:[75,100] = 625
        // giving 3125 of intersection against a union of 7500 + 7500 - 3125.
        let a = l_shape(0.0, 0.0);
        let b = l_shape(25.0, 25.0);
        let expected = 3125.0 / 11875.0;
        assert!((iou(&a, &b) - expected).abs() < 1e-9, "got {}", iou(&a, &b));
    }

    // --- Holes ---
    //
    // `area()` subtracts holes, so the intersection must account for them too or the
    // ratio exceeds 1 (or the union goes negative and the result collapses to 0).

    #[test]
    fn test_iou_holed_identical() {
        for winding in [Winding::Clockwise, Winding::CounterClockwise] {
            let a = holed_square(winding);
            assert!((iou(&a, &a) - 1.0).abs() < 1e-9, "winding {winding:?}");
        }
    }

    #[test]
    fn test_iou_holed_vs_solid() {
        // Asserted for BOTH hole windings on purpose. Only the asymmetric
        // holed-vs-solid comparison can detect a hole being silently ignored — a
        // holed contour against *itself* saturates at 1.0 under the final clamp
        // whatever the intersection does, so identity tests cannot catch it.
        for winding in [Winding::Clockwise, Winding::CounterClockwise] {
            let holed = holed_square(winding);
            let solid = square_contour(0.0, 0.0, 100.0);
            // intersection = the holed shape (7500); union = the solid square (10000).
            assert!(
                (iou(&holed, &solid) - 0.75).abs() < 1e-9,
                "winding {winding:?} gave {}",
                iou(&holed, &solid)
            );
        }
    }

    /// A 100x100 square whose hole contains a further ring.
    ///
    /// Every ring in `holes` is a hole, so the region is the exterior minus the
    /// union of the hole rings: 10000 - 6400 = 3600. The inner ring changes
    /// nothing — it lies inside a part that has already been removed.
    fn nested_holes() -> Contour {
        let ring = |pts: &[(f64, f64)]| pts.iter().map(|&(x, y)| Point::new(x, y)).collect();
        Contour::with_holes(
            ring(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]),
            vec![
                ring(&[(10.0, 10.0), (10.0, 90.0), (90.0, 90.0), (90.0, 10.0)]),
                ring(&[(40.0, 40.0), (40.0, 60.0), (60.0, 60.0), (60.0, 40.0)]),
            ],
        )
    }

    #[test]
    fn test_nested_hole_area_is_the_removed_union() {
        // Not 10000 - 6400 - 400: the rings overlap, so subtracting each in turn
        // double-counts. `contains_point` and `rasterize` have always agreed on
        // 3600; this is the value `area` must report too.
        assert!((super::super::measures::area(&nested_holes(), false) - 3600.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_nested_holes_identical() {
        let a = nested_holes();
        assert!((iou(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_nested_holes_vs_solid() {
        // The case the clamp hides: if the boolean op fills the inner ring while
        // `area` subtracts it, the intersection exceeds the smaller area and this
        // ratio comes out too high.
        let nested = nested_holes();
        let solid = square_contour(0.0, 0.0, 100.0);
        assert!(
            (iou(&nested, &solid) - 0.36).abs() < 1e-9,
            "got {}",
            iou(&nested, &solid)
        );
    }

    #[test]
    fn test_dice_nested_holes_vs_solid() {
        let nested = nested_holes();
        let solid = square_contour(0.0, 0.0, 100.0);
        // 2 * 3600 / (3600 + 10000)
        assert!((dice(&nested, &solid) - (7200.0 / 13600.0)).abs() < 1e-9);
    }

    #[test]
    fn test_hausdorff_empty_contour_is_infinite() {
        // Never a finite value, and never negative: `geo` folds with
        // `Bounded::min_value()`, which yields -f64::MAX for an empty coord set.
        let empty = Contour::new(Vec::new());
        let square = square_contour(0.0, 0.0, 10.0);

        assert_eq!(hausdorff_distance(&empty, &square), f64::INFINITY);
        assert_eq!(hausdorff_distance(&square, &empty), f64::INFINITY);
        assert_eq!(hausdorff_distance(&empty, &empty), f64::INFINITY);
    }

    #[test]
    fn test_iou_hole_winding_is_not_a_signal() {
        let cw_hole = holed_square(Winding::Clockwise);
        let ccw_hole = holed_square(Winding::CounterClockwise);
        assert!((iou(&cw_hole, &ccw_hole) - 1.0).abs() < 1e-9);
    }

    // --- Aggregates built on `iou` ---

    #[test]
    fn test_iou_matrix_matches_pairwise() {
        let a = vec![l_shape(0.0, 0.0), u_shape()];
        let b = vec![u_shape(), l_shape(50.0, 50.0)];
        let matrix = iou_matrix(&a, &b);
        for (i, ca) in a.iter().enumerate() {
            for (j, cb) in b.iter().enumerate() {
                assert!((matrix[i][j] - iou(ca, cb)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_match_detections_matches_concave_shapes_to_themselves() {
        let shapes = vec![l_shape(0.0, 0.0), u_shape(), l_shape(500.0, 500.0)];
        let result = match_detections(&shapes, &shapes, 0.5, None);
        assert_eq!(result.n_tp, 3);
        assert_eq!(result.n_fp, 0);
        assert_eq!(result.n_fn, 0);
        for (i, gt) in result.gt_idx.iter().enumerate() {
            assert_eq!(*gt, Some(i));
        }
    }

    // --- Dice shares the same intersection path ---

    #[test]
    fn test_dice_cw_identical() {
        let a = cw_square_contour(0.0, 0.0, 10.0);
        assert!((dice(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dice_l_shape_identical() {
        let a = l_shape(0.0, 0.0);
        assert!((dice(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dice_holed_identical() {
        let a = holed_square(Winding::Clockwise);
        assert!((dice(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dice_holed_vs_solid() {
        let holed = holed_square(Winding::Clockwise);
        let solid = square_contour(0.0, 0.0, 100.0);
        // 2 * 7500 / (7500 + 10000)
        assert!((dice(&holed, &solid) - (15000.0 / 17500.0)).abs() < 1e-9);
    }

    // --- Bounding boxes ---

    #[test]
    fn test_bbox_iou_identical() {
        let b = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        assert!((bbox_iou(&b, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_partial() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
        assert!((bbox_iou(&a, &b) - 25.0 / 175.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_disjoint() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(20.0, 20.0, 10.0, 10.0);
        assert_eq!(bbox_iou(&a, &b), 0.0);
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
