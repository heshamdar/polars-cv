//! Label reduction: score contour regions against a single-channel buffer.
//!
//! The scoring math lives here in the engine (it was previously implemented
//! inside the polars-cv plugin's graph executor, which also copied the whole
//! image into a `Vec<Vec<f64>>` grid first). This implementation reads the
//! contiguous buffer directly through a dtype-dispatched flat accessor.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::geometry::contour::{BoundingBox, Contour, Point};
use crate::geometry::{measures, pairwise, predicates};
use crate::ops::util::maximum;

/// Reduction applied over the pixel values of a contour's region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelReduction {
    Max,
    Mean,
    Sum,
}

/// Which pixels count as a contour's region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelRegionMode {
    /// Only pixels strictly inside the contour polygon.
    Interior,
    /// Interior pixels plus pixels whose centre lies on the contour boundary.
    Boundary,
    /// All pixels within the contour's bounding box.
    Bbox,
}

crate::naming::named_variants!(LabelReduction: "Reduction over a contour region's pixel values (``label_reduce``)." {
    "max" => Max,
    "mean" => Mean,
    "sum" => Sum,
});

crate::naming::named_variants!(LabelRegionMode: "Region selection for ``label_reduce``.\n\n- INTERIOR: Pixels strictly inside the contour polygon.\n- BOUNDARY: Interior pixels plus pixels on the contour boundary.\n- BBOX: All pixels within the bounding box." {
    "interior" => Interior,
    "boundary" => Boundary,
    "bbox" => Bbox,
});

/// Score every contour's region over a single-channel `[H, W]`/`[H, W, 1]`
/// buffer, returning one value per contour.
///
/// Pixels are sampled at their centers (`x + 0.5`, `y + 0.5`). A contour
/// extracted from a mask bounds exactly its region's pixels, so its region is
/// those pixels in `Interior` and `Boundary` mode alike. A contour with no area
/// (a point or a line) or whose region contains no pixel centre (a sub-pixel
/// contour) is scored on the pixels its outline passes through instead, in
/// every region mode, rather than as 0.0. A contour that touches no in-bounds
/// pixel scores 0.0.
pub fn score_contours_on_buffer(
    buffer: &ViewBuffer,
    contours: &[Contour],
    reduction: LabelReduction,
    region_mode: LabelRegionMode,
) -> Result<Vec<f64>, String> {
    let shape = buffer.shape();
    if shape.len() < 2 {
        return Err(format!(
            "label_reduce requires at least 2D buffer input, got shape {shape:?}"
        ));
    }
    let height = shape[0];
    let width = shape[1];
    let channels = if shape.len() > 2 { shape[2] } else { 1 };
    if channels != 1 {
        return Err(format!(
            "label_reduce currently requires a single-channel buffer, got {channels} channels"
        ));
    }

    let contig = buffer.to_contiguous();

    // Dtype-dispatched flat accessor: (y, x) -> f64, no grid copy.
    macro_rules! with_accessor {
        ($t:ty) => {{
            let data = contig.as_slice::<$t>();
            let at = |y: usize, x: usize| -> f64 { data[y * width + x] as f64 };
            contours
                .iter()
                .map(|c| score_one(c, &at, width, height, reduction, region_mode))
                .collect()
        }};
    }
    let scores: Vec<f64> = match contig.dtype() {
        DType::U8 => with_accessor!(u8),
        DType::I8 => with_accessor!(i8),
        DType::U16 => with_accessor!(u16),
        DType::I16 => with_accessor!(i16),
        DType::U32 => with_accessor!(u32),
        DType::I32 => with_accessor!(i32),
        DType::U64 => with_accessor!(u64),
        DType::I64 => with_accessor!(i64),
        DType::F32 => with_accessor!(f32),
        DType::F64 => with_accessor!(f64),
    };
    Ok(scores)
}

fn score_one(
    contour: &Contour,
    at: &dyn Fn(usize, usize) -> f64,
    width: usize,
    height: usize,
    reduction: LabelReduction,
    region_mode: LabelRegionMode,
) -> f64 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let Some(bbox) = contour.bounding_box() else {
        return 0.0;
    };

    // A point or a line has no region to scan in any mode: its pixels are the
    // ones its path covers.
    // Deciding that by area rather than by an empty scan matters for a
    // diagonal line, whose path runs through pixel centres that `Boundary`
    // would otherwise pick up (all but the last).
    let degenerate = measures::area(contour, false) < pairwise::EPSILON;
    let x0 = bbox.x.floor().max(0.0) as usize;
    let y0 = bbox.y.floor().max(0.0) as usize;
    let x1 = (bbox.x + bbox.width).ceil().min(width as f64).max(0.0) as usize;
    let y1 = (bbox.y + bbox.height).ceil().min(height as f64).max(0.0) as usize;
    let (x1, y1) = if degenerate { (x0, y0) } else { (x1, y1) };

    let mut acc = 0.0;
    let mut max_val = f64::NEG_INFINITY;
    let mut count = 0usize;
    // Converted once for the whole scan: `to_geo` allocates, and the two
    // point-in-contour modes below test every pixel in the bounding box.
    let polygon = (region_mode != LabelRegionMode::Bbox).then(|| contour.to_geo());
    for y in y0..y1 {
        for x in x0..x1 {
            let point = Point::new(x as f64 + 0.5, y as f64 + 0.5);
            let include = match (region_mode, &polygon) {
                (LabelRegionMode::Bbox, _) => true,
                (LabelRegionMode::Interior, Some(polygon)) => {
                    predicates::position_in_polygon(polygon, &point) > 0
                }
                (LabelRegionMode::Boundary, Some(polygon)) => {
                    predicates::position_in_polygon(polygon, &point) >= 0
                }
                _ => unreachable!("polygon is built for every non-Bbox mode"),
            };
            if include {
                let val = at(y, x);
                acc += val;
                max_val = maximum(max_val, val);
                count += 1;
            }
        }
    }
    if count == 0 {
        // A degenerate contour, or a sub-pixel one whose region holds no
        // pixel centre: its pixels are the ones its outline passes through.
        for (x, y) in path_pixels(&contour.exterior, width, height) {
            let val = at(y, x);
            acc += val;
            max_val = maximum(max_val, val);
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }
    }
    match reduction {
        LabelReduction::Max => max_val,
        LabelReduction::Mean => acc / count as f64,
        LabelReduction::Sum => acc,
    }
}

/// The in-bounds pixels `(x, y)` a closed ring passes through, each once.
///
/// Pixel `(x, y)` covers `[x, x + 1) x [y, y + 1)`, the cell whose centre the
/// region scan samples. Each segment is clipped to the buffer first, so a
/// contour reaching far outside it costs no more than one that does not, then
/// sampled at least once per pixel of travel along its longer axis — exactly
/// the cells of the axis-aligned and diagonal segments contour extraction
/// produces.
fn path_pixels(ring: &[Point], width: usize, height: usize) -> Vec<(usize, usize)> {
    let (w, h) = (width as f64, height as f64);
    let buffer = BoundingBox::new(0.0, 0.0, w, h);
    let mut pixels = Vec::new();
    for (i, a) in ring.iter().enumerate() {
        let b = &ring[(i + 1) % ring.len()];
        let Some((a, b)) = buffer.clip_segment(a, b) else {
            continue;
        };
        let steps = (b.x - a.x).abs().max((b.y - a.y).abs()).ceil() as usize;
        for k in 0..=steps {
            let t = if steps == 0 {
                0.0
            } else {
                k as f64 / steps as f64
            };
            let (x, y) = (
                (a.x + t * (b.x - a.x)).floor(),
                (a.y + t * (b.y - a.y)).floor(),
            );
            if (0.0..w).contains(&x) && (0.0..h).contains(&y) {
                pixels.push((x as usize, y as usize));
            }
        }
    }
    pixels.sort_unstable();
    pixels.dedup();
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, size: f64) -> Contour {
        Contour::from_tuples(&[
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
        ])
    }

    /// The flat-accessor implementation must match a naive Vec<Vec<f64>>
    /// grid reference bit-for-bit.
    #[test]
    fn score_contours_matches_grid_reference() {
        let (h, w) = (8usize, 8usize);
        let data: Vec<f32> = (0..h * w).map(|i| (i % 13) as f32 * 0.5).collect();
        let buffer = ViewBuffer::from_vec_with_shape(data.clone(), vec![h, w, 1]);
        let grid: Vec<Vec<f64>> = (0..h)
            .map(|y| (0..w).map(|x| data[y * w + x] as f64).collect())
            .collect();

        let contour = square(1.0, 1.0, 5.0);
        for reduction in [
            LabelReduction::Max,
            LabelReduction::Mean,
            LabelReduction::Sum,
        ] {
            for mode in [
                LabelRegionMode::Interior,
                LabelRegionMode::Boundary,
                LabelRegionMode::Bbox,
            ] {
                let scores = score_contours_on_buffer(
                    &buffer,
                    std::slice::from_ref(&contour),
                    reduction,
                    mode,
                )
                .unwrap();
                // Naive reference: same loops over the copied grid.
                let mut acc = 0.0;
                let mut max_val = f64::NEG_INFINITY;
                let mut count = 0usize;
                for (y, row) in grid.iter().enumerate().skip(1).take(5) {
                    for (x, v) in row.iter().enumerate().skip(1).take(5) {
                        let include = match mode {
                            LabelRegionMode::Bbox => true,
                            LabelRegionMode::Interior => {
                                predicates::contains_point(&contour, x as f64 + 0.5, y as f64 + 0.5)
                            }
                            LabelRegionMode::Boundary => {
                                predicates::point_in_contour(
                                    &Point::new(x as f64 + 0.5, y as f64 + 0.5),
                                    &contour,
                                ) >= 0
                            }
                        };
                        if include {
                            acc += v;
                            max_val = max_val.max(*v);
                            count += 1;
                        }
                    }
                }
                let expected = match reduction {
                    LabelReduction::Max => max_val,
                    LabelReduction::Mean => acc / count as f64,
                    LabelReduction::Sum => acc,
                };
                assert_eq!(scores[0], expected, "{reduction:?}/{mode:?}");
            }
        }
    }

    /// A NaN pixel in a region makes its score NaN, for every reduction and
    /// wherever it lies (numpy's `max`/`mean`/`sum`, the one ordering rule):
    /// `Max` folded with `f64::max`, which drops a NaN, so the score was
    /// the region's largest number.
    #[test]
    fn a_nan_pixel_makes_the_score_nan() {
        for at in [0usize, 5, 15] {
            let mut data: Vec<f32> = (0..16).map(|i| i as f32).collect();
            data[at] = f32::NAN;
            let buffer = ViewBuffer::from_vec_with_shape(data, vec![4, 4, 1]);
            for reduction in [
                LabelReduction::Max,
                LabelReduction::Mean,
                LabelReduction::Sum,
            ] {
                let scores = score_contours_on_buffer(
                    &buffer,
                    &[square(0.0, 0.0, 4.0)],
                    reduction,
                    LabelRegionMode::Bbox,
                )
                .unwrap();
                assert!(
                    scores[0].is_nan(),
                    "NaN at {at}, {reduction:?}: {}",
                    scores[0]
                );
            }
        }
    }

    #[test]
    fn subpixel_contour_is_scored_on_the_pixel_it_lies_in() {
        let mut data = vec![0.0f32; 16];
        data[2 * 4 + 2] = 9.0;
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![4, 4, 1]);
        // Sub-pixel contour centered on pixel (2, 2): interior catches no
        // pixel-center sample, so the pixel its outline lies in scores it.
        let tiny = square(2.3, 2.3, 0.2);
        let scores = score_contours_on_buffer(
            &buffer,
            &[tiny],
            LabelReduction::Max,
            LabelRegionMode::Interior,
        )
        .unwrap();
        assert_eq!(scores[0], 9.0);
    }

    /// A point or a line — zero area, a zero-width or zero-height bounding
    /// box, no pixel centre inside — is scored on the pixels its path passes
    /// through, in every region mode. It used to return 0.0 before the
    /// fallback was reached; when extraction traced pixel centres, that is
    /// what every one-pixel-thick region became.
    #[test]
    fn thin_contour_is_scored_on_the_pixels_it_passes_through() {
        let (h, w) = (8usize, 8usize);
        let data: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![h, w, 1]);
        let value = |x: usize, y: usize| (y * w + x) as f64;

        // (contour, the pixels it covers as (x, y))
        let cases: Vec<(Contour, Vec<(usize, usize)>)> = vec![
            (Contour::from_tuples(&[(5.0, 5.0)]), vec![(5, 5)]),
            (
                Contour::from_tuples(&[(5.0, 5.0), (6.0, 5.0)]),
                vec![(5, 5), (6, 5)],
            ),
            // A `method="simple"` 1x5 line: two end vertices, three pixels
            // between them that only the segment covers.
            (
                Contour::from_tuples(&[(2.0, 3.0), (6.0, 3.0)]),
                (2..=6).map(|x| (x, 3)).collect(),
            ),
            (
                Contour::from_tuples(&[(4.0, 1.0), (4.0, 4.0)]),
                (1..=4).map(|y| (4, y)).collect(),
            ),
            (
                Contour::from_tuples(&[(1.0, 1.0), (3.0, 3.0)]),
                vec![(1, 1), (2, 2), (3, 3)],
            ),
        ];

        for (contour, pixels) in &cases {
            let values: Vec<f64> = pixels.iter().map(|&(x, y)| value(x, y)).collect();
            let sum: f64 = values.iter().sum();
            for mode in [
                LabelRegionMode::Interior,
                LabelRegionMode::Boundary,
                LabelRegionMode::Bbox,
            ] {
                for (reduction, expected) in [
                    (
                        LabelReduction::Max,
                        values.iter().copied().fold(f64::MIN, f64::max),
                    ),
                    (LabelReduction::Mean, sum / values.len() as f64),
                    (LabelReduction::Sum, sum),
                ] {
                    let scores = score_contours_on_buffer(
                        &buffer,
                        std::slice::from_ref(contour),
                        reduction,
                        mode,
                    )
                    .unwrap();
                    assert_eq!(
                        scores[0], expected,
                        "{:?} {reduction:?}/{mode:?}",
                        contour.exterior
                    );
                }
            }
        }
    }

    /// Only the in-bounds part of a thin contour's path is read.
    #[test]
    fn thin_contour_path_is_clipped_to_the_buffer() {
        let buffer = ViewBuffer::from_vec_with_shape(vec![1.0f32; 16], vec![4, 4, 1]);
        let line = Contour::from_tuples(&[(2.0, 1.0), (9.0, 1.0)]);
        let outside = Contour::from_tuples(&[(7.0, 7.0)]);
        let scores = score_contours_on_buffer(
            &buffer,
            &[line, outside],
            LabelReduction::Sum,
            LabelRegionMode::Interior,
        )
        .unwrap();
        assert_eq!(scores, vec![2.0, 0.0]);
    }

    #[test]
    fn multichannel_buffer_is_rejected() {
        let buffer = ViewBuffer::from_vec_with_shape(vec![0u8; 12], vec![2, 2, 3]);
        let err = score_contours_on_buffer(
            &buffer,
            &[square(0.0, 0.0, 1.0)],
            LabelReduction::Max,
            LabelRegionMode::Interior,
        )
        .unwrap_err();
        assert!(err.contains("single-channel"), "{err}");
    }
}
