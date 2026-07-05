//! Label reduction: score contour regions against a single-channel buffer.
//!
//! The scoring math lives here in the engine (it was previously implemented
//! inside the polars-cv plugin's graph executor, which also copied the whole
//! image into a `Vec<Vec<f64>>` grid first). This implementation reads the
//! contiguous buffer directly through a dtype-dispatched flat accessor.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::geometry::contour::{Contour, Point};
use crate::geometry::{measures, predicates};

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
    /// Interior pixels plus pixels on the contour boundary (avoids
    /// zero-score artifacts for sub-pixel contours).
    Boundary,
    /// All pixels within the contour's bounding box.
    Bbox,
}

crate::naming::named_variants!(LabelReduction {
    "max" => Max,
    "mean" => Mean,
    "sum" => Sum,
});

crate::naming::named_variants!(LabelRegionMode {
    "interior" => Interior,
    "boundary" => Boundary,
    "bbox" => Bbox,
});

/// Score every contour's region over a single-channel `[H, W]`/`[H, W, 1]`
/// buffer, returning one value per contour.
///
/// Pixels are sampled at their centers (`x + 0.5`, `y + 0.5`). When a
/// contour's region contains no pixel (sub-pixel contours), the value at the
/// contour centroid is returned instead so single-pixel detections receive
/// their actual pixel value.
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

    let x0 = bbox.x.floor().max(0.0) as usize;
    let y0 = bbox.y.floor().max(0.0) as usize;
    let x1 = (bbox.x + bbox.width).ceil().min(width as f64).max(0.0) as usize;
    let y1 = (bbox.y + bbox.height).ceil().min(height as f64).max(0.0) as usize;
    if x0 >= x1 || y0 >= y1 {
        return 0.0;
    }

    let mut acc = 0.0;
    let mut max_val = f64::NEG_INFINITY;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let include = match region_mode {
                LabelRegionMode::Bbox => true,
                LabelRegionMode::Interior => {
                    predicates::contains_point(contour, x as f64 + 0.5, y as f64 + 0.5)
                }
                LabelRegionMode::Boundary => {
                    predicates::point_in_contour(
                        &Point::new(x as f64 + 0.5, y as f64 + 0.5),
                        contour,
                    ) >= 0
                }
            };
            if include {
                let val = at(y, x);
                acc += val;
                max_val = max_val.max(val);
                count += 1;
            }
        }
    }
    if count == 0 {
        // Centroid fallback: sub-pixel contours have an empty rasterized
        // interior. Sample the buffer at the contour centroid instead so
        // that single-pixel detections receive their actual pixel value.
        let c = measures::centroid(contour);
        let cx = c.x.floor() as usize;
        let cy = c.y.floor() as usize;
        if cy < height && cx < width {
            return at(cy, cx);
        }
        return 0.0;
    }
    match reduction {
        LabelReduction::Max => max_val,
        LabelReduction::Mean => acc / count as f64,
        LabelReduction::Sum => acc,
    }
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

    #[test]
    fn subpixel_contour_falls_back_to_centroid() {
        let mut data = vec![0.0f32; 16];
        data[2 * 4 + 2] = 9.0;
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![4, 4, 1]);
        // Sub-pixel contour centered on pixel (2, 2): interior catches no
        // pixel-center sample, so the centroid value must be returned.
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
