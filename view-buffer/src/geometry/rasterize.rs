//! Contour rasterization to binary masks.
//!
//! Converts vector contours to raster (pixel) representations.

use crate::core::buffer::ViewBuffer;

use super::contour::{Contour, Point};

/// Rasterizes a contour to a binary mask.
///
/// # Arguments
/// * `contour` - The contour to rasterize
/// * `width` - Output mask width in pixels
/// * `height` - Output mask height in pixels
/// * `fill_value` - Value for pixels inside the contour
/// * `background` - Value for pixels outside the contour
///
/// # Returns
/// A ViewBuffer with shape [height, width, 1] and dtype U8
pub fn rasterize(
    contour: &Contour,
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
) -> ViewBuffer {
    let w = width as usize;
    let h = height as usize;
    let mut data = vec![background; w * h];

    // Use scanline algorithm for efficiency
    scanline_fill(&contour.exterior, w, h, &mut data, fill_value);

    // Subtract holes
    for hole in &contour.holes {
        scanline_fill(hole, w, h, &mut data, background);
    }

    ViewBuffer::from_vec_with_shape(data, vec![h, w, 1])
}

/// Scanline polygon fill algorithm.
///
/// More efficient than point-in-polygon testing for each pixel.
///
/// A pixel belongs to the polygon when its **centre** — `(x + 0.5, y + 0.5)` —
/// lies inside it, boundary included. That is the rule `contains_point` and the
/// area measures follow, so a mask's pixel count tracks the contour's area rather
/// than exceeding it by a border. The test-only `rasterize_simple` asserts the
/// two agree pixel for pixel.
fn scanline_fill(polygon: &[Point], width: usize, height: usize, data: &mut [u8], value: u8) {
    if polygon.len() < 3 || width == 0 || height == 0 {
        return;
    }

    // Find y-range of the polygon
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in polygon {
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    // Clamp y coordinates to valid range [0, height)
    let start_y = (min_y.floor() as i32).max(0).min((height - 1) as i32) as usize;
    let end_y = ((max_y.ceil() as i32) + 1).min(height as i32) as usize;

    // For each scanline
    for y in start_y..end_y.min(height) {
        let scan_y = y as f64 + 0.5; // Sample at pixel center

        // Find all intersection points with edges
        let mut intersections: Vec<f64> = Vec::new();
        let n = polygon.len();

        for i in 0..n {
            let p1 = &polygon[i];
            let p2 = &polygon[(i + 1) % n];

            // Check if edge crosses this scanline
            if (p1.y <= scan_y && p2.y > scan_y) || (p2.y <= scan_y && p1.y > scan_y) {
                // Compute x intersection
                let t = (scan_y - p1.y) / (p2.y - p1.y);
                let x = p1.x + t * (p2.x - p1.x);
                intersections.push(x);
            }
        }

        // Sort intersections
        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Fill between pairs of intersections
        for i in (0..intersections.len()).step_by(2) {
            if i + 1 >= intersections.len() {
                break;
            }

            // Centres inside the span, i.e. `x + 0.5` in [left, right]. Rounding
            // the span itself outward instead — ceil(left), floor(right) — took in
            // a surplus column at every right edge, so a 100-wide box rasterized
            // 101 columns while the y loop, driven by `scan_y`, got it right.
            let first = (intersections[i] - 0.5).ceil();
            let last = (intersections[i + 1] - 0.5).floor();

            if last < 0.0 || first >= width as f64 {
                continue;
            }

            // Both ends are in range now, so the casts cannot saturate
            // surprisingly: `last` is non-negative and `first` is below `width`.
            let x_start = first.max(0.0) as usize;
            let x_end = (last as usize + 1).min(width);

            for x in x_start..x_end {
                data[y * width + x] = value;
            }
        }
    }
}

/// Rasterizes a contour by testing every pixel centre against the polygon.
///
/// The independent oracle [`rasterize`]'s scanline filler is checked against:
/// it asks `geo` the same question [`super::predicates::contains_point`] and the
/// area measures answer, one pixel at a time, sharing no code with the span
/// arithmetic. Too slow for production — O(w * h * n) against the scanline
/// filler's O(h * n) — so it stays test-only rather than becoming a second
/// rasterization path callers could pick.
#[cfg(test)]
fn rasterize_simple(
    contour: &Contour,
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
) -> ViewBuffer {
    let w = width as usize;
    let h = height as usize;
    let mut data = vec![background; w * h];

    // Converted once: `to_geo` allocates, and this loop runs per pixel. The
    // polygon carries its holes, so one position test replaces the old
    // exterior-then-holes walk (a point on a hole boundary still paints, as
    // before, because that reads as `0` rather than `-1`).
    let polygon = contour.to_geo();

    for y in 0..h {
        for x in 0..w {
            let point = Point::new(x as f64 + 0.5, y as f64 + 0.5);

            if super::predicates::position_in_polygon(&polygon, &point) >= 0 {
                data[y * w + x] = fill_value;
            }
        }
    }

    ViewBuffer::from_vec_with_shape(data, vec![h, w, 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dtype::DType;

    fn square_contour() -> Contour {
        Contour::from_tuples(&[(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)])
    }

    #[test]
    fn test_rasterize_shape() {
        let contour = square_contour();
        let mask = rasterize(&contour, 100, 100, 255, 0);

        assert_eq!(mask.shape(), &[100, 100, 1]);
        assert_eq!(mask.dtype(), DType::U8);
    }

    #[test]
    fn test_rasterize_content() {
        let contour = square_contour();
        let mask = rasterize(&contour, 100, 100, 255, 0);

        // Access the data
        let data = mask.to_contiguous();
        let ptr = unsafe { data.as_ptr::<u8>() };
        let slice = unsafe { std::slice::from_raw_parts(ptr, 100 * 100) };

        // Check corner (should be background = 0)
        assert_eq!(slice[0], 0);

        // Check center (should be fill = 255)
        assert_eq!(slice[50 * 100 + 50], 255);
    }

    #[test]
    fn test_rasterize_with_hole() {
        let exterior = vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
        ];
        let hole = vec![
            Point::new(30.0, 30.0),
            Point::new(70.0, 30.0),
            Point::new(70.0, 70.0),
            Point::new(30.0, 70.0),
        ];
        let contour = Contour::with_holes(exterior, vec![hole]);

        let mask = rasterize(&contour, 100, 100, 255, 0);
        let data = mask.to_contiguous();
        let ptr = unsafe { data.as_ptr::<u8>() };
        let slice = unsafe { std::slice::from_raw_parts(ptr, 100 * 100) };

        // Center (inside hole) should be background
        assert_eq!(slice[50 * 100 + 50], 0);

        // Point outside hole but inside exterior should be fill
        assert_eq!(slice[10 * 100 + 10], 255);
    }

    fn mask_pixels(mask: &ViewBuffer, len: usize) -> Vec<u8> {
        let data = mask.to_contiguous();
        let ptr = unsafe { data.as_ptr::<u8>() };
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    #[test]
    fn test_rasterize_simple_matches() {
        // On a canvas that holds the whole contour, and asserted pixel-for-pixel.
        // The previous version of this test used a 50x50 canvas for a contour
        // spanning [10, 90], so every column the two fillers disagreed about was
        // off-canvas, and it tolerated 50 differing pixels besides — which is why
        // the scanline filler's surplus right-hand column went unnoticed.
        for contour in [
            square_contour(),
            Contour::from_tuples(&[(10.0, 10.0), (90.0, 30.0), (60.0, 90.0)]),
        ] {
            let scanline = rasterize(&contour, 100, 100, 255, 0);
            let per_pixel = rasterize_simple(&contour, 100, 100, 255, 0);

            assert_eq!(
                mask_pixels(&scanline, 100 * 100),
                mask_pixels(&per_pixel, 100 * 100)
            );
        }
    }

    #[test]
    fn test_rasterized_pixel_count_is_the_area() {
        // An axis-aligned box on integer coordinates puts no pixel centre on an
        // edge, so the count is the area exactly. Filling [10, 90] rasterized 81
        // columns before the span rounding was fixed.
        let mask = rasterize(&square_contour(), 100, 100, 1, 0);
        let painted: u32 = mask_pixels(&mask, 100 * 100)
            .iter()
            .map(|&v| v as u32)
            .sum();

        assert_eq!(painted, 80 * 80);
    }

    #[test]
    fn test_rasterize_zero_sized_canvas_is_empty() {
        let mask = rasterize(&square_contour(), 0, 0, 255, 0);
        assert_eq!(mask.shape(), &[0, 0, 1]);
    }
}
