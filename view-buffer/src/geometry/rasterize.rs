//! Contour rasterization to binary masks.
//!
//! Converts vector contours to raster (pixel) representations.

use crate::core::buffer::ViewBuffer;

use super::contour::{Contour, Point};
use super::predicates::point_in_polygon;

/// Rasterizes a contour to a binary mask.
///
/// # Arguments
/// * `contour` - The contour to rasterize
/// * `width` - Output mask width in pixels
/// * `height` - Output mask height in pixels
/// * `fill_value` - Value for pixels inside the contour
/// * `background` - Value for pixels outside the contour
/// * `anti_alias` - Whether to apply anti-aliasing (not yet implemented)
///
/// # Returns
/// A ViewBuffer with shape [height, width, 1] and dtype U8
pub fn rasterize(
    contour: &Contour,
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
    _anti_alias: bool,
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
fn scanline_fill(polygon: &[Point], width: usize, height: usize, data: &mut [u8], value: u8) {
    if polygon.len() < 3 {
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

            // Clamp x coordinates to valid range [0, width)
            let x_start = (intersections[i].ceil() as i32)
                .max(0)
                .min((width - 1) as i32) as usize;
            let x_end = ((intersections[i + 1].floor() as i32) + 1).min(width as i32) as usize;

            // Ensure we don't exceed bounds
            for x in x_start..x_end.min(width) {
                let idx = y * width + x;
                if idx < data.len() {
                    data[idx] = value;
                }
            }
        }
    }
}

/// Rasterizes a contour with simple point-in-polygon testing.
///
/// Less efficient but more straightforward implementation.
/// Useful for validation and small contours.
#[allow(dead_code)]
pub fn rasterize_simple(
    contour: &Contour,
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
) -> ViewBuffer {
    let w = width as usize;
    let h = height as usize;
    let mut data = vec![background; w * h];

    for y in 0..h {
        for x in 0..w {
            let point = Point::new(x as f64 + 0.5, y as f64 + 0.5);
            let result = point_in_polygon(&point, &contour.exterior);

            let mut inside = result > 0;

            // Check holes
            if inside {
                for hole in &contour.holes {
                    if point_in_polygon(&point, hole) > 0 {
                        inside = false;
                        break;
                    }
                }
            }

            if inside || result == 0 {
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
        let mask = rasterize(&contour, 100, 100, 255, 0, false);

        assert_eq!(mask.shape(), &[100, 100, 1]);
        assert_eq!(mask.dtype(), DType::U8);
    }

    #[test]
    fn test_rasterize_content() {
        let contour = square_contour();
        let mask = rasterize(&contour, 100, 100, 255, 0, false);

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

        let mask = rasterize(&contour, 100, 100, 255, 0, false);
        let data = mask.to_contiguous();
        let ptr = unsafe { data.as_ptr::<u8>() };
        let slice = unsafe { std::slice::from_raw_parts(ptr, 100 * 100) };

        // Center (inside hole) should be background
        assert_eq!(slice[50 * 100 + 50], 0);

        // Point outside hole but inside exterior should be fill
        assert_eq!(slice[10 * 100 + 10], 255);
    }

    #[test]
    fn test_rasterize_simple_matches() {
        let contour = square_contour();

        let mask1 = rasterize(&contour, 50, 50, 255, 0, false);
        let mask2 = rasterize_simple(&contour, 50, 50, 255, 0);

        let data1 = mask1.to_contiguous();
        let data2 = mask2.to_contiguous();

        let slice1 = unsafe { std::slice::from_raw_parts(data1.as_ptr::<u8>(), 50 * 50) };
        let slice2 = unsafe { std::slice::from_raw_parts(data2.as_ptr::<u8>(), 50 * 50) };

        // Should produce identical results (or very close due to edge handling)
        let mut diff_count = 0;
        for i in 0..(50 * 50) {
            if slice1[i] != slice2[i] {
                diff_count += 1;
            }
        }
        // Allow small differences at boundaries
        assert!(diff_count < 50); // Less than 2% difference
    }
}
