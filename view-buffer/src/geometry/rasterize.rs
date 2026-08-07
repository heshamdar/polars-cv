//! Contour rasterization to binary masks.
//!
//! Converts vector contours to raster (pixel) representations.

use crate::core::buffer::ViewBuffer;

use super::contour::{Contour, Point};

/// Rasterizes a set of contours to a binary mask.
///
/// The painted region is the **union** of the contours' regions, each region
/// being that contour's exterior minus its own holes. Union, not sequential
/// painting: one contour's hole cannot erase another contour's fill, and the
/// result does not depend on the order the set arrives in.
///
/// Coverage is decided first and coloured once, so `fill_value` and `background`
/// are free to be inverted (`fill_value < background`) — the same region is
/// painted either way. Rasterizing each contour separately and folding the
/// masks with `max`, which is what the graph executor used to do, quietly
/// returned an all-background canvas for an inverted pair.
///
/// A single contour is the one-element case, not a separate entry point: every
/// caller holds a set (`parse_contour_set` for the contour source, the contour
/// domain's own vector for the `rasterize` op), so there is nothing for a
/// single-contour shorthand to serve.
///
/// # Arguments
/// * `contours` - The contours to rasterize; an empty set paints nothing
/// * `width` - Output mask width in pixels
/// * `height` - Output mask height in pixels
/// * `fill_value` - Value for pixels inside any contour
/// * `background` - Value for pixels outside every contour
///
/// # Returns
/// A ViewBuffer with shape [height, width, 1] and dtype U8
pub fn rasterize(
    contours: &[Contour],
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
) -> ViewBuffer {
    let w = width as usize;
    let h = height as usize;
    let mut covered = vec![false; w * h];
    // One contour's coverage at a time, so its holes subtract from itself only.
    let mut scratch = vec![0u8; w * h];

    for contour in contours {
        scratch.fill(0);

        // Use scanline algorithm for efficiency
        scanline_fill(&contour.exterior, w, h, &mut scratch, 1);

        // Subtract holes
        for hole in &contour.holes {
            scanline_fill(hole, w, h, &mut scratch, 0);
        }

        for (slot, painted) in covered.iter_mut().zip(scratch.iter()) {
            *slot |= *painted != 0;
        }
    }

    let data = covered
        .into_iter()
        .map(|inside| if inside { fill_value } else { background })
        .collect();

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

/// Rasterizes contours by testing every pixel centre against each polygon.
///
/// The independent oracle [`rasterize`]'s scanline filler is checked against:
/// it asks `geo` the same question [`super::predicates::contains_point`] and the
/// area measures answer, one pixel at a time, sharing no code with the span
/// arithmetic. Too slow for production — O(w * h * n) against the scanline
/// filler's O(h * n) — so it stays test-only rather than becoming a second
/// rasterization path callers could pick.
#[cfg(test)]
fn rasterize_simple(
    contours: &[Contour],
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
    let polygons: Vec<_> = contours.iter().map(|c| c.to_geo()).collect();

    for y in 0..h {
        for x in 0..w {
            let point = Point::new(x as f64 + 0.5, y as f64 + 0.5);

            if polygons
                .iter()
                .any(|p| super::predicates::position_in_polygon(p, &point) >= 0)
            {
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

    /// The one-element set, spelled once so the single-contour tests read as
    /// what they are: `rasterize` over a set of one.
    fn one(contour: &Contour) -> [Contour; 1] {
        [contour.clone()]
    }

    #[test]
    fn test_rasterize_shape() {
        let contour = square_contour();
        let mask = rasterize(&one(&contour), 100, 100, 255, 0);

        assert_eq!(mask.shape(), &[100, 100, 1]);
        assert_eq!(mask.dtype(), DType::U8);
    }

    #[test]
    fn test_rasterize_content() {
        let contour = square_contour();
        let mask = rasterize(&one(&contour), 100, 100, 255, 0);

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

        let mask = rasterize(&one(&contour), 100, 100, 255, 0);
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
        for contours in [
            vec![square_contour()],
            vec![Contour::from_tuples(&[
                (10.0, 10.0),
                (90.0, 30.0),
                (60.0, 90.0),
            ])],
            // A set whose second contour's hole falls inside the first's fill.
            // Both orders, because painting the set sequentially only goes wrong
            // when the hole arrives after the fill it would eat.
            overlapping_set(),
            overlapping_set().into_iter().rev().collect(),
        ] {
            let scanline = rasterize(&contours, 100, 100, 255, 0);
            let per_pixel = rasterize_simple(&contours, 100, 100, 255, 0);

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
        let mask = rasterize(&one(&square_contour()), 100, 100, 1, 0);
        let painted: u32 = mask_pixels(&mask, 100 * 100)
            .iter()
            .map(|&v| v as u32)
            .sum();

        assert_eq!(painted, 80 * 80);
    }

    #[test]
    fn test_rasterize_zero_sized_canvas_is_empty() {
        let mask = rasterize(&one(&square_contour()), 0, 0, 255, 0);
        assert_eq!(mask.shape(), &[0, 0, 1]);
    }

    #[test]
    fn test_rasterize_empty_set_is_all_background() {
        let mask = rasterize(&[], 10, 10, 255, 7);
        assert_eq!(mask.shape(), &[10, 10, 1]);
        assert!(mask_pixels(&mask, 100).iter().all(|&v| v == 7));
    }

    /// Two boxes: `[10, 50]` with a hole at `[20, 40]`, and `[30, 70]` covering
    /// part of that hole. The union owes the overlap to the second box.
    fn overlapping_set() -> Vec<Contour> {
        let holed = Contour::with_holes(
            vec![
                Point::new(10.0, 10.0),
                Point::new(50.0, 10.0),
                Point::new(50.0, 50.0),
                Point::new(10.0, 50.0),
            ],
            vec![vec![
                Point::new(20.0, 20.0),
                Point::new(40.0, 20.0),
                Point::new(40.0, 40.0),
                Point::new(20.0, 40.0),
            ]],
        );
        let overlapping =
            Contour::from_tuples(&[(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)]);
        vec![holed, overlapping]
    }

    #[test]
    fn test_disjoint_contours_are_both_painted() {
        let contours = vec![
            Contour::from_tuples(&[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]),
            Contour::from_tuples(&[(60.0, 60.0), (90.0, 60.0), (90.0, 90.0), (60.0, 90.0)]),
        ];
        let painted: u32 = mask_pixels(&rasterize(&contours, 100, 100, 1, 0), 100 * 100)
            .iter()
            .map(|&v| v as u32)
            .sum();

        assert_eq!(painted, 20 * 20 + 30 * 30);
    }

    #[test]
    fn test_a_hole_does_not_erase_another_contours_fill() {
        // Painting the set one contour at a time into a shared canvas lets a
        // later contour's hole cut into an earlier contour's fill, which makes
        // the mask depend on the order the set arrived in. Both orders are
        // asserted: with the holed contour first, sequential painting happens to
        // agree with the union, so a single order proves nothing.
        let mut contours = overlapping_set();
        let forward = mask_pixels(&rasterize(&contours, 100, 100, 1, 0), 100 * 100);
        contours.reverse();
        let reversed = mask_pixels(&rasterize(&contours, 100, 100, 1, 0), 100 * 100);

        assert_eq!(forward, reversed, "the mask depends on the set's order");
        // Inside the hole but also inside the overlapping box: covered.
        assert_eq!(forward[35 * 100 + 35], 1);
        // Inside the hole and outside the overlapping box: not covered.
        assert_eq!(forward[25 * 100 + 25], 0);
    }

    #[test]
    fn test_inverted_fill_and_background_paint_the_same_region() {
        // `fill_value < background` used to fold to an all-background canvas as
        // soon as the set held more than one contour.
        let contours = overlapping_set();
        let upright = mask_pixels(&rasterize(&contours, 100, 100, 255, 0), 100 * 100);
        let inverted = mask_pixels(&rasterize(&contours, 100, 100, 0, 255), 100 * 100);

        for (up, inv) in upright.iter().zip(inverted.iter()) {
            assert_eq!(*up, 255 - *inv);
        }
        assert!(inverted.contains(&0), "nothing was filled");
    }
}
