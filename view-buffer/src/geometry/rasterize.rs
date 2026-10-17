//! Contour rasterization to binary masks.
//!
//! Converts vector contours to raster (pixel) representations.

use crate::core::buffer::ViewBuffer;

use super::contour::{Contour, Point};

/// Rasterizes a set of contours to a mask.
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
/// Each pixel is sampled at its `supersample × supersample` sub-pixel centres,
/// `(x + (i + 0.5) / n, y + (j + 0.5) / n)`, and painted
/// `background + (fill_value − background) · k / n²` for its `k` covered
/// samples, stored by the conversion rule (round half away from zero). At
/// `n = 1` that is the pixel centre and a pixel is wholly one value or the
/// other: the rule `contains_point` and the area measures follow.
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
/// * `supersample` - Samples per pixel along each axis, at least 1
///
/// # Returns
/// A ViewBuffer with shape [height, width, 1] and dtype U8
pub fn rasterize<C: std::borrow::Borrow<Contour>>(
    contours: &[C],
    width: u32,
    height: u32,
    fill_value: u8,
    background: u8,
    supersample: u32,
) -> ViewBuffer {
    assert!(
        supersample >= 1,
        "rasterize: supersample must be at least 1"
    );
    let (w, h, n) = (width as usize, height as usize, supersample as usize);
    // Covered samples per pixel; n² is at most the op's 16² = 256.
    let mut counts = vec![0u32; w * h];
    // One row of samples at a time: the union, and one contour's coverage
    // (so its holes subtract from itself only).
    let mut covered = vec![false; w * n];
    let mut scratch = vec![false; w * n];
    // Each scanline's edge crossings, reused across lines and rings.
    let mut crossings: Vec<f64> = Vec::new();
    let spans: Vec<Option<(f64, f64)>> = contours
        .iter()
        .map(|c| y_extent(&c.borrow().exterior))
        .collect();

    for y in 0..h {
        for j in 0..n {
            let scan_y = y as f64 + (j as f64 + 0.5) / n as f64;
            covered.fill(false);
            for (contour, extent) in contours.iter().zip(&spans) {
                let contour = contour.borrow();
                if !extent.is_some_and(|(lo, hi)| scan_y >= lo && scan_y <= hi) {
                    continue;
                }
                scratch.fill(false);
                fill_scanline(
                    &contour.exterior,
                    scan_y,
                    n,
                    &mut scratch,
                    true,
                    &mut crossings,
                );
                for hole in &contour.holes {
                    fill_scanline(hole, scan_y, n, &mut scratch, false, &mut crossings);
                }
                for (slot, &inside) in covered.iter_mut().zip(&scratch) {
                    *slot |= inside;
                }
            }
            let row = &mut counts[y * w..(y + 1) * w];
            for (sx, &inside) in covered.iter().enumerate() {
                if inside {
                    row[sx / n] += 1;
                }
            }
        }
    }

    let samples = (n * n) as f64;
    let (fill, bg) = (f64::from(fill_value), f64::from(background));
    let data = counts
        .into_iter()
        .map(|k| match k as usize {
            0 => background,
            k if k == n * n => fill_value,
            k => crate::core::convert::CastFrom::cast_from(bg + (fill - bg) * k as f64 / samples),
        })
        .collect();

    ViewBuffer::from_vec_with_shape(data, vec![h, w, 1])
}

/// The `[min, max]` y-extent of a ring, or `None` for a ring that fills
/// nothing (fewer than three points).
fn y_extent(polygon: &[Point]) -> Option<(f64, f64)> {
    if polygon.len() < 3 {
        return None;
    }
    let (lo, hi) = polygon
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p.y), hi.max(p.y))
        });
    Some((lo, hi))
}

/// Set `row`'s samples inside `polygon` on the horizontal line `scan_y` to
/// `value`; `row` holds `n` samples per pixel, sample `s` at `(s + 0.5) / n`.
///
/// A sample belongs to the polygon when it lies inside it, boundary
/// included: the crossings of the line with each edge (an edge that starts
/// on the line counts, one that ends on it does not, so a vertex is crossed
/// once) bound spans `[left, right]`, and the samples whose positions fall in
/// a span are `ceil(left · n − 0.5)` to `floor(right · n − 0.5)`. At `n = 1`
/// those are the pixel centres `x + 0.5` in `[left, right]`. Rounding the
/// span itself outward instead — `ceil(left)`, `floor(right)` — took in a
/// surplus column at every right edge, so a 100-wide box rasterized 101
/// columns. The test-only `rasterize_simple` asserts the rule pixel for pixel.
fn fill_scanline(
    polygon: &[Point],
    scan_y: f64,
    n: usize,
    row: &mut [bool],
    value: bool,
    intersections: &mut Vec<f64>,
) {
    if polygon.len() < 3 || row.is_empty() {
        return;
    }
    intersections.clear();
    let count = polygon.len();
    for i in 0..count {
        let p1 = &polygon[i];
        let p2 = &polygon[(i + 1) % count];
        if (p1.y <= scan_y && p2.y > scan_y) || (p2.y <= scan_y && p1.y > scan_y) {
            let t = (scan_y - p1.y) / (p2.y - p1.y);
            intersections.push(p1.x + t * (p2.x - p1.x));
        }
    }
    intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let samples = row.len() as f64;
    let scale = n as f64;
    for &[left, right] in intersections.as_chunks::<2>().0 {
        let first = (left * scale - 0.5).ceil();
        let last = (right * scale - 0.5).floor();
        if last < 0.0 || first >= samples {
            continue;
        }
        // Both ends are in range now, so the casts cannot saturate
        // surprisingly: `last` is non-negative and `first` is below the row.
        let start = first.max(0.0) as usize;
        let end = (last as usize + 1).min(row.len());
        if start < end {
            row[start..end].fill(value);
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
        let mask = rasterize(&one(&contour), 100, 100, 255, 0, 1);

        assert_eq!(mask.shape(), &[100, 100, 1]);
        assert_eq!(mask.dtype(), DType::U8);
    }

    #[test]
    fn test_rasterize_content() {
        let contour = square_contour();
        let mask = rasterize(&one(&contour), 100, 100, 255, 0, 1);

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

        let mask = rasterize(&one(&contour), 100, 100, 255, 0, 1);
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
            let scanline = rasterize(&contours, 100, 100, 255, 0, 1);
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
        let mask = rasterize(&one(&square_contour()), 100, 100, 1, 0, 1);
        let painted: u32 = mask_pixels(&mask, 100 * 100)
            .iter()
            .map(|&v| v as u32)
            .sum();

        assert_eq!(painted, 80 * 80);
    }

    #[test]
    fn test_rasterize_zero_sized_canvas_is_empty() {
        let mask = rasterize(&one(&square_contour()), 0, 0, 255, 0, 1);
        assert_eq!(mask.shape(), &[0, 0, 1]);
    }

    #[test]
    fn test_rasterize_empty_set_is_all_background() {
        let mask = rasterize::<Contour>(&[], 10, 10, 255, 7, 1);
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
        let painted: u32 = mask_pixels(&rasterize(&contours, 100, 100, 1, 0, 1), 100 * 100)
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
        let forward = mask_pixels(&rasterize(&contours, 100, 100, 1, 0, 1), 100 * 100);
        contours.reverse();
        let reversed = mask_pixels(&rasterize(&contours, 100, 100, 1, 0, 1), 100 * 100);

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
        let upright = mask_pixels(&rasterize(&contours, 100, 100, 255, 0, 1), 100 * 100);
        let inverted = mask_pixels(&rasterize(&contours, 100, 100, 0, 255, 1), 100 * 100);

        for (up, inv) in upright.iter().zip(inverted.iter()) {
            assert_eq!(*up, 255 - *inv);
        }
        assert!(inverted.contains(&0), "nothing was filled");
    }

    /// The definition of supersampling, independently: geo's
    /// point-in-polygon at every sub-pixel centre, counted and blended.
    fn rasterize_sampled_simple(
        contours: &[Contour],
        w: usize,
        h: usize,
        fill: u8,
        background: u8,
        n: usize,
    ) -> Vec<u8> {
        let polygons: Vec<_> = contours.iter().map(|c| c.to_geo()).collect();
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let mut k = 0usize;
                for j in 0..n {
                    for i in 0..n {
                        let p = Point::new(
                            x as f64 + (i as f64 + 0.5) / n as f64,
                            y as f64 + (j as f64 + 0.5) / n as f64,
                        );
                        if polygons
                            .iter()
                            .any(|g| super::super::predicates::position_in_polygon(g, &p) >= 0)
                        {
                            k += 1;
                        }
                    }
                }
                let v = f64::from(background)
                    + (f64::from(fill) - f64::from(background)) * k as f64 / (n * n) as f64;
                out.push(<u8 as crate::core::convert::CastFrom<f64>>::cast_from(v));
            }
        }
        out
    }

    /// Float vertices chosen so every sample, at every count the tests use,
    /// lies at least 2.8e-5 from every edge and vertex row (checked in exact
    /// rational arithmetic). On an edge, the two tests can legitimately
    /// disagree through rounding alone: the first choice of fixture put a
    /// sample exactly on one at n = 3.
    fn float_shapes() -> Vec<Contour> {
        vec![
            Contour::with_holes(
                vec![
                    Point::new(3.317, 2.713),
                    Point::new(27.131, 4.907),
                    Point::new(22.643, 25.329),
                    Point::new(6.211, 21.837),
                ],
                vec![vec![
                    Point::new(10.419, 9.137),
                    Point::new(17.713, 10.331),
                    Point::new(14.227, 16.619),
                ]],
            ),
            Contour::from_tuples(&[(20.233, 15.143), (31.709, 18.427), (24.917, 29.311)]),
        ]
    }

    #[test]
    fn supersampling_counts_every_sub_pixel_centre() {
        let shapes = float_shapes();
        for n in [1u32, 2, 3, 4, 7, 16] {
            let got = rasterize(&shapes, 33, 31, 255, 0, n);
            assert_eq!(
                mask_pixels(&got, 33 * 31),
                rasterize_sampled_simple(&shapes, 33, 31, 255, 0, n as usize),
                "n = {n}"
            );
        }
    }

    #[test]
    fn an_axis_aligned_edge_covers_its_exact_fraction() {
        let rect = Contour::from_tuples(&[(2.25, 1.0), (5.75, 1.0), (5.75, 3.0), (2.25, 3.0)]);
        let got = mask_pixels(&rasterize(&one(&rect), 8, 4, 255, 0, 4), 32);
        let row = [0, 0, 191, 255, 255, 191, 0, 0];
        assert_eq!(&got[8..16], &row);
        assert_eq!(&got[16..24], &row);
        assert!(got[..8].iter().chain(&got[24..]).all(|&v| v == 0));
    }

    #[test]
    fn supersampled_union_and_inversion_hold() {
        let contours = overlapping_set();
        let a = mask_pixels(&rasterize(&contours, 100, 100, 255, 0, 3), 100 * 100);
        let rev: Vec<Contour> = contours.iter().rev().cloned().collect();
        let b = mask_pixels(&rasterize(&rev, 100, 100, 255, 0, 3), 100 * 100);
        let inverted = mask_pixels(&rasterize(&contours, 100, 100, 0, 255, 3), 100 * 100);
        assert_eq!(a, b, "the mask depends on the set's order");
        for (up, inv) in a.iter().zip(&inverted) {
            assert_eq!(*up, 255 - *inv);
        }
    }
}
