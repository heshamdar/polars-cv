//! Contour extraction from binary images.
//!
//! Finds the borders of a binary image's regions, traced along pixel edges.

use crate::core::buffer::ViewBuffer;

use super::contour::{Contour, Point};
use super::ops::{ApproxMethod, ExtractMode};

/// Extracts contours from a binary image.
///
/// Pixel `(x, y)` is the unit square `[x, x + 1] x [y, y + 1]` — the square
/// whose centre rasterization samples — and every contour runs along the
/// edges of those squares, its vertices on the integer lattice. A region's
/// outline therefore bounds exactly its pixels: its area is its pixel count
/// (one pixel is its unit square, area 1), and rasterizing it gives the mask
/// back. Traced through pixel *centres* instead, as this used to be, every
/// region came back inset by half a pixel, and a region one pixel thick came
/// back as a point or a line with no area at all.
///
/// Foreground regions are 8-connected and background regions 4-connected, the
/// pairing under which every border is one closed curve. Two pixels touching
/// only at a corner are one region, whose outline passes through that corner
/// twice.
///
/// Every foreground region yields its exterior. [`ExtractMode::All`] and
/// [`ExtractMode::Tree`] also yield one border per enclosed background region
/// (a hole), and include the regions inside holes; [`ExtractMode::External`]
/// keeps only the exteriors of regions that are not inside a hole. Contours
/// come in raster order of their first pixel.
///
/// # Arguments
/// * `buffer` - The binary image (should be U8 with values 0 or 255)
/// * `mode` - Which contours to extract
/// * `method` - How to approximate the contour
/// * `min_area` - Minimum area threshold (optional)
///
/// # Returns
/// Vector of extracted contours
pub fn extract_contours(
    buffer: &ViewBuffer,
    mode: ExtractMode,
    method: ApproxMethod,
    min_area: Option<f64>,
) -> Vec<Contour> {
    let shape = buffer.shape();
    if shape.len() < 2 {
        return Vec::new();
    }

    let height = shape[0];
    let width = shape[1];

    // Get the image data as contiguous bytes
    let contiguous = buffer.to_contiguous();
    let data = unsafe { std::slice::from_raw_parts(contiguous.as_ptr::<u8>(), height * width) };

    let background = label_background(data, width, height);
    let keep_holes = !matches!(mode, ExtractMode::External);
    let mut in_region = vec![false; width * height];
    let mut hole_traced = vec![false; background.regions];
    let mut contours = Vec::new();

    // The raster scan meets each region, and each hole, first at its top-left
    // pixel, whose top edge is on the region's own border — the pixel above it
    // cannot belong to the region, or the scan would have met that one first.
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let (xi, yi) = (x as isize, y as isize);

            if data[idx] > 0 {
                if in_region[idx] {
                    continue;
                }
                flood(idx, width, height, &NEIGHBOURS_8, |j| {
                    let joins = data[j] > 0 && !in_region[j];
                    in_region[j] |= joins;
                    joins
                });
                // The background above the first pixel is the one surrounding
                // the region: either the outside, or a hole of another region.
                let surrounded_by = if y == 0 {
                    OUTSIDE
                } else {
                    background.label[idx - width]
                };
                if keep_holes || surrounded_by == OUTSIDE {
                    contours.push(trace_border(data, width, height, (xi, yi), EAST));
                }
            } else {
                let hole = background.label[idx];
                if keep_holes && hole != OUTSIDE && !hole_traced[hole as usize] {
                    hole_traced[hole as usize] = true;
                    // Along the top edge, westward: the region above is then
                    // on the right, as the walk requires.
                    contours.push(trace_border(data, width, height, (xi + 1, yi), WEST));
                }
            }
        }
    }

    // Apply approximation
    let contours: Vec<Contour> = contours
        .into_iter()
        .map(|c| approximate_contour(c, method))
        .collect();

    // Filter by area
    match min_area {
        Some(min) => contours
            .into_iter()
            .filter(|c| super::measures::area(c, false) >= min)
            .collect(),
        None => contours,
    }
}

const NEIGHBOURS_4: [(isize, isize); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
const NEIGHBOURS_8: [(isize, isize); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Flood-fills from `start` over `steps`, visiting each pixel `claim` accepts
/// (it records the membership itself) and spreading only from those.
fn flood(
    start: usize,
    width: usize,
    height: usize,
    steps: &[(isize, isize)],
    mut claim: impl FnMut(usize) -> bool,
) {
    if !claim(start) {
        return;
    }
    let mut stack = vec![start];
    while let Some(i) = stack.pop() {
        let (x, y) = ((i % width) as isize, (i / width) as isize);
        for &(dx, dy) in steps {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let j = ny as usize * width + nx as usize;
            if claim(j) {
                stack.push(j);
            }
        }
    }
}

/// The label of the background region every background pixel on the image
/// edge belongs to — the region surrounding the image's outermost regions.
const OUTSIDE: u32 = 0;
/// The label a foreground pixel carries.
const FOREGROUND: u32 = u32::MAX;

/// The 4-connected background regions of a mask.
struct Background {
    /// Each pixel's region: [`OUTSIDE`], an enclosed region (a hole) numbered
    /// from 1, or [`FOREGROUND`].
    label: Vec<u32>,
    /// How many region labels are in use, [`OUTSIDE`] included.
    regions: usize,
}

fn label_background(data: &[u8], width: usize, height: usize) -> Background {
    const UNLABELLED: u32 = u32::MAX - 1;
    let mut label: Vec<u32> = data
        .iter()
        .map(|&v| if v > 0 { FOREGROUND } else { UNLABELLED })
        .collect();
    let fill = |label: &mut [u32], start: usize, id: u32| {
        flood(start, width, height, &NEIGHBOURS_4, |j| {
            let joins = label[j] == UNLABELLED;
            if joins {
                label[j] = id;
            }
            joins
        });
    };

    // Everything connected to the image edge is outside first, so the regions
    // numbered after are exactly the enclosed ones.
    for y in 0..height {
        for x in 0..width {
            if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
                fill(&mut label, y * width + x, OUTSIDE);
            }
        }
    }
    let mut regions = 1;
    for i in 0..width * height {
        if label[i] == UNLABELLED {
            fill(&mut label, i, regions as u32);
            regions += 1;
        }
    }
    Background { label, regions }
}

/// A unit step along the pixel lattice, `(dx, dy)` with y growing downward.
type Heading = (isize, isize);
const EAST: Heading = (1, 0);
const WEST: Heading = (-1, 0);

/// Walks one closed border along pixel edges, from lattice vertex `start`
/// along `heading`, keeping the foreground on its right.
///
/// "Right" as drawn: heading east along a pixel's top edge, that pixel lies to
/// the right. At each vertex the walk looks at the two pixels ahead of it, one
/// either side of the line it is on, and
///
/// - turns left if the ahead-left pixel is foreground — a concave corner, or
///   one touching the current pixel only diagonally, which 8-connectivity makes
///   part of the same region, so the walk goes round it;
/// - goes straight if only the ahead-right pixel is foreground;
/// - turns right if neither is — a convex corner.
///
/// Every border edge has exactly one successor and one predecessor under this
/// rule, so the walk returns to its first edge, and stops just before
/// repeating it. A vertex on the border twice (a diagonal pinch) is left by a
/// different edge each time, which is why the stop compares the heading too.
fn trace_border(
    data: &[u8],
    width: usize,
    height: usize,
    start: (isize, isize),
    heading: Heading,
) -> Contour {
    let foreground = |x: isize, y: isize| {
        x >= 0
            && y >= 0
            && (x as usize) < width
            && (y as usize) < height
            && data[y as usize * width + x as usize] > 0
    };
    // The pixel in quadrant `(sx, sy)` (each +-1) around lattice vertex `v`.
    let pixel = |v: (isize, isize), (sx, sy): (isize, isize)| {
        foreground(v.0 + (sx - 1) / 2, v.1 + (sy - 1) / 2)
    };

    let mut points = Vec::new();
    let (mut vertex, mut h) = (start, heading);
    loop {
        points.push(Point::new(vertex.0 as f64, vertex.1 as f64));
        vertex = (vertex.0 + h.0, vertex.1 + h.1);

        let left = (h.1, -h.0);
        let right = (-h.1, h.0);
        h = if pixel(vertex, (h.0 + left.0, h.1 + left.1)) {
            left
        } else if pixel(vertex, (h.0 + right.0, h.1 + right.1)) {
            h
        } else {
            right
        };

        if vertex == start && h == heading {
            break;
        }
        // A border has at most four edges per pixel, so a longer walk is a
        // broken rule, not a long border.
        assert!(
            points.len() <= 4 * width * height,
            "border walk from {start:?} did not close"
        );
    }

    Contour::new(points)
}

/// Applies contour approximation method.
fn approximate_contour(contour: Contour, method: ApproxMethod) -> Contour {
    match method {
        ApproxMethod::None => contour,
        ApproxMethod::Simple => simplify_collinear(contour),
        ApproxMethod::Approx => super::transforms::simplify(&contour, 1.0),
    }
}

/// Removes collinear points from a contour.
fn simplify_collinear(contour: Contour) -> Contour {
    if contour.exterior.len() < 3 {
        return contour;
    }

    let mut simplified = Vec::new();
    let n = contour.exterior.len();

    for i in 0..n {
        let prev = &contour.exterior[(i + n - 1) % n];
        let curr = &contour.exterior[i];
        let next = &contour.exterior[(i + 1) % n];

        // Check collinearity using cross product
        let cross = (curr.x - prev.x) * (next.y - curr.y) - (curr.y - prev.y) * (next.x - curr.x);

        if cross.abs() > 1e-6 {
            simplified.push(*curr);
        }
    }

    // Ensure we have at least 3 points
    if simplified.len() < 3 {
        return contour;
    }

    Contour::new(simplified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::measures;
    use crate::geometry::rasterize::rasterize;

    /// A mask with `fill` set on the cells the predicate selects.
    fn mask_from(width: usize, height: usize, fill: impl Fn(usize, usize) -> bool) -> ViewBuffer {
        let mut data = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                if fill(x, y) {
                    data[y * width + x] = 255;
                }
            }
        }
        ViewBuffer::from_vec_with_shape(data, vec![height, width, 1])
    }

    /// A white square filling pixels `[20, 80)^2` of a 100x100 canvas.
    fn square_image() -> ViewBuffer {
        mask_from(100, 100, |x, y| {
            (20..80).contains(&x) && (20..80).contains(&y)
        })
    }

    fn pixels(mask: &ViewBuffer) -> Vec<u8> {
        mask.to_contiguous().as_slice::<u8>().to_vec()
    }

    fn corners(contour: &Contour) -> Vec<(f64, f64)> {
        contour.exterior.iter().map(|p| (p.x, p.y)).collect()
    }

    /// Named single-region masks on a 40x40 canvas, none with a hole.
    fn solid_shapes() -> Vec<(&'static str, ViewBuffer)> {
        vec![
            ("pixel", mask_from(40, 40, |x, y| (x, y) == (5, 5))),
            (
                "1x5 line",
                mask_from(40, 40, |x, y| y == 5 && (3..8).contains(&x)),
            ),
            (
                "5x1 line",
                mask_from(40, 40, |x, y| x == 5 && (3..8).contains(&y)),
            ),
            (
                "diagonal pair",
                mask_from(40, 40, |x, y| (x, y) == (5, 5) || (x, y) == (6, 6)),
            ),
            (
                "diagonal line",
                mask_from(40, 40, |x, y| x == y && (3..12).contains(&x)),
            ),
            (
                "anti-diagonal",
                mask_from(40, 40, |x, y| x + y == 20 && (5..15).contains(&x)),
            ),
            (
                "L",
                mask_from(40, 40, |x, y| {
                    (x == 4 && (4..10).contains(&y)) || (y == 9 && (4..10).contains(&x))
                }),
            ),
            (
                "plus",
                mask_from(40, 40, |x, y| {
                    ((15..25).contains(&x) && (5..35).contains(&y))
                        || ((5..35).contains(&x) && (15..25).contains(&y))
                }),
            ),
            (
                "disc",
                mask_from(40, 40, |x, y| {
                    let (dx, dy) = (x as f64 - 20.0, y as f64 - 20.0);
                    dx * dx + dy * dy <= 12.0 * 12.0
                }),
            ),
            ("corner pixel", mask_from(40, 40, |x, y| (x, y) == (0, 0))),
            ("full canvas", mask_from(40, 40, |_, _| true)),
            ("edge band", mask_from(40, 40, |x, _| x >= 37)),
        ]
    }

    #[test]
    fn test_extract_single_contour() {
        let contours = extract_contours(
            &square_image(),
            ExtractMode::External,
            ApproxMethod::None,
            None,
        );

        // One filled square is one contour. Before the tracer was fixed this
        // returned 59 — one degenerate 2x2 walk per row of the square.
        assert_eq!(contours.len(), 1);

        // The outline runs along the pixel edges of [20, 80)^2: one lattice
        // vertex per unit of its 4 * 60 perimeter, each on a side of the square.
        let boundary = &contours[0].exterior;
        assert_eq!(boundary.len(), 4 * 60);
        for p in boundary {
            let on_rim = (p.x == 20.0 || p.x == 80.0) && (20.0..=80.0).contains(&p.y)
                || (p.y == 20.0 || p.y == 80.0) && (20.0..=80.0).contains(&p.x);
            assert!(on_rim, "traced point ({}, {}) is not on the rim", p.x, p.y);
        }
        for corner in [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)] {
            assert!(
                boundary.iter().any(|p| (p.x, p.y) == corner),
                "corner {corner:?} missing from the trace"
            );
        }
    }

    #[test]
    fn test_simple_square_is_its_four_pixel_corners() {
        let contours = extract_contours(
            &square_image(),
            ExtractMode::External,
            ApproxMethod::Simple,
            None,
        );
        assert_eq!(
            corners(&contours[0]),
            vec![(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
        );
    }

    /// One pixel is the unit square it covers — area 1, not a point of area 0.
    #[test]
    fn test_extract_isolated_pixel() {
        let image = mask_from(20, 20, |x, y| x == 5 && y == 5);
        let contours = extract_contours(&image, ExtractMode::External, ApproxMethod::None, None);

        assert_eq!(contours.len(), 1);
        assert_eq!(
            corners(&contours[0]),
            vec![(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]
        );
    }

    /// Traced along pixel edges, every region's polygon area is its pixel
    /// count — the one-pixel-thick ones included, which traced through pixel
    /// centres had no area at all.
    #[test]
    fn test_area_is_the_pixel_count() {
        for (name, mask) in solid_shapes() {
            let count = pixels(&mask).iter().filter(|&&v| v > 0).count() as f64;
            for method in [ApproxMethod::None, ApproxMethod::Simple] {
                let contours = extract_contours(&mask, ExtractMode::External, method, None);
                assert_eq!(contours.len(), 1, "{name}: one region, one contour");
                assert_eq!(
                    measures::area(&contours[0], false),
                    count,
                    "{name} {method:?}"
                );
            }
        }
    }

    /// Rasterizing samples pixel centres, and an edge-traced outline has every
    /// region pixel's centre strictly inside it and every other one strictly
    /// outside, so the round trip gives back the mask exactly.
    #[test]
    fn test_round_trip_through_rasterize_is_exact() {
        for (name, mask) in solid_shapes() {
            let contours =
                extract_contours(&mask, ExtractMode::External, ApproxMethod::Simple, None);
            let back = rasterize(&contours, 40, 40, 255, 0);
            assert_eq!(pixels(&back), pixels(&mask), "{name}");
        }
    }

    /// Pixels touching only at a corner are one 8-connected region, as the
    /// Moore tracer had them: one outline, pinched at the shared vertex.
    #[test]
    fn test_diagonal_neighbours_are_one_region() {
        let image = mask_from(10, 10, |x, y| (x, y) == (5, 5) || (x, y) == (6, 6));
        let contours = extract_contours(&image, ExtractMode::All, ApproxMethod::Simple, None);
        assert_eq!(contours.len(), 1);
        assert_eq!(
            corners(&contours[0]),
            vec![
                (5.0, 5.0),
                (6.0, 5.0),
                (6.0, 6.0),
                (7.0, 6.0),
                (7.0, 7.0),
                (6.0, 7.0),
                (6.0, 6.0),
                (5.0, 6.0)
            ]
        );
    }

    #[test]
    fn test_extract_separates_two_regions() {
        let image = mask_from(100, 100, |x, y| {
            (10..30).contains(&x) && (10..30).contains(&y)
                || (60..90).contains(&x) && (60..90).contains(&y)
        });
        let contours = extract_contours(&image, ExtractMode::External, ApproxMethod::None, None);

        assert_eq!(contours.len(), 2);
    }

    #[test]
    fn test_extract_hourglass_waist_is_traced_once_through() {
        // Two blocks joined by a two-cell neck: one region, one exterior.
        let image = mask_from(40, 40, |x, y| {
            (10..30).contains(&x) && (10..19).contains(&y)
                || (10..30).contains(&x) && (21..30).contains(&y)
                || (x == 19 && y == 19)
                || (x == 19 && y == 20)
        });
        let contours = extract_contours(&image, ExtractMode::External, ApproxMethod::None, None);

        assert_eq!(contours.len(), 1);
        assert_eq!(measures::area(&contours[0], false), (20 * 9 * 2 + 2) as f64);
    }

    /// `all` yields the exterior plus one border per enclosed background
    /// region, each bounding exactly its pixels; `external` only the first.
    #[test]
    fn test_hole_border_bounds_the_hole_pixels() {
        // A 20x20 block with a 6x4 hole.
        let image = mask_from(40, 40, |x, y| {
            (10..30).contains(&x)
                && (10..30).contains(&y)
                && !((15..21).contains(&x) && (12..16).contains(&y))
        });

        let external = extract_contours(&image, ExtractMode::External, ApproxMethod::Simple, None);
        assert_eq!(external.len(), 1);
        assert_eq!(measures::area(&external[0], false), 400.0);

        let all = extract_contours(&image, ExtractMode::All, ApproxMethod::Simple, None);
        assert_eq!(all.len(), 2);
        let hole = all.iter().find(|c| measures::area(c, false) == 24.0);
        assert_eq!(
            hole.map(|c| c.bounding_box().map(|b| (b.x, b.y, b.width, b.height))),
            Some(Some((15.0, 12.0, 6.0, 4.0)))
        );
    }

    /// A region inside another's hole is not external.
    #[test]
    fn test_island_in_a_hole_is_not_external() {
        let image = mask_from(40, 40, |x, y| {
            let ring = (5..35).contains(&x)
                && (5..35).contains(&y)
                && !((10..30).contains(&x) && (10..30).contains(&y));
            let island = (15..25).contains(&x) && (15..25).contains(&y);
            ring || island
        });
        let external = extract_contours(&image, ExtractMode::External, ApproxMethod::Simple, None);
        assert_eq!(external.len(), 1);
        assert_eq!(measures::area(&external[0], false), 900.0);

        let all = extract_contours(&image, ExtractMode::All, ApproxMethod::Simple, None);
        let mut areas: Vec<f64> = all.iter().map(|c| measures::area(c, false)).collect();
        areas.sort_by(f64::total_cmp);
        assert_eq!(areas, vec![100.0, 400.0, 900.0]);
    }

    #[test]
    fn test_extract_with_min_area() {
        // The outline bounds the square's 60 * 60 pixels exactly.
        let square = square_image();
        let at = |min| {
            extract_contours(
                &square,
                ExtractMode::External,
                ApproxMethod::None,
                Some(min),
            )
        };
        assert_eq!(at(3600.0).len(), 1);
        assert!(at(3600.5).is_empty());
    }
}
