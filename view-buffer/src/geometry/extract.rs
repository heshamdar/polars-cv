//! Contour extraction from binary images.
//!
//! Finds contours (boundaries) in binary/thresholded images.

use crate::core::buffer::ViewBuffer;

use super::contour::{Contour, Point};
use super::ops::{ApproxMethod, ExtractMode};

/// Extracts contours from a binary image.
///
/// Uses a border-following algorithm similar to OpenCV's findContours.
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

    // Create a mutable copy for marking visited pixels
    let mut visited = vec![false; height * width];

    let mut contours = Vec::new();

    // Start a trace only where a horizontal run of foreground begins or ends —
    // Suzuki–Abe's border-start conditions. Those are exactly the cells with a
    // background neighbour due west or due east, which is what gives the walk a
    // backtrack on the outside of the region.
    //
    // A cell touching background only *diagonally* — the inside of a reflex
    // corner, or a cell catty-corner to a hole — is on the border but is not a
    // place to begin one: every one of its west/east neighbours is foreground, so
    // the sweep would set off into the interior and wander until the length guard
    // stopped it. Skipping them costs nothing, because a closed border always has
    // a leftmost cell in some row, and that cell is a run start.
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            if data[idx] == 0 || visited[idx] {
                continue;
            }

            let run_starts = x == 0 || data[idx - 1] == 0;
            let run_ends = x + 1 == width || data[idx + 1] == 0;

            if run_starts || run_ends {
                let contour = trace_contour(data, &mut visited, width, height, x, y);

                if !contour.is_empty() {
                    contours.push(contour);
                }
            } else {
                visited[idx] = true;
            }
        }
    }

    // Filter by mode
    let contours = match mode {
        ExtractMode::External => {
            // Keep only outermost contours (those not contained by others)
            filter_external_contours(contours)
        }
        ExtractMode::All | ExtractMode::Tree => {
            // Return all contours (Tree would add hierarchy info)
            contours
        }
    };

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

/// The 8-neighbourhood in clockwise order starting due east.
///
/// Clockwise *as drawn*, because y grows downward in image coordinates.
const DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),   // Right
    (1, 1),   // Down-Right
    (0, 1),   // Down
    (-1, 1),  // Down-Left
    (-1, 0),  // Left
    (-1, -1), // Up-Left
    (0, -1),  // Up
    (1, -1),  // Up-Right
];

/// Index into [`DIRECTIONS`] of `to` seen from `from`, for 8-adjacent cells.
fn direction_index(from: (i32, i32), to: (i32, i32)) -> Option<usize> {
    let delta = (to.0 - from.0, to.1 - from.1);
    DIRECTIONS.iter().position(|&d| d == delta)
}

/// Traces the boundary of a connected region with Moore-neighbour tracing.
///
/// The walk carries a **backtrack** cell — the background cell it arrived from —
/// and each step sweeps the 8-neighbourhood clockwise starting there. That is the
/// whole algorithm: sweeping from the backtrack is what keeps the walk on the
/// rim, because the first foreground cell clockwise of a known-outside cell is
/// necessarily the next one along the boundary.
///
/// Resuming the sweep anywhere else lets it cut diagonally into the interior and
/// the walk collapses. The previous implementation resumed five positions past
/// the direction it had just moved in, so from the top-left corner of a filled
/// square its first step was *inward*, along the diagonal; it then bounced
/// between four cells and returned to the start. A filled 80x80 square extracted
/// as 79 degenerate 2x2 contours — one per row, none of them its outline. The
/// symptom reached `metrics`, which filters detections whose "rasterized interior
/// is empty and are provably artifacts of the boundary tracer".
fn trace_contour(
    data: &[u8],
    visited: &mut [bool],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
) -> Contour {
    let foreground = |x: i32, y: i32| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < width
            && (y as usize) < height
            && data[y as usize * width + x as usize] > 0
    };

    let start = (start_x as i32, start_y as i32);

    // Which background neighbour the sweep starts from is not arbitrary, and this
    // is the part Suzuki–Abe gets right by construction. The raster scan reaches a
    // cell either where a foreground run *begins* — an outer border, whose outside
    // is to the west — or where one *ends*, which happens on the rim of a hole,
    // whose outside is to the east. Start on the wrong side and the first sweep
    // steps into the interior, where every neighbour is foreground and the walk
    // circles a few cells until the length guard stops it.
    //
    // Only a cell touching background diagonally has neither, and then any
    // background neighbour will do: it is a corner, so the rim leaves it the same
    // way whichever side is chosen.
    let west = !foreground(start.0 - 1, start.1);
    let east = !foreground(start.0 + 1, start.1);

    let backtrack = if west {
        Some(4) // West
    } else if east {
        Some(0) // East
    } else {
        (0..8).find(|&d| !foreground(start.0 + DIRECTIONS[d].0, start.1 + DIRECTIONS[d].1))
    };

    let Some(mut backtrack) = backtrack else {
        return Contour::new(vec![Point::new(start.0 as f64, start.1 as f64)]);
    };

    let mut current = start;
    let mut points = Vec::new();
    let mut first_move: Option<usize> = None;

    loop {
        let step = (1..=8).find_map(|i| {
            let d = (backtrack + i) % 8;
            let candidate = (current.0 + DIRECTIONS[d].0, current.1 + DIRECTIONS[d].1);
            foreground(candidate.0, candidate.1).then_some((candidate, d))
        });

        let Some((next, d)) = step else {
            // An isolated pixel: the region is this cell alone.
            points.push(Point::new(current.0 as f64, current.1 as f64));
            visited[current.1 as usize * width + current.0 as usize] = true;
            break;
        };

        // Jacob's criterion: the rim is closed once the walk stands on the start
        // again *about to repeat its first move*. Stopping merely on reaching the
        // start would cut the walk short at a cell the boundary legitimately
        // passes through twice, such as the waist of an hourglass; and comparing
        // the arriving backtrack would never match, because the initial one was
        // picked by scan order rather than by where the walk comes back from.
        if current == start {
            match first_move {
                None => first_move = Some(d),
                Some(first) if first == d => break,
                Some(_) => {}
            }
        }

        points.push(Point::new(current.0 as f64, current.1 as f64));
        visited[current.1 as usize * width + current.0 as usize] = true;

        // The cell examined just before `next` was background, so it is where the
        // sweep resumes from the new position.
        let previous = (
            current.0 + DIRECTIONS[(d + 7) % 8].0,
            current.1 + DIRECTIONS[(d + 7) % 8].1,
        );
        // Consecutive neighbourhood cells are themselves adjacent, so this always
        // resolves; the opposite direction is a safe fallback either way.
        backtrack = direction_index(next, previous).unwrap_or((d + 4) % 8);
        current = next;

        // A boundary cannot be longer than the image; bounds any pathological walk.
        if points.len() > width * height {
            break;
        }
    }

    Contour::new(points)
}

/// Filters to keep only external (outermost) contours.
fn filter_external_contours(contours: Vec<Contour>) -> Vec<Contour> {
    if contours.len() <= 1 {
        return contours;
    }

    let mut external = Vec::new();

    // Converted once per contour rather than once per (i, j) pair: this scan is
    // quadratic, and `to_geo` allocates.
    let polygons: Vec<geo::Polygon<f64>> = contours.iter().map(Contour::to_geo).collect();

    for (i, contour) in contours.iter().enumerate() {
        let is_contained = polygons.iter().enumerate().any(|(j, other)| {
            if i == j {
                return false;
            }
            // Check if contour's first point is inside other
            match contour.exterior.first() {
                Some(p) => super::predicates::position_in_polygon(other, p) > 0,
                None => false,
            }
        });

        if !is_contained {
            external.push(contour.clone());
        }
    }

    external
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

    /// Creates a test image with a white square on black background.
    fn create_test_image(width: usize, height: usize) -> ViewBuffer {
        let mut data = vec![0u8; width * height];

        // Draw a square from (20, 20) to (80, 80)
        for y in 20..80 {
            for x in 20..80 {
                data[y * width + x] = 255;
            }
        }

        ViewBuffer::from_vec_with_shape(data, vec![height, width, 1])
    }

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

    #[test]
    fn test_extract_single_contour() {
        let image = create_test_image(100, 100);
        let contours = extract_contours(&image, ExtractMode::External, ApproxMethod::None, None);

        // One filled square is one contour. Before the tracer was fixed this
        // returned 59 — one degenerate 2x2 walk per row of the square.
        assert_eq!(contours.len(), 1);

        // The square occupies pixels [20, 79]^2, so its rim is 60 + 60 + 58 + 58
        // cells and the trace visits each exactly once.
        let boundary = &contours[0].exterior;
        assert_eq!(boundary.len(), 2 * 60 + 2 * 58);

        // Every traced point is on the rim, and all four sides are reached.
        for p in boundary {
            let on_rim = (p.x == 20.0 || p.x == 79.0) && (20.0..=79.0).contains(&p.y)
                || (p.y == 20.0 || p.y == 79.0) && (20.0..=79.0).contains(&p.x);
            assert!(on_rim, "traced point ({}, {}) is not on the rim", p.x, p.y);
        }
        for corner in [(20.0, 20.0), (79.0, 20.0), (79.0, 79.0), (20.0, 79.0)] {
            assert!(
                boundary.iter().any(|p| (p.x, p.y) == corner),
                "corner {corner:?} missing from the trace"
            );
        }
    }

    #[test]
    fn test_extract_traces_the_rim_of_a_disc() {
        // A shape with no axis-aligned edges: the walk has to turn on diagonals.
        let image = mask_from(80, 80, |x, y| {
            let (dx, dy) = (x as f64 - 40.0, y as f64 - 40.0);
            dx * dx + dy * dy <= 25.0 * 25.0
        });
        let contours = extract_contours(&image, ExtractMode::External, ApproxMethod::None, None);

        assert_eq!(contours.len(), 1);
        // A radius-25 rim is ~2*pi*25 cells; a collapsed walk would be a handful.
        assert!(
            contours[0].exterior.len() > 100,
            "traced only {} points",
            contours[0].exterior.len()
        );
        for p in &contours[0].exterior {
            let (dx, dy) = (p.x - 40.0, p.y - 40.0);
            let r = (dx * dx + dy * dy).sqrt();
            assert!(
                (23.0..=25.5).contains(&r),
                "point at radius {r} is not on the rim"
            );
        }
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
        // Two blocks joined at a single cell. Plain "stop on returning to the
        // start" would end the walk at the waist and lose the second block;
        // Jacob's criterion carries it through.
        let image = mask_from(40, 40, |x, y| {
            (10..30).contains(&x) && (10..19).contains(&y)
                || (10..30).contains(&x) && (21..30).contains(&y)
                || (x == 19 && y == 19)
                || (x == 19 && y == 20)
        });
        let contours = extract_contours(&image, ExtractMode::All, ApproxMethod::None, None);

        let traced: Vec<_> = contours.iter().flat_map(|c| c.exterior.iter()).collect();
        assert!(
            traced.iter().any(|p| p.y < 19.0) && traced.iter().any(|p| p.y > 20.0),
            "the trace never reached both blocks"
        );
    }

    #[test]
    fn test_extract_isolated_pixel() {
        let image = mask_from(20, 20, |x, y| x == 5 && y == 5);
        let contours = extract_contours(&image, ExtractMode::External, ApproxMethod::None, None);

        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].exterior.len(), 1);
        assert_eq!(
            (contours[0].exterior[0].x, contours[0].exterior[0].y),
            (5.0, 5.0)
        );
    }

    #[test]
    fn test_extract_with_min_area() {
        let image = create_test_image(100, 100);

        // The traced rim runs through pixel centres, so it bounds [20, 79]^2 —
        // an area of 59*59 = 3481, not the 60*60 cells the mask paints.
        let kept = extract_contours(
            &image,
            ExtractMode::External,
            ApproxMethod::None,
            Some(3000.0),
        );
        assert_eq!(kept.len(), 1);

        let dropped = extract_contours(
            &image,
            ExtractMode::External,
            ApproxMethod::None,
            Some(5000.0),
        );
        assert!(dropped.is_empty());
    }
}
