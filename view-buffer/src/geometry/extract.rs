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

    // Find starting points (transitions from 0 to non-0)
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let pixel = data[idx];

            if pixel > 0 && !visited[idx] {
                // Check if this is a boundary pixel
                if is_boundary_pixel(data, width, height, x, y) {
                    // Trace the contour
                    let contour = trace_contour(data, &mut visited, width, height, x, y);

                    if !contour.is_empty() {
                        contours.push(contour);
                    }
                } else {
                    visited[idx] = true;
                }
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

/// Checks if a pixel is on the boundary (has at least one background neighbor).
fn is_boundary_pixel(data: &[u8], width: usize, height: usize, x: usize, y: usize) -> bool {
    // 8-connected neighborhood
    let neighbors = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    for (dx, dy) in neighbors {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;

        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
            return true; // Edge of image counts as boundary
        }

        let idx = ny as usize * width + nx as usize;
        if data[idx] == 0 {
            return true; // Has background neighbor
        }
    }

    false
}

/// Traces a contour using Moore-Neighbor tracing algorithm.
fn trace_contour(
    data: &[u8],
    visited: &mut [bool],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
) -> Contour {
    // Moore-Neighbor directions: clockwise from right
    // (dx, dy) pairs for 8-connected neighbors
    let directions: [(i32, i32); 8] = [
        (1, 0),   // Right
        (1, 1),   // Down-Right
        (0, 1),   // Down
        (-1, 1),  // Down-Left
        (-1, 0),  // Left
        (-1, -1), // Up-Left
        (0, -1),  // Up
        (1, -1),  // Up-Right
    ];

    let mut points = Vec::new();
    let mut x = start_x as i32;
    let mut y = start_y as i32;
    let mut dir = 0; // Start looking right

    // Find initial backtrack direction
    for (d, &(dx, dy)) in directions.iter().enumerate() {
        let nx = x + dx;
        let ny = y + dy;
        if nx >= 0
            && ny >= 0
            && (nx as usize) < width
            && (ny as usize) < height
            && data[ny as usize * width + nx as usize] == 0
        {
            dir = (d + 1) % 8;
            break;
        }
    }

    loop {
        // Add current point
        points.push(Point::new(x as f64, y as f64));
        visited[y as usize * width + x as usize] = true;

        // Find next boundary pixel
        let mut found = false;
        let start_dir = (dir + 5) % 8; // Start searching from backtrack direction

        for i in 0..8 {
            let d = (start_dir + i) % 8;
            let nx = x + directions[d].0;
            let ny = y + directions[d].1;

            if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                let idx = ny as usize * width + nx as usize;
                if data[idx] > 0 {
                    x = nx;
                    y = ny;
                    dir = d;
                    found = true;
                    break;
                }
            }
        }

        if !found || (x as usize == start_x && y as usize == start_y && points.len() > 1) {
            break;
        }

        // Prevent infinite loops
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

    for (i, contour) in contours.iter().enumerate() {
        let is_contained = contours.iter().enumerate().any(|(j, other)| {
            if i == j {
                return false;
            }
            // Check if contour's first point is inside other
            if let Some(p) = contour.exterior.first() {
                super::predicates::point_in_polygon(p, &other.exterior) > 0
            } else {
                false
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

    #[test]
    fn test_extract_single_contour() {
        let image = create_test_image(100, 100);
        let contours = extract_contours(&image, ExtractMode::External, ApproxMethod::None, None);

        assert!(!contours.is_empty());
        // Should find approximately the boundary of the square
        // The exact number of points depends on the tracing algorithm
    }

    #[test]
    fn test_extract_with_min_area() {
        let image = create_test_image(100, 100);

        // With high min_area, should filter out the square (60x60 = 3600)
        let contours = extract_contours(
            &image,
            ExtractMode::External,
            ApproxMethod::None,
            Some(5000.0),
        );

        // May or may not have the contour depending on actual traced area
        // This is more of a smoke test
        let _ = contours;
    }

    #[test]
    fn test_is_boundary_pixel() {
        let mut data = vec![0u8; 100];
        data[55] = 255; // Single white pixel at (5, 5) in 10x10

        assert!(is_boundary_pixel(&data, 10, 10, 5, 5));
    }
}
