//! Shared utility functions for ops modules.

/// Convert a linear index to multi-dimensional coordinates.
pub fn linear_to_coords(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut remaining = index;

    for i in (0..shape.len()).rev() {
        coords[i] = remaining % shape[i];
        remaining /= shape[i];
    }

    coords
}

/// Convert multi-dimensional coordinates to a linear index using strides.
pub fn coords_to_linear(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
}
