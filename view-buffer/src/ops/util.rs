//! Shared utility functions for ops modules.

use std::cmp::Ordering;

/// Whether `b` displaces `a` as the running extreme in direction `want`
/// (`Greater` for a maximum, `Less` for a minimum).
///
/// **The one NaN ordering rule** (numpy's `maximum`/`minimum`/`max`/`argmax`):
/// a NaN — the one value unordered against itself — is absorbing, the first
/// met being kept (so an arg-reduction reports its index); otherwise only a
/// strictly better value displaces, so a tie keeps the incumbent. Every op
/// that orders values reads it, through [`maximum`]/[`minimum`] or directly,
/// rather than a comparison of its own: each used to decide with `>`/`<` or
/// `f32::max`, and the answer depended on which side the NaN sat.
#[inline(always)]
pub(crate) fn displaces<T: PartialOrd>(a: &T, b: &T, want: Ordering) -> bool {
    #[allow(clippy::eq_op)]
    let is_nan = |x: &T| x.partial_cmp(x).is_none();
    !is_nan(a) && (is_nan(b) || b.partial_cmp(a) == Some(want))
}

/// The larger of `a` and `b`, NaN if either is ([`displaces`]).
#[inline(always)]
pub(crate) fn maximum<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if displaces(&a, &b, Ordering::Greater) {
        b
    } else {
        a
    }
}

/// The smaller of `a` and `b`, NaN if either is ([`displaces`]).
#[inline(always)]
pub(crate) fn minimum<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if displaces(&a, &b, Ordering::Less) {
        b
    } else {
        a
    }
}

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
