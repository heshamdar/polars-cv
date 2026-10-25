//! Histogram binning: which bin a value is in, every comparison exact.

use crate::core::cut::{AgainstCut, Cut};
use crate::ops::histogram::HistogramClosed;

/// Which bin a value is in, by numpy's rule, every comparison exact.
pub(crate) struct Binner {
    cuts: Vec<Cut>,
    closed: HistogramClosed,
    /// Equal-width edges: the bin is guessed arithmetically, then settled by
    /// the edges (a value on an edge can compute a bin low).
    uniform: Option<(f64, f64)>,
}

impl Binner {
    pub(crate) fn new(edges: &[f64], closed: HistogramClosed, uniform: bool) -> Self {
        let n = edges.len() - 1;
        Binner {
            cuts: edges.iter().map(|&e| Cut::of(e)).collect(),
            closed,
            uniform: uniform.then(|| (edges[0], (edges[n] - edges[0]) / n as f64)),
        }
    }

    /// Whether `x` is on the far side of interior edge `i`: it reaches it
    /// (left-closed) or exceeds it (right-closed). Monotone in `i`, so the
    /// bin is how many interior edges `x` passes -- `searchsorted`.
    #[inline(always)]
    fn passes<T: AgainstCut>(&self, x: T, i: usize) -> bool {
        match self.closed {
            HistogramClosed::Left => x.reaches(&self.cuts[i]),
            HistogramClosed::Right => x.above(&self.cuts[i]),
        }
    }

    #[inline(always)]
    pub(crate) fn of<T: AgainstCut + num_traits::NumCast>(&self, x: T) -> Option<usize> {
        let n = self.cuts.len() - 1;
        // Inside [e_0, e_n] (NaN is inside nothing).
        if !x.reaches(&self.cuts[0]) || x.above(&self.cuts[n]) {
            return None;
        }
        let last = n - 1;
        let mut b = match self.uniform {
            Some((lo, width)) if width > 0.0 => {
                let xf: f64 = num_traits::NumCast::from(x).unwrap_or(lo);
                let guess = ((xf - lo) / width).floor();
                if guess.is_nan() || guess < 0.0 {
                    0
                } else {
                    (guess as usize).min(last)
                }
            }
            _ => {
                // Interior edges 1..n passed, by binary search.
                let interior = &self.cuts[1..n];
                return Some(interior.partition_point(|c| match self.closed {
                    HistogramClosed::Left => x.reaches(c),
                    HistogramClosed::Right => x.above(c),
                }));
            }
        };
        while b < last && self.passes(x, b + 1) {
            b += 1;
        }
        while b > 0 && !self.passes(x, b) {
            b -= 1;
        }
        Some(b)
    }
}
