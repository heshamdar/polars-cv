//! Histogram and quantization operations.
//!
//! This module provides operations for computing histograms and
//! quantizing arrays into discrete bins.

use crate::core::buffer::ViewBuffer;
use crate::core::cut::{AgainstCut, Cut};
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

use crate::mode::{Exec, FieldType, Literal, Mode, Param, TypeDesc, Wire};
use crate::naming::Bound;
use polars_cv_macros::{Ops, Resolve};

/// Output mode for histogram operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramOutput {
    /// Return bin counts as a 1D array.
    Counts,
    /// Return normalized histogram (sums to 1.0).
    Normalized,
    /// Return image with pixels replaced by bin indices.
    Quantized,
    /// Return bin edge values.
    Edges,
    /// Return bucket boundaries and statistics as a flattened array (to be parsed into structs).
    Buckets,
}

/// Interval closedness for histogram bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistogramClosed {
    /// Intervals are left-closed [a, b). (Last bin is [a, b]).
    #[default]
    Left,
    /// Intervals are right-closed (a, b]. (First bin is [a, b]).
    Right,
}

crate::naming::named_variants!(HistogramOutput: "Histogram output mode selection.\n\nControls what the histogram operation returns:\n- COUNTS: Bin counts as a 1D array\n- NORMALIZED: Histogram normalized to sum to 1.0\n- QUANTIZED: Input array with pixels replaced by bin indices\n- EDGES: Bin edge values\n- BUCKETS: List of bucket structs (lower_edge, upper_edge, count, normalized)" {
    "counts" => Counts,
    "normalized" => Normalized,
    "quantized" => Quantized,
    "edges" => Edges,
    "buckets" => Buckets,
});

crate::naming::named_variants!(HistogramClosed: "Interval inclusiveness for histogram binning.\n\n- LEFT: Intervals are left-closed ``[a, b)``.\n- RIGHT: Intervals are right-closed ``(a, b]``." {
    "left" => Left,
    "right" => Right,
});

/// Compute pixel value histogram.
///
/// Bins follow ``numpy.histogram``: a value outside the range or the edges
/// is in no bin (not counted; ``"quantized"`` gives it the index one past the
/// last bin), as is NaN. Infinite outer edges are open bounds, e.g.
/// ``bins=[-inf, 0, 10, inf]`` keeps every value.
///
/// Example:
///     >>> Pipeline().source("image_bytes").grayscale().histogram(bins=8)
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
#[op(name = "histogram", sample = {"bins": 8, "range": null, "closed": "left",
                                   "output": "counts"})]
pub struct HistogramOp<M: Mode = Exec> {
    /// Number of bins (default 256), a Polars expression for per-row dynamic
    /// bin count, or an explicit list of non-decreasing bin edges (the outer
    /// ones may be infinite).
    #[param(default = 256)]
    pub bins: Bins<M>,
    /// (min, max) tuple, finite. Auto-detected if None, widened to hold every
    /// value.
    pub range: Option<[M::V<f64>; 2]>,
    /// "left" or "right" interval inclusiveness (default "left").
    #[param(default = "left")]
    pub closed: M::L<HistogramClosed>,
    /// "buckets" (list of structs), "counts" (bin counts), "normalized" (sum to
    /// 1.0), "quantized" (pixel indices), "edges" (bin edges).
    #[param(default = "buckets")]
    pub output: M::L<HistogramOutput>,
}

/// How the bins are given: a count, or the edges themselves.
///
/// Two variants rather than a count plus optional edges, so a spec cannot carry
/// both and have one ignored. On the wire a list is `Edges`, anything else
/// (a number or a slot) is `Count`.
#[derive(Debug, Clone, PartialEq, Resolve)]
pub enum Bins<M: Mode = Exec> {
    /// This many equal-width bins over the range; may be per-row (the output
    /// is a list, so its length may vary by row).
    Count(M::V<u32>),
    /// Explicit, literal bin edges; the outer ones may be infinite.
    Edges(Vec<M::L<Bound>>),
}

impl serde::Serialize for Bins<Wire> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Bins::Count(n) => n.serialize(s),
            Bins::Edges(e) => e.serialize(s),
        }
    }
}

impl<'de> serde::Deserialize<'de> for Bins<Wire> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let value = serde_json::Value::deserialize(d)?;
        if value.is_array() {
            serde_json::from_value(value).map(Bins::Edges)
        } else {
            serde_json::from_value(value).map(Bins::Count)
        }
        .map_err(D::Error::custom)
    }
}

impl FieldType for Bins<Wire> {
    fn describe() -> TypeDesc {
        TypeDesc::OneOf {
            options: vec![
                <Param<u32> as FieldType>::describe(),
                <Vec<Literal<Bound>> as FieldType>::describe(),
            ],
        }
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        match self {
            Bins::Count(p) => p.visit_slots(f),
            Bins::Edges(e) => e.visit_slots(f),
        }
    }
}

impl<M: Mode> HistogramOp<M> {
    /// Every histogram parameter is independent; the counts are checked by
    /// `validate`.
    pub fn check(&self) -> Result<(), String> {
        Ok(())
    }

    /// The number of bins: the count, or one fewer than the edges.
    fn num_bins(&self) -> Sym<usize> {
        match &self.bins {
            Bins::Count(n) => match M::sym(n) {
                Sym::Known(n) => Sym::Known(n as usize),
                Sym::PerRow => Sym::PerRow,
            },
            Bins::Edges(edges) => Sym::Known(edges.len().saturating_sub(1)),
        }
    }

    /// How the output shape follows from the input: the bins' vector (or
    /// table), or the input itself for `quantized`. A per-row bin count leaves
    /// the length per-row.
    pub fn shape(&self) -> OpShape {
        let bins = self.num_bins();
        let plus_one = match bins {
            Sym::Known(n) => Sym::Known(n + 1),
            Sym::PerRow => Sym::PerRow,
        };
        match M::lit(&self.output) {
            HistogramOutput::Counts | HistogramOutput::Normalized => OpShape::Fixed(vec![bins]),
            HistogramOutput::Quantized => OpShape::Preserve,
            HistogramOutput::Edges => OpShape::Fixed(vec![plus_one]),
            HistogramOutput::Buckets => OpShape::Fixed(vec![bins, Sym::Known(4)]),
        }
    }

    /// The domain of this histogram's result: `Quantized` maps pixels in
    /// place (buffer); every other output mode yields a 1-D vector.
    pub fn output_domain(&self) -> Domain {
        match M::lit(&self.output) {
            HistogramOutput::Quantized => Domain::Buffer,
            _ => Domain::Vector,
        }
    }
}

impl HistogramOp {
    /// Create a new histogram operation.
    pub fn new(bins: usize) -> Self {
        Self {
            bins: Bins::Count(bins as u32),
            range: None,
            closed: HistogramClosed::Left,
            output: HistogramOutput::Counts,
        }
    }

    /// Set the value range.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.range = Some([min, max]);
        self
    }

    /// Set explicit edges.
    pub fn with_edges(mut self, edges: Vec<f64>) -> Self {
        self.bins = Bins::Edges(edges.into_iter().map(Bound).collect());
        self
    }

    /// Set interval closedness.
    pub fn with_closed(mut self, closed: HistogramClosed) -> Self {
        self.closed = closed;
        self
    }

    /// Set the output mode.
    pub fn with_output(mut self, output: HistogramOutput) -> Self {
        self.output = output;
        self
    }

    /// The explicit edges, when the bins are given that way.
    fn explicit_edges(&self) -> Option<Vec<f64>> {
        match &self.bins {
            Bins::Edges(edges) => Some(edges.iter().map(|b| b.0).collect()),
            Bins::Count(_) => None,
        }
    }

    /// The equal-width bin count (edges: one fewer than their number).
    fn bin_count(&self) -> usize {
        match self.num_bins() {
            Sym::Known(n) => n,
            Sym::PerRow => unreachable!("an executed op's values are known"),
        }
    }

    /// The `(min, max)` range, when given.
    fn value_range(&self) -> Option<(f64, f64)> {
        self.range.map(|[min, max]| (min, max))
    }

    /// Execute the histogram operation.
    ///
    /// Membership follows numpy: left-closed bins are `[e_i, e_i+1)` with the
    /// last closed, right-closed `(e_i, e_i+1]` with the first closed, and a
    /// tie of edges resolves as `searchsorted` does. A value in no bin —
    /// outside the range or the edges, or NaN — is not counted, and its
    /// quantized index is one past the last bin; infinite outer edges are
    /// open bounds that keep every value. A pixel is compared with an edge
    /// exactly ([`Cut`]). A range that is not finite — supplied, or
    /// auto-detected over NaN or infinity — is an error, there being no
    /// equal-width bins to make.
    pub fn execute(&self, buffer: &ViewBuffer) -> Result<ViewBuffer, String> {
        let contig = buffer.to_contiguous();
        crate::core::dtype::with_dtype!(buffer.dtype(), T => self.execute_typed::<T>(&contig))
    }

    /// The equal-width edges over `[lo, hi]`, the last exactly `hi` (as
    /// `np.linspace`), so the top of the range is not lost to rounding.
    fn uniform_edges(&self, lo: f64, hi: f64) -> Vec<f64> {
        let bins = self.bin_count();
        let width = (hi - lo) / bins as f64;
        let mut e: Vec<f64> = (0..bins).map(|i| lo + i as f64 * width).collect();
        e.push(hi);
        e
    }

    /// The edges: given, or equal-width over the given or detected range.
    fn edges_for<T>(&self, data: &[T]) -> Result<Vec<f64>, String>
    where
        T: AgainstCut + num_traits::NumCast + PartialOrd,
    {
        if let Some(e) = self.explicit_edges() {
            return Ok(e);
        }
        let to_f64 = |x: T| -> f64 { num_traits::NumCast::from(x).unwrap_or(f64::NAN) };
        let (lo, hi) = match self.value_range() {
            Some((min, max)) if min.is_finite() && max.is_finite() => {
                if min > max {
                    return Err(format!("range min {min} must not exceed max {max}"));
                }
                (min, max)
            }
            Some((min, max)) => {
                return Err(format!("supplied range of [{min}, {max}] is not finite"));
            }
            None => {
                // The extremes as elements, so the edges can contain them
                // exactly; a NaN anywhere makes the range NaN.
                let mut it = data.iter().copied();
                let Some(first) = it.next() else {
                    return Ok(self.uniform_edges(-0.5, 0.5));
                };
                let (mut min, mut max) = (first, first);
                let mut nan = first.is_nan();
                for x in it {
                    nan |= x.is_nan();
                    if x < min {
                        min = x;
                    }
                    if x > max {
                        max = x;
                    }
                }
                let (mut lo, mut hi) = (to_f64(min), to_f64(max));
                if nan || !(lo.is_finite() && hi.is_finite()) {
                    let (lo, hi) = if nan { (f64::NAN, f64::NAN) } else { (lo, hi) };
                    return Err(format!(
                        "autodetected range of [{lo}, {hi}] is not finite; \
                         pass `range=` or explicit bin edges"
                    ));
                }
                // An extreme f64 rounded past widens outward until it is
                // inside, so a detected range drops nothing.
                while !min.reaches(&Cut::of(lo)) {
                    lo = lo.next_down();
                }
                while max.above(&Cut::of(hi)) {
                    hi = hi.next_up();
                }
                (lo, hi)
            }
        };
        // A zero-width range widens by 0.5 each way, as numpy's.
        if hi - lo == 0.0 {
            return Ok(self.uniform_edges(lo - 0.5, hi + 0.5));
        }
        Ok(self.uniform_edges(lo, hi))
    }

    fn execute_typed<T>(&self, buffer: &ViewBuffer) -> Result<ViewBuffer, String>
    where
        T: Copy + AgainstCut + num_traits::NumCast + PartialOrd + ViewType + 'static,
    {
        let data = buffer.as_slice::<T>();
        let shape = buffer.shape();
        let edges = self.edges_for(data)?;

        let num_bins = edges.len().saturating_sub(1);
        if num_bins == 0 {
            // Edge case: no bins
            return Ok(match self.output {
                HistogramOutput::Counts => {
                    ViewBuffer::from_vec_with_shape(Vec::<u64>::new(), vec![0])
                }
                HistogramOutput::Normalized => {
                    ViewBuffer::from_vec_with_shape(Vec::<f64>::new(), vec![0])
                }
                HistogramOutput::Quantized => {
                    ViewBuffer::from_vec_with_shape(vec![0u32; data.len()], shape.to_vec())
                }
                HistogramOutput::Edges => {
                    let len = edges.len();
                    ViewBuffer::from_vec_with_shape(edges, vec![len])
                }
                HistogramOutput::Buckets => {
                    ViewBuffer::from_vec_with_shape(Vec::<f64>::new(), vec![0, 4])
                }
            });
        }

        let mut counts = vec![0u64; num_bins];
        let mut quantized = if self.output == HistogramOutput::Quantized {
            Some(Vec::with_capacity(data.len()))
        } else {
            None
        };
        let bin = Binner::new(&edges, self.closed, self.explicit_edges().is_none());
        for &x in data {
            let b = bin.of(x);
            if let Some(b) = b {
                counts[b] += 1;
            }
            if let Some(ref mut q) = quantized {
                // In no bin: one past the last, as `np.digitize` puts it.
                q.push(b.unwrap_or(num_bins) as u32);
            }
        }

        // What was counted: NaN is in no bin, so the shares sum to 1 over the rest.
        let total = counts.iter().sum::<u64>() as f64;
        Ok(match self.output {
            HistogramOutput::Counts => ViewBuffer::from_vec_with_shape(counts, vec![num_bins]),
            HistogramOutput::Normalized => {
                let normalized: Vec<f64> = counts
                    .iter()
                    .map(|&c| if total > 0.0 { c as f64 / total } else { 0.0 })
                    .collect();
                ViewBuffer::from_vec_with_shape(normalized, vec![num_bins])
            }
            HistogramOutput::Quantized => {
                ViewBuffer::from_vec_with_shape(quantized.unwrap(), shape.to_vec())
            }
            HistogramOutput::Edges => {
                let len = edges.len();
                ViewBuffer::from_vec_with_shape(edges, vec![len])
            }
            HistogramOutput::Buckets => {
                let mut buckets = Vec::with_capacity(num_bins * 4);
                for i in 0..num_bins {
                    buckets.push(edges[i]);
                    buckets.push(edges[i + 1]);
                    buckets.push(counts[i] as f64);
                    buckets.push(if total > 0.0 {
                        counts[i] as f64 / total
                    } else {
                        0.0
                    });
                }
                ViewBuffer::from_vec_with_shape(buckets, vec![num_bins, 4])
            }
        })
    }
}

/// Which bin a value is in, by numpy's rule, every comparison exact.
struct Binner {
    cuts: Vec<Cut>,
    closed: HistogramClosed,
    /// Equal-width edges: the bin is guessed arithmetically, then settled by
    /// the edges (a value on an edge can compute a bin low).
    uniform: Option<(f64, f64)>,
}

impl Binner {
    fn new(edges: &[f64], closed: HistogramClosed, uniform: bool) -> Self {
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
    fn of<T: AgainstCut + num_traits::NumCast>(&self, x: T) -> Option<usize> {
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

impl<M: Mode> Op for HistogramOp<M> {
    fn name(&self) -> &'static str {
        "Histogram"
    }

    fn shape(&self) -> OpShape {
        HistogramOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::RequiresContiguous
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false // A histogram is not an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // Binning aggregates over all pixels.
        SpatialDependency::Global
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        None
    }

    fn validate(
        &self,
        _input_shapes: &[&[crate::ops::Dim]],
        _input_dtypes: &[crate::PlannedDType],
    ) -> Result<(), ValidationError> {
        // A per-row bin count is checked per row.
        if let Bins::Edges(edges) = &self.bins {
            if edges.len() < 2 {
                return Err(ValidationError::InvalidParameter {
                    param: "edges".to_string(),
                    reason: "edges must contain at least 2 values".to_string(),
                });
            }
            // numpy's rule; a NaN edge orders against nothing, so it fails too.
            let edges: Vec<f64> = edges.iter().map(|e| M::lit(e).0).collect();
            if edges
                .windows(2)
                .any(|w| w[0].partial_cmp(&w[1]).is_none_or(|o| o.is_gt()))
            {
                return Err(ValidationError::InvalidParameter {
                    param: "bins".to_string(),
                    reason: format!("bin edges must increase monotonically, got {edges:?}"),
                });
            }
        } else if self.num_bins() == Sym::Known(0) {
            return Err(ValidationError::InvalidParameter {
                param: "bins".to_string(),
                reason: "bins must be > 0".to_string(),
            });
        }
        Ok(())
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Numeric
    }

    fn working_dtype(&self) -> Option<DType> {
        None
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match M::lit(&self.output) {
            HistogramOutput::Counts => OutputDTypeRule::ForceU64,
            HistogramOutput::Normalized | HistogramOutput::Edges | HistogramOutput::Buckets => {
                OutputDTypeRule::ForceF64
            }
            HistogramOutput::Quantized => OutputDTypeRule::ForceU32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value lands in the bin its own `edges` output says it belongs
    /// to: `[e_i, e_i+1)` (left-closed, the last bin closed) or
    /// `(e_i, e_i+1]` (right-closed, the first bin closed). The bin was
    /// computed as `floor((x - lo) / width)`, which for a value exactly on an
    /// edge can land one bin low: -7142.857142857143 equals edge 1 of
    /// [-10000, 10000] in 7 bins but was counted in bin 0.
    #[test]
    fn a_value_on_an_edge_is_in_the_bin_the_edges_give() {
        let gradient: Vec<f64> = (0..8)
            .map(|i| -10000.0 + i as f64 * 20000.0 / 7.0)
            .collect();
        let mut values = gradient.clone();
        values.extend((0..200).map(|i| ((i * 7919) % 1000) as f64 * 0.37 - 150.0));
        for closed in [HistogramClosed::Left, HistogramClosed::Right] {
            for bins in [3usize, 7, 10] {
                for data in [&gradient, &values] {
                    let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![data.len()]);
                    let op = HistogramOp::new(bins).with_closed(closed);
                    let edges = op
                        .clone()
                        .with_output(HistogramOutput::Edges)
                        .execute(&buf)
                        .unwrap();
                    let edges = edges.as_slice::<f64>().to_vec();
                    let q = op
                        .with_output(HistogramOutput::Quantized)
                        .execute(&buf)
                        .unwrap();
                    for (&x, &b) in data.iter().zip(q.as_slice::<u32>()) {
                        let (b, last) = (b as usize, bins - 1);
                        let (lo, hi) = (edges[b], edges[b + 1]);
                        let inside = match closed {
                            HistogramClosed::Left => lo <= x && (x < hi || (b == last && x <= hi)),
                            HistogramClosed::Right => (lo < x || (b == 0 && x >= lo)) && x <= hi,
                        };
                        assert!(
                            inside,
                            "{closed:?} {bins} bins: {x} in bin {b} [{lo}, {hi}]"
                        );
                    }
                }
            }
        }
    }

    fn counts(op: &HistogramOp, buf: &ViewBuffer) -> Vec<u64> {
        op.execute(buf).unwrap().as_slice::<u64>().to_vec()
    }

    fn quantized(op: &HistogramOp, buf: &ViewBuffer) -> Vec<u32> {
        let q = op.clone().with_output(HistogramOutput::Quantized);
        q.execute(buf).unwrap().as_slice::<u32>().to_vec()
    }

    /// A value outside the range or the edges is in no bin, as in numpy: not
    /// counted, and quantized one past the last bin (as NaN is). It used to
    /// be clamped into the end bin, so `range=(0, 4)` counted 9 in bin 3.
    #[test]
    fn a_value_outside_the_bins_is_in_no_bin() {
        let buf = ViewBuffer::from_vec_with_shape(vec![-1.0, 0.0, 1.5, 4.0, 9.0], vec![5]);
        let ranged = HistogramOp::new(4).with_range(0.0, 4.0);
        let edged = HistogramOp::new(4).with_edges(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        for closed in [HistogramClosed::Left, HistogramClosed::Right] {
            for op in [&ranged, &edged] {
                let op = op.clone().with_closed(closed);
                assert_eq!(counts(&op, &buf), [1, 1, 0, 1], "{op:?}");
                assert_eq!(quantized(&op, &buf), [4, 0, 1, 3, 4], "{op:?}");
                let norm = op.clone().with_output(HistogramOutput::Normalized);
                let norm = norm.execute(&buf).unwrap();
                assert_eq!(
                    norm.as_slice::<f64>(),
                    &[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0]
                );
            }
        }
    }

    /// Tied edges follow numpy's `searchsorted`: left-closed, a value goes to
    /// the last bin whose lower edge it reaches (the zero-width `[1, 1)` is
    /// empty); right-closed, to the first bin whose upper edge reaches it.
    /// A binary search over the ties used to pick any of them.
    #[test]
    fn tied_edges_follow_numpy() {
        let buf = ViewBuffer::from_vec_with_shape(vec![0.5, 1.0, 1.0, 1.5, 2.0], vec![5]);
        let op = HistogramOp::new(1).with_edges(vec![0.0, 1.0, 1.0, 2.0]);
        // np.histogram([0.5, 1, 1, 1.5, 2], [0, 1, 1, 2]) -> [1, 0, 4]
        assert_eq!(counts(&op, &buf), [1, 0, 4]);
        let right = op.with_closed(HistogramClosed::Right);
        assert_eq!(counts(&right, &buf), [3, 0, 2]);
        for edges in [vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]] {
            let op = HistogramOp::new(1).with_edges(edges.clone());
            // np.histogram(..., [0, 0, 1]) -> [0, 5]; (..., [0, 1, 1]) -> [3, 2]
            let ref_left = if edges[1] == 0.0 { [0, 5] } else { [3, 2] };
            let got = counts(
                &op,
                &ViewBuffer::from_vec_with_shape(vec![0.0, 0.5, 1.0, 1.0, 0.25], vec![5]),
            );
            assert_eq!(got, ref_left, "{edges:?}");
        }
    }

    /// Infinite outer edges are open bounds (as polars' `hist` breaks): with
    /// out-of-range values dropped, they are how every value is kept.
    #[test]
    fn infinite_outer_edges_are_open_bounds() {
        let data = vec![
            f64::NEG_INFINITY,
            -1e300,
            -1.0,
            0.0,
            5.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let buf = ViewBuffer::from_vec_with_shape(data, vec![7]);
        let op = HistogramOp::new(1).with_edges(vec![f64::NEG_INFINITY, 0.0, f64::INFINITY]);
        assert_eq!(counts(&op, &buf), [3, 3]);
        assert_eq!(quantized(&op, &buf), [0, 0, 0, 1, 1, 1, 2]);
        let right = op.clone().with_closed(HistogramClosed::Right);
        assert_eq!(counts(&right, &buf), [4, 2]);
        let edges = op
            .with_output(HistogramOutput::Edges)
            .execute(&buf)
            .unwrap();
        assert_eq!(
            edges.as_slice::<f64>(),
            &[f64::NEG_INFINITY, 0.0, f64::INFINITY]
        );
        // One open side: the one bin is closed, so it holds 0.
        let below = HistogramOp::new(1).with_edges(vec![f64::NEG_INFINITY, 0.0]);
        assert_eq!(counts(&below, &buf), [4]);
    }

    /// An integer pixel meets a float edge exactly: 2^53 + 3 is below the
    /// edge 2^53 + 4, though it rounds to it in f64 (F7, the threshold's
    /// exact cut reused).
    #[test]
    fn an_integer_pixel_meets_an_edge_exactly() {
        let e = 9_007_199_254_740_996.0; // 2^53 + 4
        let wide = [9_007_199_254_740_995u64, 9_007_199_254_740_996];
        let buf = ViewBuffer::from_vec_with_shape(wide.to_vec(), vec![2]);
        let op = HistogramOp::new(1).with_edges(vec![0.0, e, 2.0 * e]);
        assert_eq!(counts(&op, &buf), [1, 1]);
        let right = op.with_closed(HistogramClosed::Right);
        assert_eq!(counts(&right, &buf), [2, 0]);
        let signed = ViewBuffer::from_vec_with_shape(vec![-9_007_199_254_740_995i64], vec![1]);
        let op = HistogramOp::new(1).with_edges(vec![-e, 0.0]);
        assert_eq!(counts(&op, &signed), [1], "-2^53 - 3 is above -2^53 - 4");
    }

    /// An auto-detected range holds every pixel, though the extremes round
    /// to f64: the edges widen outward to contain them, so none is dropped.
    #[test]
    fn an_auto_range_keeps_every_integer_pixel() {
        let wide = vec![
            9_007_199_254_740_995u64,
            1 << 60,
            u64::MAX,
            9_007_199_254_740_997,
        ];
        let buf = ViewBuffer::from_vec_with_shape(wide.clone(), vec![4]);
        for bins in [1usize, 3, 7] {
            for closed in [HistogramClosed::Left, HistogramClosed::Right] {
                let op = HistogramOp::new(bins).with_closed(closed);
                assert_eq!(
                    counts(&op, &buf).iter().sum::<u64>(),
                    4,
                    "{bins} {closed:?}"
                );
            }
        }
        let signed = vec![i64::MIN, -9_007_199_254_740_995, i64::MAX];
        let buf = ViewBuffer::from_vec_with_shape(signed, vec![3]);
        assert_eq!(counts(&HistogramOp::new(5), &buf).iter().sum::<u64>(), 3);
    }

    /// NaN follows numpy (`np.histogram`, `np.digitize`): it is in no bin, so
    /// it is not counted (and `normalized` sums to 1 over what was), and its
    /// quantized index is one past the last bin. It used to land in bin 0
    /// with equal-width bins and panic the engine with explicit edges.
    #[test]
    fn nan_is_in_no_bin() {
        let data = vec![f64::NAN, 0.5, 1.5, f64::NAN, 3.5];
        let buf = ViewBuffer::from_vec_with_shape(data, vec![5]);
        let ranged = HistogramOp::new(4).with_range(0.0, 4.0);
        let edged = HistogramOp::new(4).with_edges(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        for op in [ranged, edged] {
            let counts = op.execute(&buf).unwrap();
            assert_eq!(counts.as_slice::<u64>(), &[1, 1, 0, 1], "{op:?}");
            let norm = op
                .clone()
                .with_output(HistogramOutput::Normalized)
                .execute(&buf)
                .unwrap();
            assert!((norm.as_slice::<f64>().iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let q = op
                .clone()
                .with_output(HistogramOutput::Quantized)
                .execute(&buf)
                .unwrap();
            assert_eq!(q.as_slice::<u32>(), &[4, 0, 1, 4, 3], "{op:?}");
        }
    }

    /// An auto-detected range over NaN or infinity, or a supplied range that
    /// is not finite, is an error, as in numpy ("autodetected range of
    /// [nan, nan] is not finite"): there are no equal-width bins to make.
    #[test]
    fn a_range_that_is_not_finite_is_an_error() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let buf = ViewBuffer::from_vec_with_shape(vec![1.0, bad, 2.0], vec![3]);
            for output in [HistogramOutput::Counts, HistogramOutput::Edges] {
                let err = HistogramOp::new(4).with_output(output).execute(&buf);
                assert!(
                    err.as_ref().is_err_and(|e| e.contains("not finite")),
                    "{bad} {output:?}: {err:?}"
                );
            }
            let finite = ViewBuffer::from_vec_with_shape(vec![1.0f64, 2.0], vec![2]);
            let err = HistogramOp::new(4).with_range(0.0, bad).execute(&finite);
            assert!(
                err.is_err_and(|e| e.contains("not finite")),
                "range [0, {bad}]"
            );
        }
    }

    /// Explicit edges must increase monotonically (numpy's rule); a NaN
    /// edge orders against nothing and is refused with them.
    #[test]
    fn edges_must_increase_monotonically() {
        for edges in [vec![0.0, 2.0, 1.0], vec![0.0, f64::NAN, 1.0]] {
            let op = HistogramOp::new(1).with_edges(edges.clone());
            assert!(op.validate(&[], &[]).is_err(), "{edges:?}");
        }
        let op = HistogramOp::new(1).with_edges(vec![0.0, 1.0, 1.0, 2.0]);
        assert!(op.validate(&[], &[]).is_ok());
    }

    #[test]
    fn test_histogram_counts() {
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![8]);

        let op = HistogramOp::new(4).with_range(0.0, 8.0);
        let result = op.execute(&buffer).unwrap();

        let counts = result.as_slice::<u64>();
        assert_eq!(counts.len(), 4);
        // Each bin should have 2 values: [0,1], [2,3], [4,5], [6,7]
        assert_eq!(counts, &[2, 2, 2, 2]);
    }

    #[test]
    fn test_histogram_normalized() {
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![8]);

        let op = HistogramOp::new(4)
            .with_range(0.0, 8.0)
            .with_output(HistogramOutput::Normalized);
        let result = op.execute(&buffer).unwrap();

        let normalized = result.as_slice::<f64>();
        let sum: f64 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_quantized() {
        let data = vec![0u8, 128, 255];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![3]);

        let op = HistogramOp::new(4)
            .with_range(0.0, 256.0)
            .with_output(HistogramOutput::Quantized);
        let result = op.execute(&buffer).unwrap();

        let quantized = result.as_slice::<u32>();
        assert_eq!(quantized.len(), 3);
        assert_eq!(quantized[0], 0); // 0 -> bin 0
        assert_eq!(quantized[1], 2); // 128 -> bin 2
        assert_eq!(quantized[2], 3); // 255 -> bin 3 (clamped)
    }
}
