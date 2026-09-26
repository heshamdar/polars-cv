//! Histogram and quantization operations.
//!
//! This module provides operations for computing histograms and
//! quantizing arrays into discrete bins.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

use crate::mode::{Exec, FieldType, Literal, Mode, Param, TypeDesc, Wire};
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
/// Example:
///     >>> Pipeline().source("image_bytes").grayscale().histogram(bins=8)
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
#[op(name = "histogram", sample = {"bins": 8, "range": null, "closed": "left",
                                   "output": "counts"})]
pub struct HistogramOp<M: Mode = Exec> {
    /// Number of bins (default 256), a Polars expression for per-row dynamic
    /// bin count, or an explicit list of bin edges.
    #[param(default = 256)]
    pub bins: Bins<M>,
    /// (min, max) tuple. Auto-detected if None.
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
    /// Explicit, literal bin edges.
    Edges(Vec<M::L<f64>>),
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
                <Vec<Literal<f64>> as FieldType>::describe(),
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
        self.bins = Bins::Edges(edges);
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
    fn explicit_edges(&self) -> Option<&Vec<f64>> {
        match &self.bins {
            Bins::Edges(edges) => Some(edges),
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
    pub fn execute(&self, buffer: &ViewBuffer) -> ViewBuffer {
        let contig = buffer.to_contiguous();

        match buffer.dtype() {
            DType::U8 => self.execute_typed::<u8>(&contig),
            DType::I8 => self.execute_typed::<i8>(&contig),
            DType::U16 => self.execute_typed::<u16>(&contig),
            DType::I16 => self.execute_typed::<i16>(&contig),
            DType::U32 => self.execute_typed::<u32>(&contig),
            DType::I32 => self.execute_typed::<i32>(&contig),
            DType::U64 => self.execute_typed::<u64>(&contig),
            DType::I64 => self.execute_typed::<i64>(&contig),
            DType::F32 => self.execute_typed::<f32>(&contig),
            DType::F64 => self.execute_typed::<f64>(&contig),
        }
    }

    fn execute_typed<T>(&self, buffer: &ViewBuffer) -> ViewBuffer
    where
        T: Copy + num_traits::NumCast + PartialOrd + ViewType + 'static,
    {
        let data = buffer.as_slice::<T>();
        let shape = buffer.shape();

        // Determine edges
        let edges = if let Some(e) = self.explicit_edges() {
            e.clone()
        } else {
            let (mut min_val, mut max_val) = match self.value_range() {
                Some((min, max)) => (min, max),
                None => {
                    // Auto-detect from data
                    let (dmin, dmax) = data.iter().fold((f64::MAX, f64::MIN), |(min, max), &x| {
                        let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                        (min.min(xf), max.max(xf))
                    });
                    (dmin, dmax)
                }
            };

            // Fix uniform range bug by extending it similarly to numpy
            if (max_val - min_val).abs() < f64::EPSILON {
                min_val -= 0.5;
                max_val += 0.5;
            }

            let bins = self.bin_count();
            let bin_width = (max_val - min_val) / bins as f64;
            let mut e = Vec::with_capacity(bins + 1);
            for i in 0..=bins {
                e.push(min_val + i as f64 * bin_width);
            }
            e
        };

        let num_bins = edges.len().saturating_sub(1);
        if num_bins == 0 {
            // Edge case: no bins
            return match self.output {
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
            };
        }

        let mut counts = vec![0u64; num_bins];
        let mut quantized = if self.output == HistogramOutput::Quantized {
            Some(Vec::with_capacity(data.len()))
        } else {
            None
        };

        let is_uniform = self.explicit_edges().is_none();

        for &x in data {
            let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);

            // Find bin
            let bin_idx = if is_uniform {
                let bin_width = (edges.last().unwrap() - edges[0]) / num_bins as f64;
                if bin_width == 0.0 {
                    0
                } else {
                    let mut b = ((xf - edges[0]) / bin_width).floor() as isize;

                    // Handle bounds based on closed strategy
                    match self.closed {
                        HistogramClosed::Left => {
                            // [a, b) except last bin is [a, b]
                            if xf == *edges.last().unwrap() {
                                b = (num_bins - 1) as isize;
                            }
                        }
                        HistogramClosed::Right => {
                            // (a, b] except first bin is [a, b]
                            if xf == edges[0] {
                                b = 0;
                            } else if ((xf - edges[0]) % bin_width).abs() < f64::EPSILON
                                && xf != edges[0]
                            {
                                b -= 1;
                            }
                        }
                    }
                    b.clamp(0, (num_bins - 1) as isize) as usize
                }
            } else {
                match edges.binary_search_by(|e| e.partial_cmp(&xf).unwrap()) {
                    Ok(i) => {
                        // exact match on edge
                        match self.closed {
                            HistogramClosed::Left => {
                                if i == num_bins {
                                    num_bins - 1
                                } else {
                                    i
                                }
                            }
                            HistogramClosed::Right => {
                                if i == 0 {
                                    0
                                } else {
                                    i - 1
                                }
                            }
                        }
                    }
                    Err(i) => {
                        // i is the insertion point
                        if i == 0 {
                            0 // out of bounds left
                        } else if i > num_bins {
                            num_bins - 1 // out of bounds right
                        } else {
                            i - 1
                        }
                    }
                }
            };

            counts[bin_idx] += 1;
            if let Some(ref mut q) = quantized {
                q.push(bin_idx as u32);
            }
        }

        match self.output {
            HistogramOutput::Counts => ViewBuffer::from_vec_with_shape(counts, vec![num_bins]),
            HistogramOutput::Normalized => {
                let total = data.len() as f64;
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
                let total = data.len() as f64;
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
        }
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
        _input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        // A per-row bin count is checked per row.
        if let Bins::Edges(edges) = &self.bins {
            if edges.len() < 2 {
                return Err(ValidationError::InvalidParameter {
                    param: "edges".to_string(),
                    reason: "edges must contain at least 2 values".to_string(),
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

    #[test]
    fn test_histogram_counts() {
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![8]);

        let op = HistogramOp::new(4).with_range(0.0, 8.0);
        let result = op.execute(&buffer);

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
        let result = op.execute(&buffer);

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
        let result = op.execute(&buffer);

        let quantized = result.as_slice::<u32>();
        assert_eq!(quantized.len(), 3);
        assert_eq!(quantized[0], 0); // 0 -> bin 0
        assert_eq!(quantized[1], 2); // 128 -> bin 2
        assert_eq!(quantized[2], 3); // 255 -> bin 3 (clamped)
    }
}
