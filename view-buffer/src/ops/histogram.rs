//! Histogram and quantization operations.
//!
//! This module provides operations for computing histograms and
//! quantizing arrays into discrete bins.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::cost::OpCost;
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Output mode for histogram operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HistogramClosed {
    /// Intervals are left-closed [a, b). (Last bin is [a, b]).
    #[default]
    Left,
    /// Intervals are right-closed (a, b]. (First bin is [a, b]).
    Right,
}

crate::naming::named_variants!(HistogramOutput {
    "counts" => Counts,
    "normalized" => Normalized,
    "quantized" => Quantized,
    "edges" => Edges,
    "buckets" => Buckets,
});

crate::naming::named_variants!(HistogramClosed {
    "left" => Left,
    "right" => Right,
});

/// Histogram and quantization operations.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HistogramOp {
    /// Number of bins.
    pub bins: usize,
    /// Value range (min, max). None = auto from data.
    pub range: Option<(f64, f64)>,
    /// Explicit bin edges. If provided, overrides `bins` and `range`.
    pub edges: Option<Vec<f64>>,
    /// Interval closedness.
    pub closed: HistogramClosed,
    /// Output mode.
    pub output: HistogramOutput,
}

impl HistogramOp {
    /// Create a new histogram operation.
    pub fn new(bins: usize) -> Self {
        Self {
            bins,
            range: None,
            edges: None,
            closed: HistogramClosed::Left,
            output: HistogramOutput::Counts,
        }
    }

    /// Set the value range.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Set explicit edges.
    pub fn with_edges(mut self, edges: Vec<f64>) -> Self {
        self.edges = Some(edges);
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
        let edges = if let Some(ref e) = self.edges {
            e.clone()
        } else {
            let (mut min_val, mut max_val) = match self.range {
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

            let bin_width = (max_val - min_val) / self.bins as f64;
            let mut e = Vec::with_capacity(self.bins + 1);
            for i in 0..=self.bins {
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

        let is_uniform = self.edges.is_none();

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

impl HistogramOp {
    /// The domain of this histogram's result: `Quantized` maps pixels in
    /// place (buffer); every other output mode yields a 1-D vector.
    ///
    /// Lives here, next to the op, as the single authority the graph layer
    /// reads (formerly duplicated in the DTO's output_domain match).
    pub fn output_domain(&self) -> Domain {
        match self.output {
            HistogramOutput::Quantized => Domain::Buffer,
            _ => Domain::Vector,
        }
    }
}

impl Op for HistogramOp {
    fn name(&self) -> &'static str {
        "Histogram"
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let num_bins = if let Some(ref edges) = self.edges {
            edges.len().saturating_sub(1)
        } else {
            self.bins
        };
        match self.output {
            HistogramOutput::Counts | HistogramOutput::Normalized => vec![num_bins],
            HistogramOutput::Quantized => inputs[0].to_vec(),
            HistogramOutput::Edges => vec![num_bins + 1],
            HistogramOutput::Buckets => vec![num_bins, 4],
        }
    }

    fn output_rank_rule(&self) -> OutputRankRule {
        match self.output {
            // 1-D bin vectors.
            HistogramOutput::Counts | HistogramOutput::Normalized | HistogramOutput::Edges => {
                OutputRankRule::Fixed(1)
            }
            // [num_bins, 4] bucket table.
            HistogramOutput::Buckets => OutputRankRule::Fixed(2),
            // Quantized relabels in place, preserving the input rank.
            HistogramOutput::Quantized => OutputRankRule::PreserveRank,
        }
    }

    fn output_channel_rule(&self) -> OutputChannelRule {
        // Bin vectors/tables have no channel concept; quantized output mirrors
        // the input channels but is consumed as a relabelled buffer.
        OutputChannelRule::NotApplicable
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::RequiresContiguous
    }

    fn intrinsic_cost(&self) -> OpCost {
        OpCost::Allocating
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
        if let Some(ref edges) = self.edges {
            if edges.len() < 2 {
                return Err(ValidationError::InvalidParameter {
                    param: "edges".to_string(),
                    reason: "edges must contain at least 2 values".to_string(),
                });
            }
        } else if self.bins == 0 {
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
        match self.output {
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
