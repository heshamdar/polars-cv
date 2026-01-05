//! Histogram and quantization operations.
//!
//! This module provides operations for computing histograms and
//! quantizing arrays into discrete bins.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;

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
}

/// Histogram and quantization operations.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HistogramOp {
    /// Number of bins.
    pub bins: usize,
    /// Value range (min, max). None = auto from data.
    pub range: Option<(f64, f64)>,
    /// Output mode.
    pub output: HistogramOutput,
}

impl HistogramOp {
    /// Create a new histogram operation.
    pub fn new(bins: usize) -> Self {
        Self {
            bins,
            range: None,
            output: HistogramOutput::Counts,
        }
    }

    /// Set the value range.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.range = Some((min, max));
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

        // Determine range
        let (min_val, max_val) = match self.range {
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

        let bin_width = (max_val - min_val) / self.bins as f64;

        match self.output {
            HistogramOutput::Counts => {
                let mut counts = vec![0u64; self.bins];
                for &x in data {
                    let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                    let bin = ((xf - min_val) / bin_width) as usize;
                    let bin = bin.min(self.bins - 1); // Clamp to last bin
                    counts[bin] += 1;
                }
                ViewBuffer::from_vec_with_shape(counts, vec![self.bins])
            }
            HistogramOutput::Normalized => {
                let mut counts = vec![0u64; self.bins];
                for &x in data {
                    let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                    let bin = ((xf - min_val) / bin_width) as usize;
                    let bin = bin.min(self.bins - 1);
                    counts[bin] += 1;
                }
                let total = data.len() as f64;
                let normalized: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();
                ViewBuffer::from_vec_with_shape(normalized, vec![self.bins])
            }
            HistogramOutput::Quantized => {
                let mut quantized = Vec::with_capacity(data.len());
                for &x in data {
                    let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                    let bin = ((xf - min_val) / bin_width) as u32;
                    let bin = bin.min(self.bins as u32 - 1);
                    quantized.push(bin);
                }
                ViewBuffer::from_vec_with_shape(quantized, shape.to_vec())
            }
            HistogramOutput::Edges => {
                let edges: Vec<f64> = (0..=self.bins)
                    .map(|i| min_val + i as f64 * bin_width)
                    .collect();
                ViewBuffer::from_vec_with_shape(edges, vec![self.bins + 1])
            }
        }
    }
}

impl Op for HistogramOp {
    fn name(&self) -> &'static str {
        "Histogram"
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        match self.output {
            HistogramOutput::Counts | HistogramOutput::Normalized => vec![self.bins],
            HistogramOutput::Quantized => inputs[0].to_vec(),
            HistogramOutput::Edges => vec![self.bins + 1],
        }
    }

    fn infer_dtype(&self, _inputs: &[DType]) -> DType {
        match self.output {
            HistogramOutput::Counts => DType::U64,
            HistogramOutput::Normalized => DType::F64,
            HistogramOutput::Quantized => DType::U32,
            HistogramOutput::Edges => DType::F64,
        }
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
        if self.bins == 0 {
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
            HistogramOutput::Normalized | HistogramOutput::Edges => OutputDTypeRule::ForceF64,
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
