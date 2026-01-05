//! Reduction operations for statistical aggregations.
//!
//! This module provides operations that reduce array dimensions
//! by computing statistics like max, min, mean, std, and sum.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Reduction operations that aggregate across dimensions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ReductionOp {
    /// Maximum value (global or along axis).
    Max {
        /// Axis to reduce. None = global reduction.
        axis: Option<usize>,
    },
    /// Minimum value (global or along axis).
    Min {
        /// Axis to reduce. None = global reduction.
        axis: Option<usize>,
    },
    /// Arithmetic mean (global or along axis).
    Mean {
        /// Axis to reduce. None = global reduction.
        axis: Option<usize>,
    },
    /// Standard deviation (global or along axis).
    Std {
        /// Axis to reduce. None = global reduction.
        axis: Option<usize>,
        /// Degrees of freedom (0 = population, 1 = sample).
        ddof: u8,
    },
    /// Sum (global or along axis).
    Sum {
        /// Axis to reduce. None = global reduction.
        axis: Option<usize>,
    },
    /// Index of maximum value along axis.
    ArgMax {
        /// Axis along which to find the maximum.
        axis: usize,
    },
    /// Index of minimum value along axis.
    ArgMin {
        /// Axis along which to find the minimum.
        axis: usize,
    },
    /// Population count - count the number of set bits in the buffer.
    ///
    /// This is a global reduction that counts all set bits across the entire buffer.
    /// Useful for:
    /// - Computing Hamming distance between hashes (XOR then popcount)
    /// - Counting pixels in binary masks
    /// - Sparse array analysis
    ///
    /// For integer types, counts actual bits. For float types, casts to i64 first.
    PopCount,
}

impl ReductionOp {
    /// Execute the reduction on a buffer.
    pub fn execute(&self, buffer: &ViewBuffer) -> ViewBuffer {
        let contig = buffer.to_contiguous();

        // PopCount has special handling for each type
        if matches!(self, ReductionOp::PopCount) {
            return self.execute_popcount(&contig);
        }

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

    /// Execute popcount reduction - count all set bits in the buffer.
    fn execute_popcount(&self, buffer: &ViewBuffer) -> ViewBuffer {
        let count: u64 = match buffer.dtype() {
            DType::U8 => buffer
                .as_slice::<u8>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I8 => buffer
                .as_slice::<i8>()
                .iter()
                .map(|x| (*x as u8).count_ones() as u64)
                .sum(),
            DType::U16 => buffer
                .as_slice::<u16>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I16 => buffer
                .as_slice::<i16>()
                .iter()
                .map(|x| (*x as u16).count_ones() as u64)
                .sum(),
            DType::U32 => buffer
                .as_slice::<u32>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I32 => buffer
                .as_slice::<i32>()
                .iter()
                .map(|x| (*x as u32).count_ones() as u64)
                .sum(),
            DType::U64 => buffer
                .as_slice::<u64>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I64 => buffer
                .as_slice::<i64>()
                .iter()
                .map(|x| (*x as u64).count_ones() as u64)
                .sum(),
            // For floats, cast to i64 and count bits
            DType::F32 => buffer
                .as_slice::<f32>()
                .iter()
                .map(|x| (*x as i64 as u64).count_ones() as u64)
                .sum(),
            DType::F64 => buffer
                .as_slice::<f64>()
                .iter()
                .map(|x| (*x as i64 as u64).count_ones() as u64)
                .sum(),
        };

        ViewBuffer::from_scalar(count as f64)
    }

    fn execute_typed<T>(&self, buffer: &ViewBuffer) -> ViewBuffer
    where
        T: Copy + Default + PartialOrd + num_traits::Num + num_traits::NumCast + ViewType + 'static,
    {
        let data = buffer.as_slice::<T>();
        let _shape = buffer.shape();

        match self {
            ReductionOp::Max { axis: None } => {
                let max_val = data
                    .iter()
                    .copied()
                    .fold(data[0], |a, b| if a > b { a } else { b });
                ViewBuffer::from_scalar(max_val)
            }
            ReductionOp::Min { axis: None } => {
                let min_val = data
                    .iter()
                    .copied()
                    .fold(data[0], |a, b| if a < b { a } else { b });
                ViewBuffer::from_scalar(min_val)
            }
            ReductionOp::Mean { axis: None } => {
                let sum: f64 = data
                    .iter()
                    .copied()
                    .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                    .sum();
                let mean = sum / data.len() as f64;
                ViewBuffer::from_scalar(mean)
            }
            ReductionOp::Std { axis: None, ddof } => {
                let n = data.len() as f64;
                let sum: f64 = data
                    .iter()
                    .copied()
                    .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                    .sum();
                let mean = sum / n;
                let variance: f64 = data
                    .iter()
                    .copied()
                    .map(|x| {
                        let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                        (xf - mean).powi(2)
                    })
                    .sum::<f64>()
                    / (n - *ddof as f64);
                let std = variance.sqrt();
                ViewBuffer::from_scalar(std)
            }
            ReductionOp::Sum { axis: None } => {
                let sum: f64 = data
                    .iter()
                    .copied()
                    .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                    .sum();
                ViewBuffer::from_scalar(sum)
            }
            // Axis-based reductions
            ReductionOp::Max { axis: Some(ax) } => {
                self.reduce_axis::<T, _>(buffer, *ax, |slice: &[T]| {
                    slice
                        .iter()
                        .copied()
                        .fold(slice[0], |a, b| if a > b { a } else { b })
                })
            }
            ReductionOp::Min { axis: Some(ax) } => {
                self.reduce_axis::<T, _>(buffer, *ax, |slice: &[T]| {
                    slice
                        .iter()
                        .copied()
                        .fold(slice[0], |a, b| if a < b { a } else { b })
                })
            }
            ReductionOp::Mean { axis: Some(ax) } => {
                // For axis reduction, output is float
                self.reduce_axis_to_f64::<T, _>(buffer, *ax, |slice: &[T]| {
                    let sum: f64 = slice
                        .iter()
                        .copied()
                        .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                        .sum();
                    sum / slice.len() as f64
                })
            }
            ReductionOp::Std {
                axis: Some(ax),
                ddof,
            } => {
                let ddof_val = *ddof;
                self.reduce_axis_to_f64::<T, _>(buffer, *ax, move |slice: &[T]| {
                    let n = slice.len() as f64;
                    let sum: f64 = slice
                        .iter()
                        .copied()
                        .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                        .sum();
                    let mean = sum / n;
                    let variance: f64 = slice
                        .iter()
                        .copied()
                        .map(|x| {
                            let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                            (xf - mean).powi(2)
                        })
                        .sum::<f64>()
                        / (n - ddof_val as f64);
                    variance.sqrt()
                })
            }
            ReductionOp::Sum { axis: Some(ax) } => {
                self.reduce_axis_to_f64::<T, _>(buffer, *ax, |slice: &[T]| {
                    slice
                        .iter()
                        .copied()
                        .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                        .sum()
                })
            }
            ReductionOp::ArgMax { axis } => self.reduce_axis_argmax::<T>(buffer, *axis, true),
            ReductionOp::ArgMin { axis } => self.reduce_axis_argmax::<T>(buffer, *axis, false),
            // PopCount is handled specially in execute() before calling execute_typed()
            ReductionOp::PopCount => unreachable!("PopCount is handled in execute()"),
        }
    }

    fn reduce_axis<T, F>(&self, buffer: &ViewBuffer, axis: usize, f: F) -> ViewBuffer
    where
        T: Copy + Default + ViewType + 'static,
        F: Fn(&[T]) -> T,
    {
        let shape = buffer.shape();
        let data = buffer.as_slice::<T>();

        // Calculate output shape (remove the reduced axis)
        let mut out_shape: Vec<usize> = shape.to_vec();
        let axis_size = out_shape.remove(axis);

        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![T::default(); out_size];

        // Compute strides
        let strides = compute_strides(shape);

        // Iterate over output positions
        for (out_idx, out) in output.iter_mut().enumerate() {
            // Convert to coordinates
            let out_coords = linear_to_coords(out_idx, &out_shape);

            // Gather values along axis
            let mut slice = Vec::with_capacity(axis_size);
            for a in 0..axis_size {
                // Insert axis coordinate
                let mut in_coords = out_coords.clone();
                in_coords.insert(axis, a);
                let in_idx = coords_to_linear(&in_coords, &strides);
                slice.push(data[in_idx]);
            }

            *out = f(&slice);
        }

        ViewBuffer::from_vec_with_shape(output, out_shape)
    }

    fn reduce_axis_to_f64<T, F>(&self, buffer: &ViewBuffer, axis: usize, f: F) -> ViewBuffer
    where
        T: Copy + Default + num_traits::NumCast + ViewType + 'static,
        F: Fn(&[T]) -> f64,
    {
        let shape = buffer.shape();
        let data = buffer.as_slice::<T>();

        let mut out_shape: Vec<usize> = shape.to_vec();
        let axis_size = out_shape.remove(axis);

        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![0.0f64; out_size];

        let strides = compute_strides(shape);

        for (out_idx, out) in output.iter_mut().enumerate() {
            let out_coords = linear_to_coords(out_idx, &out_shape);

            let mut slice = Vec::with_capacity(axis_size);
            for a in 0..axis_size {
                let mut in_coords = out_coords.clone();
                in_coords.insert(axis, a);
                let in_idx = coords_to_linear(&in_coords, &strides);
                slice.push(data[in_idx]);
            }

            *out = f(&slice);
        }

        ViewBuffer::from_vec_with_shape(output, out_shape)
    }

    fn reduce_axis_argmax<T>(&self, buffer: &ViewBuffer, axis: usize, is_max: bool) -> ViewBuffer
    where
        T: Copy + Default + PartialOrd + ViewType + 'static,
    {
        let shape = buffer.shape();
        let data = buffer.as_slice::<T>();

        let mut out_shape: Vec<usize> = shape.to_vec();
        let axis_size = out_shape.remove(axis);

        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![0i64; out_size];

        let strides = compute_strides(shape);

        for (out_idx, out) in output.iter_mut().enumerate() {
            let out_coords = linear_to_coords(out_idx, &out_shape);

            let mut best_idx = 0usize;
            let mut in_coords = out_coords.clone();
            in_coords.insert(axis, 0);
            let first_in_idx = coords_to_linear(&in_coords, &strides);
            let mut best_val = data[first_in_idx];

            for a in 1..axis_size {
                let mut in_coords = out_coords.clone();
                in_coords.insert(axis, a);
                let in_idx = coords_to_linear(&in_coords, &strides);
                let val = data[in_idx];

                let is_better = if is_max {
                    val > best_val
                } else {
                    val < best_val
                };
                if is_better {
                    best_val = val;
                    best_idx = a;
                }
            }

            *out = best_idx as i64;
        }

        ViewBuffer::from_vec_with_shape(output, out_shape)
    }
}

impl Op for ReductionOp {
    fn name(&self) -> &'static str {
        match self {
            ReductionOp::Max { .. } => "Max",
            ReductionOp::Min { .. } => "Min",
            ReductionOp::Mean { .. } => "Mean",
            ReductionOp::Std { .. } => "Std",
            ReductionOp::Sum { .. } => "Sum",
            ReductionOp::ArgMax { .. } => "ArgMax",
            ReductionOp::ArgMin { .. } => "ArgMin",
            ReductionOp::PopCount => "PopCount",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_shape = inputs[0];

        let axis = match self {
            ReductionOp::Max { axis }
            | ReductionOp::Min { axis }
            | ReductionOp::Mean { axis }
            | ReductionOp::Std { axis, .. }
            | ReductionOp::Sum { axis } => *axis,
            ReductionOp::ArgMax { axis } | ReductionOp::ArgMin { axis } => Some(*axis),
            // PopCount is always a global reduction
            ReductionOp::PopCount => None,
        };

        match axis {
            None => vec![1], // Global reduction
            Some(ax) => {
                let mut out_shape: Vec<usize> = input_shape.to_vec();
                if ax < out_shape.len() {
                    out_shape.remove(ax);
                }
                if out_shape.is_empty() {
                    out_shape.push(1);
                }
                out_shape
            }
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        match self {
            ReductionOp::Mean { .. } | ReductionOp::Std { .. } => DType::F64,
            ReductionOp::Sum { axis: None } => DType::F64,
            ReductionOp::Sum { axis: Some(_) } => DType::F64,
            ReductionOp::ArgMax { .. } | ReductionOp::ArgMin { .. } => DType::I64,
            // PopCount returns the count as f64
            ReductionOp::PopCount => DType::F64,
            _ => inputs[0],
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
        input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        let axis = match self {
            ReductionOp::Max { axis }
            | ReductionOp::Min { axis }
            | ReductionOp::Mean { axis }
            | ReductionOp::Std { axis, .. }
            | ReductionOp::Sum { axis } => *axis,
            ReductionOp::ArgMax { axis } | ReductionOp::ArgMin { axis } => Some(*axis),
            // PopCount is always a global reduction (no axis)
            ReductionOp::PopCount => None,
        };

        if let Some(ax) = axis {
            if ax >= input_shapes[0].len() {
                return Err(ValidationError::InvalidAxis {
                    axis: ax,
                    ndim: input_shapes[0].len(),
                });
            }
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
        match self {
            ReductionOp::Mean { .. }
            | ReductionOp::Std { .. }
            | ReductionOp::Sum { .. }
            | ReductionOp::PopCount => OutputDTypeRule::ForceF64,
            ReductionOp::ArgMax { .. } | ReductionOp::ArgMin { .. } => OutputDTypeRule::ForceI64,
            _ => OutputDTypeRule::PreserveInput,
        }
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn linear_to_coords(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut remaining = index;
    for i in (0..shape.len()).rev() {
        coords[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    coords
}

fn coords_to_linear(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_max() {
        let data = vec![1u8, 5, 3, 9, 2];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![5]);
        let op = ReductionOp::Max { axis: None };
        let result = op.execute(&buffer);
        assert_eq!(result.as_slice::<u8>()[0], 9);
    }

    #[test]
    fn test_global_mean() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![5]);
        let op = ReductionOp::Mean { axis: None };
        let result = op.execute(&buffer);
        assert!((result.as_slice::<f64>()[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_popcount_u8() {
        // 0xFF = 8 bits, 0x00 = 0 bits, 0x0F = 4 bits, 0xAA = 4 bits (10101010)
        let data = vec![0xFFu8, 0x00, 0x0F, 0xAA];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![4]);
        let op = ReductionOp::PopCount;
        let result = op.execute(&buffer);
        // 8 + 0 + 4 + 4 = 16 bits
        assert!((result.as_slice::<f64>()[0] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_popcount_for_hamming_distance() {
        // Simulate XOR of two hashes and count bits
        // hash1 = [0xFF, 0x00] (11111111 00000000)
        // hash2 = [0x0F, 0x0F] (00001111 00001111)
        // XOR   = [0xF0, 0x0F] (11110000 00001111)
        // popcount = 4 + 4 = 8 bits different
        let xor_result = vec![0xF0u8, 0x0F];
        let buffer = ViewBuffer::from_vec_with_shape(xor_result, vec![2]);
        let op = ReductionOp::PopCount;
        let result = op.execute(&buffer);
        assert!((result.as_slice::<f64>()[0] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_popcount_identical_hashes() {
        // XOR of identical hashes = all zeros = 0 bits
        let xor_result = vec![0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let buffer = ViewBuffer::from_vec_with_shape(xor_result, vec![8]);
        let op = ReductionOp::PopCount;
        let result = op.execute(&buffer);
        assert!((result.as_slice::<f64>()[0] - 0.0).abs() < 1e-10);
    }
}
