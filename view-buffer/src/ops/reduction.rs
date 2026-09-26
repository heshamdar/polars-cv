//! Reduction operations for statistical aggregations.
//!
//! This module provides operations that reduce array dimensions
//! by computing statistics like max, min, mean, std, and sum.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::shape_rule::OpShape;
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

use crate::mode::{Exec, Mode};
use polars_cv_macros::{Ops, Resolve};

/// The reductions: one variant per wire op (see `crate::mode`).
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum ReductionOp<M: Mode = Exec> {
    /// Reduce buffer by computing the maximum value.
    #[op(name = "reduce_max", sample = {"axis": 0})]
    Max {
        /// Axis to reduce along. None for global reduction. It fixes the
        /// output rank, so it is literal-only.
        axis: Option<M::L<u32>>,
    },
    /// Reduce buffer by computing the minimum value.
    #[op(name = "reduce_min", sample = {"axis": 0})]
    Min {
        /// Axis to reduce along. None for global reduction. It fixes the
        /// output rank, so it is literal-only.
        axis: Option<M::L<u32>>,
    },
    /// Compute arithmetic mean.
    #[op(name = "reduce_mean", sample = {"axis": 1})]
    Mean {
        /// Axis to reduce along. None for global reduction. It fixes the
        /// output rank, so it is literal-only.
        axis: Option<M::L<u32>>,
    },
    /// Reduce buffer by computing the standard deviation.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").reduce_std(ddof=1)
    #[op(name = "reduce_std", sample = {"axis": 0, "ddof": 1})]
    Std {
        /// Axis to reduce along. None for global reduction.
        axis: Option<M::L<u32>>,
        /// Delta degrees of freedom. 0 for population std (default), 1 for sample
        /// std. Accepts a Polars expression for per-row dynamic values.
        #[param(default = 0)]
        ddof: M::V<u8>,
    },
    /// Sum all elements in the buffer.
    #[op(name = "reduce_sum", sample = {})]
    Sum,
    /// Index of the maximum value along an axis.
    ///
    /// Unlike other reductions it always requires an axis: a global index
    /// is ambiguous for a multi-dimensional array. The result has the reduced
    /// shape and an i64 dtype.
    #[op(name = "reduce_argmax", sample = {"axis": 0})]
    ArgMax {
        /// Axis along which to find the index.
        axis: M::L<u32>,
    },
    /// Index of the minimum value along an axis.
    ///
    /// Unlike other reductions it always requires an axis: a global index
    /// is ambiguous for a multi-dimensional array. The result has the reduced
    /// shape and an i64 dtype.
    #[op(name = "reduce_argmin", sample = {"axis": 0})]
    ArgMin {
        /// Axis along which to find the index.
        axis: M::L<u32>,
    },
    /// Count set bits (1s) in the buffer.
    #[op(name = "reduce_popcount", sample = {})]
    PopCount,
    /// Compute the q-th percentile of all values (linear interpolation, as numpy's
    /// default).
    #[op(name = "reduce_percentile", sample = {"q": 50.0})]
    Percentile {
        /// Percentile to compute, in [0, 100]. Accepts a Polars expression for
        /// per-row dynamic values.
        q: M::V<f64>,
    },
}

impl<M: Mode> ReductionOp<M> {
    /// Every reduction's parameters are independent.
    pub fn check(&self) -> Result<(), String> {
        Ok(())
    }

    /// The axis reduced, `None` for a global reduction.
    pub fn axis(&self) -> Option<usize> {
        match self {
            ReductionOp::Max { axis }
            | ReductionOp::Min { axis }
            | ReductionOp::Mean { axis }
            | ReductionOp::Std { axis, .. } => axis.as_ref().map(|a| M::lit(a) as usize),
            ReductionOp::ArgMax { axis } | ReductionOp::ArgMin { axis } => {
                Some(M::lit(axis) as usize)
            }
            ReductionOp::Sum | ReductionOp::PopCount | ReductionOp::Percentile { .. } => None,
        }
    }

    /// How this op's output shape follows from its input: the axis removed,
    /// or a single slot for a global reduction.
    pub fn shape(&self) -> OpShape {
        OpShape::Reduce { axis: self.axis() }
    }

    /// The domain this reduction produces: a scalar for a global one, a
    /// smaller buffer for an axis one.
    pub fn output_domain(&self) -> Domain {
        if self.axis().is_some() {
            Domain::Buffer
        } else {
            Domain::Scalar
        }
    }
}

impl ReductionOp {
    /// Execute the reduction on a buffer.
    pub fn execute(&self, buffer: &ViewBuffer) -> ViewBuffer {
        let contig = buffer.to_contiguous();

        // PopCount has special handling for each type
        if matches!(self, ReductionOp::PopCount) {
            return self.execute_popcount(&contig);
        }

        // Percentile has special handling (needs sorting)
        if let ReductionOp::Percentile { q } = self {
            return self.execute_percentile(&contig, *q);
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

    /// Execute percentile reduction with linear interpolation (matches numpy default).
    fn execute_percentile(&self, buffer: &ViewBuffer, q: f64) -> ViewBuffer {
        let mut values: Vec<f64> = match buffer.dtype() {
            DType::U8 => buffer.as_slice::<u8>().iter().map(|&x| x as f64).collect(),
            DType::I8 => buffer.as_slice::<i8>().iter().map(|&x| x as f64).collect(),
            DType::U16 => buffer.as_slice::<u16>().iter().map(|&x| x as f64).collect(),
            DType::I16 => buffer.as_slice::<i16>().iter().map(|&x| x as f64).collect(),
            DType::U32 => buffer.as_slice::<u32>().iter().map(|&x| x as f64).collect(),
            DType::I32 => buffer.as_slice::<i32>().iter().map(|&x| x as f64).collect(),
            DType::U64 => buffer.as_slice::<u64>().iter().map(|&x| x as f64).collect(),
            DType::I64 => buffer.as_slice::<i64>().iter().map(|&x| x as f64).collect(),
            DType::F32 => buffer.as_slice::<f32>().iter().map(|&x| x as f64).collect(),
            DType::F64 => buffer.as_slice::<f64>().to_vec(),
        };

        if values.is_empty() {
            return ViewBuffer::from_scalar(f64::NAN);
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        if n == 1 {
            return ViewBuffer::from_scalar(values[0]);
        }

        // Linear interpolation matching numpy.percentile default method
        let q_frac = q.clamp(0.0, 100.0) / 100.0;
        let pos = q_frac * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = lo + 1;
        let frac = pos - lo as f64;

        let result = if hi >= n {
            values[lo]
        } else {
            values[lo] * (1.0 - frac) + values[hi] * frac
        };

        ViewBuffer::from_scalar(result)
    }

    fn execute_typed<T>(&self, buffer: &ViewBuffer) -> ViewBuffer
    where
        T: Copy + Default + PartialOrd + num_traits::Num + num_traits::NumCast + ViewType + 'static,
    {
        let data = buffer.as_slice::<T>();
        let _shape = buffer.shape();

        match self {
            ReductionOp::Max { axis: None } => {
                assert!(!data.is_empty(), "Cannot reduce Max on empty buffer");
                let max_val = data
                    .iter()
                    .copied()
                    .fold(data[0], |a, b| if a > b { a } else { b });
                ViewBuffer::from_scalar(max_val)
            }
            ReductionOp::Min { axis: None } => {
                assert!(!data.is_empty(), "Cannot reduce Min on empty buffer");
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
                let denominator = n - *ddof as f64;
                if denominator <= 0.0 {
                    return ViewBuffer::from_scalar(f64::NAN);
                }
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
                    / denominator;
                let std = variance.sqrt();
                ViewBuffer::from_scalar(std)
            }
            ReductionOp::Sum => {
                let sum: f64 = data
                    .iter()
                    .copied()
                    .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                    .sum();
                ViewBuffer::from_scalar(sum)
            }
            // Axis-based reductions
            ReductionOp::Max { axis: Some(ax) } => {
                self.reduce_axis::<T, _>(buffer, *ax as usize, |slice: &[T]| {
                    slice
                        .iter()
                        .copied()
                        .fold(slice[0], |a, b| if a > b { a } else { b })
                })
            }
            ReductionOp::Min { axis: Some(ax) } => {
                self.reduce_axis::<T, _>(buffer, *ax as usize, |slice: &[T]| {
                    slice
                        .iter()
                        .copied()
                        .fold(slice[0], |a, b| if a < b { a } else { b })
                })
            }
            ReductionOp::Mean { axis: Some(ax) } => {
                // For axis reduction, output is float
                self.reduce_axis_to_f64::<T, _>(buffer, *ax as usize, |slice: &[T]| {
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
                self.reduce_axis_to_f64::<T, _>(buffer, *ax as usize, move |slice: &[T]| {
                    let n = slice.len() as f64;
                    let denominator = n - ddof_val as f64;
                    if denominator <= 0.0 {
                        return f64::NAN;
                    }
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
                        / denominator;
                    variance.sqrt()
                })
            }
            ReductionOp::ArgMax { axis } => {
                self.reduce_axis_argmax::<T>(buffer, *axis as usize, true)
            }
            ReductionOp::ArgMin { axis } => {
                self.reduce_axis_argmax::<T>(buffer, *axis as usize, false)
            }
            // PopCount and Percentile are handled specially in execute() before calling execute_typed()
            ReductionOp::PopCount => unreachable!("PopCount is handled in execute()"),
            ReductionOp::Percentile { .. } => unreachable!("Percentile is handled in execute()"),
        }
    }

    #[allow(clippy::needless_range_loop)]
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

        // Coordinate mapping must run over the REAL remaining dims (possibly
        // none, for a 1-D input); the [1] clamp below only pads the returned
        // buffer's shape.
        let coord_shape = out_shape.clone();
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![T::default(); out_size];

        // Compute strides
        let strides = compute_strides(shape);

        // Allocate gather buffer once and reuse for each output element
        // to avoid O(out_size) heap allocations.
        let mut gather_buf = vec![T::default(); axis_size];
        let mut in_coords = vec![0usize; shape.len()];

        for (out_idx, out) in output.iter_mut().enumerate() {
            let out_coords = linear_to_coords(out_idx, &coord_shape);

            // Build full input coordinates from output coordinates with axis slot
            for (i, &c) in out_coords.iter().enumerate() {
                let target = if i < axis { i } else { i + 1 };
                in_coords[target] = c;
            }

            // Gather values along the reduction axis
            for a in 0..axis_size {
                in_coords[axis] = a;
                let in_idx = coords_to_linear(&in_coords, &strides);
                gather_buf[a] = data[in_idx];
            }

            *out = f(&gather_buf);
        }

        ViewBuffer::from_vec_with_shape(output, out_shape)
    }

    #[allow(clippy::needless_range_loop)]
    fn reduce_axis_to_f64<T, F>(&self, buffer: &ViewBuffer, axis: usize, f: F) -> ViewBuffer
    where
        T: Copy + Default + num_traits::NumCast + ViewType + 'static,
        F: Fn(&[T]) -> f64,
    {
        let shape = buffer.shape();
        let data = buffer.as_slice::<T>();

        let mut out_shape: Vec<usize> = shape.to_vec();
        let axis_size = out_shape.remove(axis);

        // Coordinate mapping must run over the REAL remaining dims (possibly
        // none, for a 1-D input); the [1] clamp below only pads the returned
        // buffer's shape.
        let coord_shape = out_shape.clone();
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![0.0f64; out_size];

        let strides = compute_strides(shape);

        // Allocate gather buffer once and reuse
        let mut gather_buf = vec![T::default(); axis_size];
        let mut in_coords = vec![0usize; shape.len()];

        for (out_idx, out) in output.iter_mut().enumerate() {
            let out_coords = linear_to_coords(out_idx, &coord_shape);

            for (i, &c) in out_coords.iter().enumerate() {
                let target = if i < axis { i } else { i + 1 };
                in_coords[target] = c;
            }

            for a in 0..axis_size {
                in_coords[axis] = a;
                let in_idx = coords_to_linear(&in_coords, &strides);
                gather_buf[a] = data[in_idx];
            }

            *out = f(&gather_buf);
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

        // Coordinate mapping must run over the REAL remaining dims (possibly
        // none, for a 1-D input); the [1] clamp below only pads the returned
        // buffer's shape.
        let coord_shape = out_shape.clone();
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![0i64; out_size];

        let strides = compute_strides(shape);

        // Reusable coordinate buffer (avoids per-iteration clone + insert)
        let mut in_coords = vec![0usize; shape.len()];

        for (out_idx, out) in output.iter_mut().enumerate() {
            let out_coords = linear_to_coords(out_idx, &coord_shape);

            // Build full coordinates from output coords
            for (i, &c) in out_coords.iter().enumerate() {
                let target = if i < axis { i } else { i + 1 };
                in_coords[target] = c;
            }

            in_coords[axis] = 0;
            let first_in_idx = coords_to_linear(&in_coords, &strides);
            let mut best_val = data[first_in_idx];
            let mut best_idx = 0usize;

            for a in 1..axis_size {
                in_coords[axis] = a;
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

impl<M: Mode> Op for ReductionOp<M> {
    fn name(&self) -> &'static str {
        match self {
            ReductionOp::Max { .. } => "Max",
            ReductionOp::Min { .. } => "Min",
            ReductionOp::Mean { .. } => "Mean",
            ReductionOp::Std { .. } => "Std",
            ReductionOp::Sum => "Sum",
            ReductionOp::ArgMax { .. } => "ArgMax",
            ReductionOp::ArgMin { .. } => "ArgMin",
            ReductionOp::PopCount => "PopCount",
            ReductionOp::Percentile { .. } => "Percentile",
        }
    }

    fn shape(&self) -> OpShape {
        ReductionOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::RequiresContiguous
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false // A reduction is not an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // A reduction aggregates over the whole input (or a whole axis), so no
        // output pixel maps to a bounded input window: a hard reorder barrier.
        // Matched exhaustively (not a blanket) so a new variant must reconfirm
        // this rather than silently inherit Global.
        match self {
            ReductionOp::Max { .. }
            | ReductionOp::Min { .. }
            | ReductionOp::Mean { .. }
            | ReductionOp::Std { .. }
            | ReductionOp::Sum
            | ReductionOp::ArgMax { .. }
            | ReductionOp::ArgMin { .. }
            | ReductionOp::PopCount
            | ReductionOp::Percentile { .. } => SpatialDependency::Global,
        }
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
        if let Some(ax) = self.axis() {
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
            | ReductionOp::Sum
            | ReductionOp::PopCount
            | ReductionOp::Percentile { .. } => OutputDTypeRule::ForceF64,
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

use super::util::{coords_to_linear, linear_to_coords};

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
    fn test_axis_reduction_of_1d_input() {
        // Regression: reducing a 1-D buffer along axis 0 panicked — the [1]
        // shape clamp fed the coordinate mapping a phantom dimension. The
        // result is a rank-1 single-slot buffer, per ReduceByOne's clamp.
        let buffer = ViewBuffer::from_vec_with_shape(vec![1.0f32, 5.0, 3.0], vec![3]);
        for (op, expected) in [
            (ReductionOp::Max { axis: Some(0) }, 5.0f32),
            (ReductionOp::Min { axis: Some(0) }, 1.0),
        ] {
            let result = op.execute(&buffer);
            assert_eq!(result.shape(), &[1]);
            assert_eq!(result.as_slice::<f32>(), &[expected]);
        }

        // Mean computes in f64 along an axis.
        let result = ReductionOp::Mean { axis: Some(0) }.execute(&buffer);
        assert_eq!(result.shape(), &[1]);
        assert!((result.as_slice::<f64>()[0] - 3.0).abs() < 1e-10);

        let argmax = ReductionOp::ArgMax { axis: 0 }.execute(&buffer);
        assert_eq!(argmax.shape(), &[1]);
        assert_eq!(argmax.as_slice::<i64>(), &[1]);
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
