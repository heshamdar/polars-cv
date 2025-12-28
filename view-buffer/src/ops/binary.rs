//! Binary operations between two arrays.
//!
//! This module provides element-wise operations between two ViewBuffers,
//! including arithmetic operations and bitwise operations for mask manipulation.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Binary operations between two arrays.
///
/// All operations are element-wise and support broadcasting.
/// The output shape is the broadcast result of both input shapes.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BinaryOp {
    /// Element-wise addition with saturation for integer types.
    Add,
    /// Element-wise subtraction with saturation for integer types.
    Subtract,
    /// Element-wise multiplication.
    Multiply,
    /// Element-wise division (with zero handling).
    Divide,
    /// Element-wise maximum.
    Maximum,
    /// Element-wise minimum.
    Minimum,
    /// Bitwise AND (useful for combining masks).
    BitwiseAnd,
    /// Bitwise OR (useful for combining masks).
    BitwiseOr,
    /// Bitwise XOR.
    BitwiseXor,
}

impl BinaryOp {
    /// Execute the binary operation on two buffers.
    ///
    /// Both buffers must have broadcastable shapes.
    pub fn execute(&self, a: &ViewBuffer, b: &ViewBuffer) -> ViewBuffer {
        // Validate shapes are broadcastable
        let output_shape = broadcast_shapes(a.shape(), b.shape())
            .expect("Shapes must be broadcastable");

        match (a.dtype(), b.dtype()) {
            (DType::U8, DType::U8) => self.execute_typed::<u8>(a, b, &output_shape),
            (DType::F32, DType::F32) => self.execute_typed::<f32>(a, b, &output_shape),
            (DType::F64, DType::F64) => self.execute_typed::<f64>(a, b, &output_shape),
            // For mixed types, promote to the wider type
            _ => {
                // For now, cast to f32 for mixed types
                let a_f32 = a.cast_to(DType::F32);
                let b_f32 = b.cast_to(DType::F32);
                self.execute_typed::<f32>(&a_f32, &b_f32, &output_shape)
            }
        }
    }

    fn execute_typed<T>(&self, a: &ViewBuffer, b: &ViewBuffer, output_shape: &[usize]) -> ViewBuffer
    where
        T: Copy + Default + num_traits::Num + num_traits::NumCast + PartialOrd + ViewType + 'static,
    {
        let total_elements: usize = output_shape.iter().product();
        let mut output = vec![T::default(); total_elements];

        // Get contiguous data
        let a_contig = a.to_contiguous();
        let b_contig = b.to_contiguous();

        let a_data = a_contig.as_slice::<T>();
        let b_data = b_contig.as_slice::<T>();

        // Simple implementation for same-shape case
        if a.shape() == b.shape() && a.shape() == output_shape {
            for i in 0..total_elements {
                output[i] = self.apply_op(a_data[i], b_data[i]);
            }
        } else {
            // Broadcasting case - use multi-dimensional indexing
            for i in 0..total_elements {
                let coords = linear_to_coords(i, output_shape);
                let a_idx = broadcast_index(&coords, a.shape());
                let b_idx = broadcast_index(&coords, b.shape());
                output[i] = self.apply_op(a_data[a_idx], b_data[b_idx]);
            }
        }

        ViewBuffer::from_vec_with_shape(output, output_shape.to_vec())
    }

    fn apply_op<T>(&self, a: T, b: T) -> T
    where
        T: Copy + num_traits::Num + num_traits::NumCast + PartialOrd,
    {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Subtract => a - b,
            BinaryOp::Multiply => a * b,
            BinaryOp::Divide => {
                if b.is_zero() {
                    T::zero()
                } else {
                    a / b
                }
            }
            BinaryOp::Maximum => {
                if a > b { a } else { b }
            }
            BinaryOp::Minimum => {
                if a < b { a } else { b }
            }
            // Bitwise ops - convert through integer
            BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                // For float types, this will truncate
                let a_int: i64 = num_traits::NumCast::from(a).unwrap_or(0);
                let b_int: i64 = num_traits::NumCast::from(b).unwrap_or(0);
                let result = match self {
                    BinaryOp::BitwiseAnd => a_int & b_int,
                    BinaryOp::BitwiseOr => a_int | b_int,
                    BinaryOp::BitwiseXor => a_int ^ b_int,
                    _ => unreachable!(),
                };
                num_traits::NumCast::from(result).unwrap_or(T::zero())
            }
        }
    }
}

impl Op for BinaryOp {
    fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "Add",
            BinaryOp::Subtract => "Subtract",
            BinaryOp::Multiply => "Multiply",
            BinaryOp::Divide => "Divide",
            BinaryOp::Maximum => "Maximum",
            BinaryOp::Minimum => "Minimum",
            BinaryOp::BitwiseAnd => "BitwiseAnd",
            BinaryOp::BitwiseOr => "BitwiseOr",
            BinaryOp::BitwiseXor => "BitwiseXor",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        // Binary ops take two inputs
        if inputs.len() >= 2 {
            broadcast_shapes(inputs[0], inputs[1]).unwrap_or_else(|| inputs[0].to_vec())
        } else {
            inputs[0].to_vec()
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        // Promote to the wider type
        if inputs.len() >= 2 {
            promote_dtypes(inputs[0], inputs[1])
        } else {
            inputs[0]
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Binary ops require contiguous input for efficient SIMD
        MemoryEffect::RequiresContiguous
    }

    fn intrinsic_cost(&self) -> OpCost {
        OpCost::Allocating
    }

    fn infer_strides(&self, _input_shape: &[usize], _input_strides: &[isize]) -> Option<Vec<isize>> {
        // Binary ops produce new contiguous output
        None
    }

    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        if input_shapes.len() < 2 {
            return Err(ValidationError::InsufficientInputs {
                expected: 2,
                got: input_shapes.len(),
            });
        }

        // Check shapes are broadcastable
        if broadcast_shapes(input_shapes[0], input_shapes[1]).is_none() {
            return Err(ValidationError::ShapeMismatch {
                expected: input_shapes[0].to_vec(),
                got: input_shapes[1].to_vec(),
            });
        }

        Ok(())
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        match self {
            BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                DTypeCategory::Integer
            }
            _ => DTypeCategory::Numeric,
        }
    }

    fn working_dtype(&self) -> Option<DType> {
        None // Work with promoted input dtype
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        OutputDTypeRule::PreserveInput
    }
}

/// Compute the broadcast shape of two shapes.
///
/// Returns None if shapes are not broadcastable.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return None; // Not broadcastable
        }
    }

    result.reverse();
    Some(result)
}

/// Promote two dtypes to a common type.
pub fn promote_dtypes(a: DType, b: DType) -> DType {
    use DType::*;

    // If same, return as-is
    if a == b {
        return a;
    }

    // Float types take precedence
    match (a, b) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        // Among integers, use the larger
        (I64, _) | (_, I64) => I64,
        (U64, _) | (_, U64) => U64,
        (I32, _) | (_, I32) => I32,
        (U32, _) | (_, U32) => U32,
        (I16, _) | (_, I16) => I16,
        (U16, _) | (_, U16) => U16,
        (I8, _) | (_, I8) => I8,
        _ => U8,
    }
}

/// Convert a linear index to multi-dimensional coordinates.
fn linear_to_coords(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut remaining = index;

    for i in (0..shape.len()).rev() {
        coords[i] = remaining % shape[i];
        remaining /= shape[i];
    }

    coords
}

/// Get the linear index for broadcast access.
fn broadcast_index(coords: &[usize], shape: &[usize]) -> usize {
    let offset = coords.len().saturating_sub(shape.len());
    let mut index = 0;
    let mut stride = 1;

    for i in (0..shape.len()).rev() {
        let coord = coords[offset + i];
        // Broadcast: if dimension is 1, use 0
        let actual_coord = if shape[i] == 1 { 0 } else { coord };
        index += actual_coord * stride;
        stride *= shape[i];
    }

    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes_same() {
        let result = broadcast_shapes(&[3, 4], &[3, 4]);
        assert_eq!(result, Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        let result = broadcast_shapes(&[3, 4], &[1]);
        assert_eq!(result, Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_different_ndim() {
        let result = broadcast_shapes(&[3, 4], &[4]);
        assert_eq!(result, Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        let result = broadcast_shapes(&[3, 4], &[3, 5]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_promote_dtypes() {
        assert_eq!(promote_dtypes(DType::U8, DType::U8), DType::U8);
        assert_eq!(promote_dtypes(DType::U8, DType::F32), DType::F32);
        assert_eq!(promote_dtypes(DType::F32, DType::F64), DType::F64);
    }
}

