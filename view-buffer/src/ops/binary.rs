//! Binary operations between two arrays.
//!
//! This module provides element-wise operations between two ViewBuffers,
//! including arithmetic operations and bitwise operations for mask manipulation.
//!
//! # Operation Semantics
//!
//! Operations have type-dependent semantics to match common library expectations:
//!
//! ## For integer types (u8, u16):
//! - `Add`/`Subtract`: Saturating arithmetic (clamps to valid range)
//! - `Multiply`: Saturating multiplication (clamps to max value)
//! - `Blend`: Normalized multiplication ((a/max) * (b/max) * max)
//! - `Divide`: Integer division with zero protection
//! - `Ratio`: Scaled division ((a/b) * max, clamped)
//!
//! ## For float types (f32, f64):
//! - All operations use standard IEEE 754 arithmetic

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Binary operations between two arrays.
///
/// All operations are element-wise and support broadcasting.
/// The output shape is the broadcast result of both input shapes.
///
/// Operations have type-dependent semantics:
/// - For `u8`/`u16`: Image-processing semantics (saturating, normalized)
/// - For `f32`/`f64`: Standard numerical semantics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BinaryOp {
    /// Element-wise addition.
    ///
    /// For u8/u16: Saturating addition (clamps to max value).
    /// For f32/f64: Standard addition.
    Add,
    /// Element-wise subtraction.
    ///
    /// For u8/u16: Saturating subtraction (clamps to 0).
    /// For f32/f64: Standard subtraction.
    Subtract,
    /// Element-wise multiplication.
    ///
    /// For u8/u16: Saturating multiplication (clamps to max value).
    /// For f32/f64: Standard multiplication.
    Multiply,
    /// Normalized blend (element-wise).
    ///
    /// For u8: (a/255) * (b/255) * 255
    /// For u16: (a/65535) * (b/65535) * 65535
    /// For f32/f64: Standard multiplication (same as Multiply).
    Blend,
    /// Element-wise division.
    ///
    /// For u8/u16: Integer division with zero protection (returns 0).
    /// For f32/f64: Standard division.
    Divide,
    /// Scaled ratio division.
    ///
    /// For u8: (a/b) * 255, clamped to [0, 255]
    /// For u16: (a/b) * 65535, clamped to [0, 65535]
    /// For f32/f64: Standard division (same as Divide).
    Ratio,
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
    /// The operation semantics depend on the data type:
    /// - For u8/u16: Image-processing semantics (saturating, normalized)
    /// - For f32/f64: Standard numerical semantics
    pub fn execute(&self, a: &ViewBuffer, b: &ViewBuffer) -> ViewBuffer {
        // Validate shapes are broadcastable
        let output_shape =
            broadcast_shapes(a.shape(), b.shape()).expect("Shapes must be broadcastable");

        match (a.dtype(), b.dtype()) {
            (DType::U8, DType::U8) => self.execute_u8(a, b, &output_shape),
            (DType::U16, DType::U16) => self.execute_u16(a, b, &output_shape),
            (DType::F32, DType::F32) => self.execute_f32(a, b, &output_shape),
            (DType::F64, DType::F64) => self.execute_f64(a, b, &output_shape),
            // For mixed types, promote to the wider type
            _ => {
                // For now, cast to f32 for mixed types
                let a_f32 = a.cast_to(DType::F32);
                let b_f32 = b.cast_to(DType::F32);
                self.execute_f32(&a_f32, &b_f32, &output_shape)
            }
        }
    }

    /// Execute operation on u8 buffers with image-processing semantics.
    fn execute_u8(&self, a: &ViewBuffer, b: &ViewBuffer, output_shape: &[usize]) -> ViewBuffer {
        let total_elements: usize = output_shape.iter().product();
        let mut output = vec![0u8; total_elements];

        let a_contig = a.to_contiguous();
        let b_contig = b.to_contiguous();
        let a_data = a_contig.as_slice::<u8>();
        let b_data = b_contig.as_slice::<u8>();

        let same_shape = a.shape() == b.shape() && a.shape() == output_shape;

        for (i, out) in output.iter_mut().enumerate() {
            let (a_val, b_val) = if same_shape {
                (a_data[i], b_data[i])
            } else {
                let coords = linear_to_coords(i, output_shape);
                let a_idx = broadcast_index(&coords, a.shape());
                let b_idx = broadcast_index(&coords, b.shape());
                (a_data[a_idx], b_data[b_idx])
            };

            *out = match self {
                BinaryOp::Add => a_val.saturating_add(b_val),
                BinaryOp::Subtract => a_val.saturating_sub(b_val),
                BinaryOp::Multiply => {
                    // Saturating multiply: clamp to 255
                    let result = (a_val as u16) * (b_val as u16);
                    if result > 255 {
                        255
                    } else {
                        result as u8
                    }
                }
                BinaryOp::Blend => {
                    // Normalized blend: (a/255) * (b/255) * 255
                    // = (a * b) / 255
                    let product = (a_val as u32) * (b_val as u32);
                    // Use rounding division: (product + 127) / 255
                    ((product + 127) / 255) as u8
                }
                BinaryOp::Divide => {
                    // Integer division with zero protection
                    if b_val == 0 {
                        0
                    } else {
                        a_val / b_val
                    }
                }
                BinaryOp::Ratio => {
                    // Scaled ratio: (a/b) * 255, clamped
                    if b_val == 0 {
                        if a_val == 0 {
                            0
                        } else {
                            255 // a/0 where a > 0 saturates to max
                        }
                    } else {
                        let ratio = (a_val as u32) * 255 / (b_val as u32);
                        if ratio > 255 {
                            255
                        } else {
                            ratio as u8
                        }
                    }
                }
                BinaryOp::Maximum => a_val.max(b_val),
                BinaryOp::Minimum => a_val.min(b_val),
                BinaryOp::BitwiseAnd => a_val & b_val,
                BinaryOp::BitwiseOr => a_val | b_val,
                BinaryOp::BitwiseXor => a_val ^ b_val,
            };
        }

        ViewBuffer::from_vec_with_shape(output, output_shape.to_vec())
    }

    /// Execute operation on u16 buffers with image-processing semantics.
    fn execute_u16(&self, a: &ViewBuffer, b: &ViewBuffer, output_shape: &[usize]) -> ViewBuffer {
        let total_elements: usize = output_shape.iter().product();
        let mut output = vec![0u16; total_elements];

        let a_contig = a.to_contiguous();
        let b_contig = b.to_contiguous();
        let a_data = a_contig.as_slice::<u16>();
        let b_data = b_contig.as_slice::<u16>();

        let same_shape = a.shape() == b.shape() && a.shape() == output_shape;

        for (i, out) in output.iter_mut().enumerate() {
            let (a_val, b_val) = if same_shape {
                (a_data[i], b_data[i])
            } else {
                let coords = linear_to_coords(i, output_shape);
                let a_idx = broadcast_index(&coords, a.shape());
                let b_idx = broadcast_index(&coords, b.shape());
                (a_data[a_idx], b_data[b_idx])
            };

            *out = match self {
                BinaryOp::Add => a_val.saturating_add(b_val),
                BinaryOp::Subtract => a_val.saturating_sub(b_val),
                BinaryOp::Multiply => {
                    // Saturating multiply: clamp to 65535
                    let result = (a_val as u32) * (b_val as u32);
                    if result > 65535 {
                        65535
                    } else {
                        result as u16
                    }
                }
                BinaryOp::Blend => {
                    // Normalized blend: (a/65535) * (b/65535) * 65535
                    // = (a * b) / 65535
                    let product = (a_val as u64) * (b_val as u64);
                    // Use rounding division
                    ((product + 32767) / 65535) as u16
                }
                BinaryOp::Divide => {
                    // Integer division with zero protection
                    if b_val == 0 {
                        0
                    } else {
                        a_val / b_val
                    }
                }
                BinaryOp::Ratio => {
                    // Scaled ratio: (a/b) * 65535, clamped
                    if b_val == 0 {
                        if a_val == 0 {
                            0
                        } else {
                            65535
                        }
                    } else {
                        let ratio = (a_val as u64) * 65535 / (b_val as u64);
                        if ratio > 65535 {
                            65535
                        } else {
                            ratio as u16
                        }
                    }
                }
                BinaryOp::Maximum => a_val.max(b_val),
                BinaryOp::Minimum => a_val.min(b_val),
                BinaryOp::BitwiseAnd => a_val & b_val,
                BinaryOp::BitwiseOr => a_val | b_val,
                BinaryOp::BitwiseXor => a_val ^ b_val,
            };
        }

        ViewBuffer::from_vec_with_shape(output, output_shape.to_vec())
    }

    /// Execute operation on f32 buffers with standard numerical semantics.
    fn execute_f32(&self, a: &ViewBuffer, b: &ViewBuffer, output_shape: &[usize]) -> ViewBuffer {
        let total_elements: usize = output_shape.iter().product();
        let mut output = vec![0.0f32; total_elements];

        let a_contig = a.to_contiguous();
        let b_contig = b.to_contiguous();
        let a_data = a_contig.as_slice::<f32>();
        let b_data = b_contig.as_slice::<f32>();

        let same_shape = a.shape() == b.shape() && a.shape() == output_shape;

        for (i, out) in output.iter_mut().enumerate() {
            let (a_val, b_val) = if same_shape {
                (a_data[i], b_data[i])
            } else {
                let coords = linear_to_coords(i, output_shape);
                let a_idx = broadcast_index(&coords, a.shape());
                let b_idx = broadcast_index(&coords, b.shape());
                (a_data[a_idx], b_data[b_idx])
            };

            *out = match self {
                BinaryOp::Add => a_val + b_val,
                BinaryOp::Subtract => a_val - b_val,
                BinaryOp::Multiply | BinaryOp::Blend => a_val * b_val,
                BinaryOp::Divide | BinaryOp::Ratio => {
                    if b_val == 0.0 {
                        0.0
                    } else {
                        a_val / b_val
                    }
                }
                BinaryOp::Maximum => a_val.max(b_val),
                BinaryOp::Minimum => a_val.min(b_val),
                BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                    // For floats, truncate to i64 for bitwise ops
                    let a_int = a_val as i64;
                    let b_int = b_val as i64;
                    let result = match self {
                        BinaryOp::BitwiseAnd => a_int & b_int,
                        BinaryOp::BitwiseOr => a_int | b_int,
                        BinaryOp::BitwiseXor => a_int ^ b_int,
                        _ => unreachable!(),
                    };
                    result as f32
                }
            };
        }

        ViewBuffer::from_vec_with_shape(output, output_shape.to_vec())
    }

    /// Execute operation on f64 buffers with standard numerical semantics.
    fn execute_f64(&self, a: &ViewBuffer, b: &ViewBuffer, output_shape: &[usize]) -> ViewBuffer {
        let total_elements: usize = output_shape.iter().product();
        let mut output = vec![0.0f64; total_elements];

        let a_contig = a.to_contiguous();
        let b_contig = b.to_contiguous();
        let a_data = a_contig.as_slice::<f64>();
        let b_data = b_contig.as_slice::<f64>();

        let same_shape = a.shape() == b.shape() && a.shape() == output_shape;

        for (i, out) in output.iter_mut().enumerate() {
            let (a_val, b_val) = if same_shape {
                (a_data[i], b_data[i])
            } else {
                let coords = linear_to_coords(i, output_shape);
                let a_idx = broadcast_index(&coords, a.shape());
                let b_idx = broadcast_index(&coords, b.shape());
                (a_data[a_idx], b_data[b_idx])
            };

            *out = match self {
                BinaryOp::Add => a_val + b_val,
                BinaryOp::Subtract => a_val - b_val,
                BinaryOp::Multiply | BinaryOp::Blend => a_val * b_val,
                BinaryOp::Divide | BinaryOp::Ratio => {
                    if b_val == 0.0 {
                        0.0
                    } else {
                        a_val / b_val
                    }
                }
                BinaryOp::Maximum => a_val.max(b_val),
                BinaryOp::Minimum => a_val.min(b_val),
                BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                    // For floats, truncate to i64 for bitwise ops
                    let a_int = a_val as i64;
                    let b_int = b_val as i64;
                    let result = match self {
                        BinaryOp::BitwiseAnd => a_int & b_int,
                        BinaryOp::BitwiseOr => a_int | b_int,
                        BinaryOp::BitwiseXor => a_int ^ b_int,
                        _ => unreachable!(),
                    };
                    result as f64
                }
            };
        }

        ViewBuffer::from_vec_with_shape(output, output_shape.to_vec())
    }
}

impl Op for BinaryOp {
    fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "Add",
            BinaryOp::Subtract => "Subtract",
            BinaryOp::Multiply => "Multiply",
            BinaryOp::Blend => "Blend",
            BinaryOp::Divide => "Divide",
            BinaryOp::Ratio => "Ratio",
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

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
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

    #[test]
    fn test_u8_saturating_add() {
        let a = ViewBuffer::from_vec_with_shape(vec![200u8, 100, 50], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![100u8, 50, 10], vec![3]);
        let result = BinaryOp::Add.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 255); // 200 + 100 = 255 (saturated)
        assert_eq!(data[1], 150); // 100 + 50 = 150
        assert_eq!(data[2], 60); // 50 + 10 = 60
    }

    #[test]
    fn test_u8_saturating_subtract() {
        let a = ViewBuffer::from_vec_with_shape(vec![50u8, 100, 200], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![100u8, 50, 50], vec![3]);
        let result = BinaryOp::Subtract.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 0); // 50 - 100 = 0 (saturated)
        assert_eq!(data[1], 50); // 100 - 50 = 50
        assert_eq!(data[2], 150); // 200 - 50 = 150
    }

    #[test]
    fn test_u8_saturating_multiply() {
        let a = ViewBuffer::from_vec_with_shape(vec![10u8, 16, 20], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![10u8, 16, 20], vec![3]);
        let result = BinaryOp::Multiply.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 100); // 10 * 10 = 100
        assert_eq!(data[1], 255); // 16 * 16 = 256 -> 255 (saturated)
        assert_eq!(data[2], 255); // 20 * 20 = 400 -> 255 (saturated)
    }

    #[test]
    fn test_u8_blend() {
        let a = ViewBuffer::from_vec_with_shape(vec![255u8, 128, 0], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![255u8, 128, 255], vec![3]);
        let result = BinaryOp::Blend.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 255); // (255/255) * (255/255) * 255 = 255
        assert_eq!(data[1], 64); // (128/255) * (128/255) * 255 ≈ 64
        assert_eq!(data[2], 0); // (0/255) * (255/255) * 255 = 0
    }

    #[test]
    fn test_u8_ratio() {
        let a = ViewBuffer::from_vec_with_shape(vec![128u8, 64, 255], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![64u8, 128, 255], vec![3]);
        let result = BinaryOp::Ratio.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 255); // (128/64) * 255 = 510 -> 255 (clamped)
        assert_eq!(data[1], 127); // (64/128) * 255 = 127.5 -> 127
        assert_eq!(data[2], 255); // (255/255) * 255 = 255
    }

    #[test]
    fn test_f32_standard_arithmetic() {
        let a = ViewBuffer::from_vec_with_shape(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![0.5f32, 0.5, 0.5], vec![3]);

        let add_result = BinaryOp::Add.execute(&a, &b);
        let add_data = add_result.as_slice::<f32>();
        assert!((add_data[0] - 1.5).abs() < 1e-6);

        let mul_result = BinaryOp::Multiply.execute(&a, &b);
        let mul_data = mul_result.as_slice::<f32>();
        assert!((mul_data[0] - 0.5).abs() < 1e-6);
        assert!((mul_data[1] - 1.0).abs() < 1e-6);
    }
}
