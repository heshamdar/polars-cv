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
//! - `Divide`/`Ratio`: True division — integer operands promote to float and
//!   `a / b` is computed in float (zero divisor yields 0), so the result dtype
//!   is `f32` (or `f64` when an operand is already `f64`).
//!
//! ## For float types (f32, f64):
//! - All operations use standard IEEE 754 arithmetic

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
///
/// Operations have type-dependent semantics:
/// - For `u8`/`u16`: Image-processing semantics (saturating, normalized)
/// - For `f32`/`f64`: Standard numerical semantics
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Element-wise division (true division).
    ///
    /// Integer operands promote to float; `a / b` is computed in float with zero
    /// protection (returns 0 when the divisor is 0). Output dtype is `f32` (or
    /// `f64` when an operand is already `f64`).
    Divide,
    /// Element-wise ratio (true division).
    ///
    /// Currently identical to [`Divide`](BinaryOp::Divide): integer operands
    /// promote to float and `a / b` is computed in float with zero protection.
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

        // The result dtype comes from the single authority shared with planning
        // (`output_dtype`). Computing it here lets execution match the plan-time
        // schema by construction.
        let out_dtype = self.output_dtype(a.dtype(), b.dtype());

        // Divide and Ratio use *true division*: integer operands promote to
        // float so `a / b` is computed in float rather than truncated. Route
        // them through the float path even when both inputs are the same int.
        let force_float = matches!(self, BinaryOp::Divide | BinaryOp::Ratio);

        let result = match (a.dtype(), b.dtype()) {
            (DType::U8, DType::U8) if !force_float => self.execute_u8(a, b, &output_shape),
            (DType::U16, DType::U16) if !force_float => self.execute_u16(a, b, &output_shape),
            (DType::F64, DType::F64) => self.execute_float::<f64>(a, b, &output_shape),
            (DType::F32, DType::F32) => self.execute_float::<f32>(a, b, &output_shape),
            // Mixed dtypes, or an integer pair whose declared output is float
            // (true division): promote both operands to the float compute type.
            _ => {
                let compute = if out_dtype == DType::F64 {
                    DType::F64
                } else {
                    DType::F32
                };
                let a_c = a.cast_to(compute);
                let b_c = b.cast_to(compute);
                match compute {
                    DType::F64 => self.execute_float::<f64>(&a_c, &b_c, &output_shape),
                    _ => self.execute_float::<f32>(&a_c, &b_c, &output_shape),
                }
            }
        };

        // Guarantee the produced buffer carries exactly the authority dtype
        // (e.g. an int+int promotion that computed in f32 is cast back down).
        if result.dtype() == out_dtype {
            result
        } else {
            result.cast_to(out_dtype)
        }
    }

    /// The output dtype of this binary op for the given operand dtypes.
    ///
    /// This is the single authority shared by planning (the `binary_output_dtype`
    /// FFI) and execution ([`execute`](BinaryOp::execute)). Divide and Ratio use
    /// *true division*: integer operands promote to float (`F32`, or `F64` when an
    /// operand is already `F64`), matching numpy-style semantics. All other ops
    /// use standard numeric promotion of the two operands.
    pub fn output_dtype(&self, left: DType, right: DType) -> DType {
        let promoted = promote_dtypes(left, right);
        match self {
            BinaryOp::Divide | BinaryOp::Ratio => to_float(promoted),
            _ => promoted,
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

        if same_shape {
            // Fast path: same shapes, process in chunks for better auto-vectorization
            Self::execute_u8_same_shape(self, a_data, b_data, &mut output);
        } else {
            // Broadcast path: element-by-element with coordinate mapping
            for (i, out) in output.iter_mut().enumerate() {
                let coords = linear_to_coords(i, output_shape);
                let a_idx = broadcast_index(&coords, a.shape());
                let b_idx = broadcast_index(&coords, b.shape());
                *out = self.apply_u8(a_data[a_idx], b_data[b_idx]);
            }
        }

        ViewBuffer::from_vec_with_shape(output, output_shape.to_vec())
    }

    /// Apply u8 binary operation on a single pair of values.
    #[inline(always)]
    fn apply_u8(&self, a_val: u8, b_val: u8) -> u8 {
        match self {
            BinaryOp::Add => a_val.saturating_add(b_val),
            BinaryOp::Subtract => a_val.saturating_sub(b_val),
            BinaryOp::Multiply => {
                let result = (a_val as u16) * (b_val as u16);
                if result > 255 {
                    255
                } else {
                    result as u8
                }
            }
            BinaryOp::Blend => {
                let product = (a_val as u32) * (b_val as u32);
                ((product + 127) / 255) as u8
            }
            BinaryOp::Divide => a_val.checked_div(b_val).unwrap_or(0),
            BinaryOp::Ratio => {
                if b_val == 0 {
                    if a_val == 0 {
                        0
                    } else {
                        255
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
        }
    }

    /// SIMD-friendly same-shape u8 operation using chunked processing.
    #[inline]
    fn execute_u8_same_shape(&self, a: &[u8], b: &[u8], output: &mut [u8]) {
        // Process in chunks of 32 for u8 (32 bytes = 256 bits = AVX2)
        const CHUNK: usize = 32;
        let len = output.len();
        let chunks = len / CHUNK;
        let remainder = len % CHUNK;

        for c in 0..chunks {
            let base = c * CHUNK;
            for j in 0..CHUNK {
                output[base + j] = self.apply_u8(a[base + j], b[base + j]);
            }
        }

        let rem_start = chunks * CHUNK;
        for j in 0..remainder {
            output[rem_start + j] = self.apply_u8(a[rem_start + j], b[rem_start + j]);
        }
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
                // Integer division with zero protection
                BinaryOp::Divide => a_val.checked_div(b_val).unwrap_or(0),
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

    /// Execute operation on float buffers with standard numerical semantics.
    ///
    /// Uses chunked processing for same-shape operations to enable SIMD
    /// auto-vectorization (e.g. 8 f32s = 256-bit AVX).
    fn execute_float<T>(&self, a: &ViewBuffer, b: &ViewBuffer, output_shape: &[usize]) -> ViewBuffer
    where
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + ViewType
            + num_traits::NumCast
            + 'static,
    {
        let total_elements: usize = output_shape.iter().product();
        let mut output = vec![T::default(); total_elements];

        let a_contig = a.to_contiguous();
        let b_contig = b.to_contiguous();
        let a_data = a_contig.as_slice::<T>();
        let b_data = b_contig.as_slice::<T>();

        let same_shape = a.shape() == b.shape() && a.shape() == output_shape;
        let zero: T = num_traits::NumCast::from(0.0f64).unwrap_or(T::default());

        if same_shape {
            // Fast path: chunked processing for SIMD auto-vectorization
            // Process in chunks of 8 (f32 x 8 = 256 bits = AVX, f64 x 4 = 256 bits)
            const CHUNK: usize = 8;
            let chunks = total_elements / CHUNK;
            let remainder = total_elements % CHUNK;

            // Simple operations get dedicated tight loops for best vectorization
            match self {
                BinaryOp::Add => {
                    for c in 0..chunks {
                        let base = c * CHUNK;
                        for j in 0..CHUNK {
                            output[base + j] = a_data[base + j] + b_data[base + j];
                        }
                    }
                    let rem = chunks * CHUNK;
                    for j in 0..remainder {
                        output[rem + j] = a_data[rem + j] + b_data[rem + j];
                    }
                }
                BinaryOp::Subtract => {
                    for c in 0..chunks {
                        let base = c * CHUNK;
                        for j in 0..CHUNK {
                            output[base + j] = a_data[base + j] - b_data[base + j];
                        }
                    }
                    let rem = chunks * CHUNK;
                    for j in 0..remainder {
                        output[rem + j] = a_data[rem + j] - b_data[rem + j];
                    }
                }
                BinaryOp::Multiply | BinaryOp::Blend => {
                    for c in 0..chunks {
                        let base = c * CHUNK;
                        for j in 0..CHUNK {
                            output[base + j] = a_data[base + j] * b_data[base + j];
                        }
                    }
                    let rem = chunks * CHUNK;
                    for j in 0..remainder {
                        output[rem + j] = a_data[rem + j] * b_data[rem + j];
                    }
                }
                _ => {
                    // Other ops: fall through to per-element
                    for (i, out) in output.iter_mut().enumerate() {
                        *out = self.apply_float(a_data[i], b_data[i], zero);
                    }
                }
            }
        } else {
            // Broadcast path: element-by-element with coordinate mapping
            for (i, out) in output.iter_mut().enumerate() {
                let coords = linear_to_coords(i, output_shape);
                let a_idx = broadcast_index(&coords, a.shape());
                let b_idx = broadcast_index(&coords, b.shape());
                *out = self.apply_float(a_data[a_idx], b_data[b_idx], zero);
            }
        }

        ViewBuffer::from_vec_with_shape(output, output_shape.to_vec())
    }

    /// Apply float binary operation on a single pair of values.
    #[inline(always)]
    fn apply_float<T>(&self, a_val: T, b_val: T, zero: T) -> T
    where
        T: Copy
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + num_traits::NumCast,
    {
        match self {
            BinaryOp::Add => a_val + b_val,
            BinaryOp::Subtract => a_val - b_val,
            BinaryOp::Multiply | BinaryOp::Blend => a_val * b_val,
            BinaryOp::Divide | BinaryOp::Ratio => {
                if b_val == zero {
                    zero
                } else {
                    a_val / b_val
                }
            }
            BinaryOp::Maximum => {
                if a_val > b_val {
                    a_val
                } else {
                    b_val
                }
            }
            BinaryOp::Minimum => {
                if a_val < b_val {
                    a_val
                } else {
                    b_val
                }
            }
            BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                let a_int: i64 = num_traits::NumCast::from(a_val).unwrap_or(0);
                let b_int: i64 = num_traits::NumCast::from(b_val).unwrap_or(0);
                let result = match self {
                    BinaryOp::BitwiseAnd => a_int & b_int,
                    BinaryOp::BitwiseOr => a_int | b_int,
                    BinaryOp::BitwiseXor => a_int ^ b_int,
                    _ => unreachable!(),
                };
                num_traits::NumCast::from(result).unwrap_or(zero)
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
        // Delegate to the single output-dtype authority so plan-time and
        // exec-time dtypes are computed by exactly one rule (true division for
        // Divide/Ratio, standard promotion otherwise).
        if inputs.len() >= 2 {
            self.output_dtype(inputs[0], inputs[1])
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

/// Promote an integer dtype to `F32` for true division; floats keep their width.
///
/// Used by [`BinaryOp::output_dtype`] for Divide/Ratio so the result of `a / b`
/// is a float regardless of the (integer) input types.
fn to_float(d: DType) -> DType {
    match d {
        DType::F64 => DType::F64,
        _ => DType::F32,
    }
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

use super::util::linear_to_coords;

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
    fn test_u8_ratio_is_true_division() {
        // Ratio now uses true division: integer operands promote to f32 and the
        // result is `a / b` (not the old scaled `(a/b) * 255`).
        let a = ViewBuffer::from_vec_with_shape(vec![128u8, 64, 255], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![64u8, 128, 255], vec![3]);
        let result = BinaryOp::Ratio.execute(&a, &b);
        assert_eq!(result.dtype(), DType::F32);
        let data = result.as_slice::<f32>();
        assert!((data[0] - 2.0).abs() < 1e-6); // 128 / 64
        assert!((data[1] - 0.5).abs() < 1e-6); // 64 / 128
        assert!((data[2] - 1.0).abs() < 1e-6); // 255 / 255
    }

    #[test]
    fn test_u8_divide_is_true_division() {
        // divide(u8, u8) promotes to f32 and computes true division, not the
        // truncating integer division it used to.
        let a = ViewBuffer::from_vec_with_shape(vec![130u8, 128, 1], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![64u8, 64, 0], vec![3]);
        let result = BinaryOp::Divide.execute(&a, &b);
        assert_eq!(result.dtype(), DType::F32);
        let data = result.as_slice::<f32>();
        assert!((data[0] - (130.0 / 64.0)).abs() < 1e-6); // ~2.031, not 2
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert_eq!(data[2], 0.0); // zero divisor protected
    }

    #[test]
    fn test_output_dtype_authority() {
        // Standard promotion for non-dividing ops.
        assert_eq!(BinaryOp::Add.output_dtype(DType::U8, DType::U8), DType::U8);
        assert_eq!(
            BinaryOp::Add.output_dtype(DType::U8, DType::U16),
            DType::U16
        );
        assert_eq!(
            BinaryOp::Add.output_dtype(DType::U8, DType::F32),
            DType::F32
        );
        // True division always lands on a float.
        assert_eq!(
            BinaryOp::Divide.output_dtype(DType::U8, DType::U8),
            DType::F32
        );
        assert_eq!(
            BinaryOp::Ratio.output_dtype(DType::U16, DType::U16),
            DType::F32
        );
        assert_eq!(
            BinaryOp::Divide.output_dtype(DType::F64, DType::F64),
            DType::F64
        );
        assert_eq!(
            BinaryOp::Divide.output_dtype(DType::U8, DType::F64),
            DType::F64
        );
    }

    #[test]
    fn test_mixed_dtype_add_matches_authority() {
        // A promoting integer add computes in float internally but the produced
        // buffer must carry the authority dtype (u16), matching planning.
        let a = ViewBuffer::from_vec_with_shape(vec![200u8, 100], vec![2]);
        let b = ViewBuffer::from_vec_with_shape(vec![400u16, 50], vec![2]);
        let result = BinaryOp::Add.execute(&a, &b);
        assert_eq!(result.dtype(), DType::U16);
        let data = result.as_slice::<u16>();
        assert_eq!(data[0], 600);
        assert_eq!(data[1], 150);
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
