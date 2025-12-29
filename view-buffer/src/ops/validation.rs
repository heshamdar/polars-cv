//! Validation framework for operation requirements.
//!
//! Provides plan-time validation of shape and dtype constraints,
//! allowing invalid pipelines to be rejected before execution.

use crate::core::dtype::DType;
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Errors that can occur during operation validation.
#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ValidationError {
    /// Shape requirement not met.
    #[error("Shape requirement: {requirement}. Got shape {got:?}")]
    ShapeRequirement {
        requirement: &'static str,
        got: Vec<usize>,
    },

    /// DType requirement not met.
    #[error("DType requirement: expected one of {expected:?}, got {got:?}")]
    DTypeRequirement { expected: Vec<DType>, got: DType },

    /// Rank requirement not met.
    #[error("Rank requirement: expected {expected}, got {got}")]
    RankRequirement { expected: &'static str, got: usize },

    /// Generic validation error.
    #[error("Validation failed: {message}")]
    Generic { message: String },

    /// Insufficient inputs for operation.
    #[error("Operation requires {expected} inputs, got {got}")]
    InsufficientInputs { expected: usize, got: usize },

    /// Shape mismatch between inputs.
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Invalid axis for operation.
    #[error("Invalid axis {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    /// Invalid parameter value.
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParameter { param: String, reason: String },
}

// --- Shape Predicates ---

/// Checks if shape is 2D-like: either HW (rank 2) or HW1 (rank 3 with C=1).
///
/// This is the required shape for operations like Normalize that operate
/// on single-channel or grayscale data.
pub fn is_2d_like(shape: &[usize]) -> bool {
    match shape.len() {
        2 => true,
        3 => shape[2] == 1,
        _ => false,
    }
}

/// Checks if shape is image-like: HWC with C in {1, 3, 4}.
///
/// Supports grayscale (1), RGB (3), and RGBA (4) channel layouts.
pub fn is_image_like(shape: &[usize]) -> bool {
    shape.len() == 3 && matches!(shape[2], 1 | 3 | 4)
}

/// Checks if shape has exactly the specified rank.
pub fn has_rank(shape: &[usize], rank: usize) -> bool {
    shape.len() == rank
}

/// Checks if shape has rank within the specified range (inclusive).
pub fn has_rank_range(shape: &[usize], min: usize, max: usize) -> bool {
    shape.len() >= min && shape.len() <= max
}

// --- DType Predicates ---

/// Checks if dtype is a floating-point type.
pub fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
}

/// Checks if dtype is an integer type.
pub fn is_integer_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::U8
            | DType::I8
            | DType::U16
            | DType::I16
            | DType::U32
            | DType::I32
            | DType::U64
            | DType::I64
    )
}

/// Checks if dtype is unsigned.
pub fn is_unsigned_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::U8 | DType::U16 | DType::U32 | DType::U64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_2d_like() {
        assert!(is_2d_like(&[100, 100]));
        assert!(is_2d_like(&[100, 100, 1]));
        assert!(!is_2d_like(&[100, 100, 3]));
        assert!(!is_2d_like(&[100]));
        assert!(!is_2d_like(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_is_image_like() {
        assert!(is_image_like(&[100, 100, 1]));
        assert!(is_image_like(&[100, 100, 3]));
        assert!(is_image_like(&[100, 100, 4]));
        assert!(!is_image_like(&[100, 100, 2]));
        assert!(!is_image_like(&[100, 100]));
    }

    #[test]
    fn test_dtype_predicates() {
        assert!(is_float_dtype(DType::F32));
        assert!(is_float_dtype(DType::F64));
        assert!(!is_float_dtype(DType::U8));

        assert!(is_integer_dtype(DType::U8));
        assert!(is_integer_dtype(DType::I32));
        assert!(!is_integer_dtype(DType::F32));

        assert!(is_unsigned_dtype(DType::U8));
        assert!(!is_unsigned_dtype(DType::I8));
    }
}
