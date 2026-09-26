//! Validation framework for operation requirements.
//!
//! Provides plan-time validation of shape and dtype constraints,
//! allowing invalid pipelines to be rejected before execution.

use crate::core::dtype::{DType, PlannedDType};
use crate::ops::shape_rule::{known_dims, show_dims, Dim};
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Why an operation cannot run on its inputs. Every variant is a verdict on
/// a known fact (see [`Op::validate`](crate::ops::traits::Op::validate)).
#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ValidationError {
    /// Shape requirement not met.
    #[error("Shape requirement: {requirement}. Got shape {}", show_dims(.got))]
    ShapeRequirement {
        requirement: &'static str,
        got: Vec<Dim>,
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

    /// Invalid axis for operation.
    #[error("Invalid axis {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    /// Invalid parameter value.
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParameter { param: String, reason: String },

    /// An axis list that must reorder every input axis exactly once does not.
    #[error("axes {axes:?} are not a permutation of the {ndim} input axes")]
    NotAPermutation { axes: Vec<usize>, ndim: usize },
}

/// [`Op::validate`](crate::ops::traits::Op::validate) over inputs whose every
/// size and dtype is known — the executor's call, before an op runs on a row.
pub fn validate_concrete(
    op: &dyn crate::ops::traits::Op,
    input_shapes: &[&[usize]],
    input_dtypes: &[DType],
) -> Result<(), ValidationError> {
    let shapes: Vec<Vec<Dim>> = input_shapes.iter().map(|s| known_dims(s)).collect();
    let shapes: Vec<&[Dim]> = shapes.iter().map(Vec::as_slice).collect();
    let dtypes: Vec<PlannedDType> = input_dtypes
        .iter()
        .map(|&d| PlannedDType::Known(d))
        .collect();
    op.validate(&shapes, &dtypes)
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

/// `[H, W]` or `[H, W, C]`: the layout the image kernels index directly.
pub fn require_hw_or_hwc(shape: &[Dim]) -> Result<(), ValidationError> {
    if matches!(shape.len(), 2 | 3) {
        Ok(())
    } else {
        Err(ValidationError::ShapeRequirement {
            requirement: "a [H, W] or [H, W, C] buffer",
            got: shape.to_vec(),
        })
    }
}

/// At least `[H, W]`: kernels that read the first two axes as height and width.
pub fn require_spatial(shape: &[Dim]) -> Result<(), ValidationError> {
    if shape.len() >= 2 {
        Ok(())
    } else {
        Err(ValidationError::ShapeRequirement {
            requirement: "at least two dimensions [H, W, ...]",
            got: shape.to_vec(),
        })
    }
}

/// `[H, W]` or `[H, W, 1]`: kernels defined on one channel. An unknown
/// channel count is not refused.
pub fn require_single_channel(shape: &[Dim]) -> Result<(), ValidationError> {
    let ok = match shape {
        [_, _] => true,
        [_, _, c] => c.known().is_none_or(|c| c == 1),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(ValidationError::ShapeRequirement {
            requirement: "single-channel input [H, W] or [H, W, 1]; \
                          use .grayscale() or .channel_select() first",
            got: shape.to_vec(),
        })
    }
}

/// At least `[H, W, C]` with `C >= min_channels` in axis 2. An unknown channel
/// count is not refused.
pub fn require_channels_at_least(
    shape: &[Dim],
    min_channels: usize,
    requirement: &'static str,
) -> Result<(), ValidationError> {
    let ok = shape.len() >= 3 && shape[2].known().is_none_or(|c| c >= min_channels);
    if ok {
        Ok(())
    } else {
        Err(ValidationError::ShapeRequirement {
            requirement,
            got: shape.to_vec(),
        })
    }
}

/// Every index in `axes` names an axis of `shape`.
pub fn require_axes(shape: &[Dim], axes: &[usize]) -> Result<(), ValidationError> {
    match axes.iter().find(|&&a| a >= shape.len()) {
        Some(&axis) => Err(ValidationError::InvalidAxis {
            axis,
            ndim: shape.len(),
        }),
        None => Ok(()),
    }
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
