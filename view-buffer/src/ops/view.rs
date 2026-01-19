//! View operations that perform zero-copy transformations.

use crate::core::dtype::DType;
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// View operations that modify layout without copying data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ViewOp {
    /// Permutes dimensions according to the given order.
    Transpose(Vec<usize>),
    /// Reshapes to a new shape (requires contiguous input).
    Reshape(Vec<usize>),
    /// Flips along the specified axes.
    Flip(Vec<usize>),
    /// Crops to a region defined by start and end indices.
    Crop { start: Vec<usize>, end: Vec<usize> },
    /// Rotates 90 degrees clockwise (zero-copy via transpose + flip).
    Rotate90,
    /// Rotates 180 degrees (zero-copy via double flip).
    Rotate180,
    /// Rotates 270 degrees clockwise / 90 degrees counter-clockwise (zero-copy via transpose + flip).
    Rotate270,
}

impl Op for ViewOp {
    fn name(&self) -> &'static str {
        match self {
            ViewOp::Transpose(_) => "Transpose",
            ViewOp::Reshape(_) => "Reshape",
            ViewOp::Flip(_) => "Flip",
            ViewOp::Crop { .. } => "Crop",
            ViewOp::Rotate90 => "Rotate90",
            ViewOp::Rotate180 => "Rotate180",
            ViewOp::Rotate270 => "Rotate270",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_shape = inputs[0];
        match self {
            ViewOp::Transpose(perm) => perm.iter().map(|&i| input_shape[i]).collect(),
            ViewOp::Reshape(new_shape) => new_shape.clone(),
            ViewOp::Flip(_) => input_shape.to_vec(),
            ViewOp::Crop { start, end } => {
                start.iter().zip(end.iter()).map(|(s, e)| e - s).collect()
            }
            ViewOp::Rotate90 | ViewOp::Rotate270 => {
                // For 2D images [H, W] or [H, W, C], swap H and W
                if input_shape.len() >= 2 {
                    let mut new_shape = input_shape.to_vec();
                    new_shape.swap(0, 1);
                    new_shape
                } else {
                    input_shape.to_vec()
                }
            }
            ViewOp::Rotate180 => input_shape.to_vec(),
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        inputs[0]
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::View
    }

    fn intrinsic_cost(&self) -> OpCost {
        OpCost::ZeroCopy
    }

    fn infer_strides(&self, _input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match self {
            ViewOp::Transpose(perm) => Some(perm.iter().map(|&i| input_strides[i]).collect()),
            ViewOp::Reshape(_new_shape) => {
                // Reshape as a view operation defers stride calculation to runtime/planner
                // since we need to verify contiguity with the actual DType.
                // Both contiguous and non-contiguous cases return None here.
                None
            }
            ViewOp::Flip(_) => {
                let axes = match self {
                    ViewOp::Flip(a) => a,
                    _ => unreachable!(),
                };
                let mut new_strides = input_strides.to_vec();
                for &axis in axes {
                    new_strides[axis] = -new_strides[axis];
                }
                Some(new_strides)
            }
            ViewOp::Crop { .. } => Some(input_strides.to_vec()),
            ViewOp::Rotate90 => {
                // Rotate90: transpose [0,1] then flip axis 1
                // Stride calculation: swap strides[0] and strides[1], then negate strides[1]
                if input_strides.len() >= 2 {
                    let mut new_strides = input_strides.to_vec();
                    new_strides.swap(0, 1);
                    new_strides[1] = -new_strides[1];
                    Some(new_strides)
                } else {
                    Some(input_strides.to_vec())
                }
            }
            ViewOp::Rotate180 => {
                // Rotate180: flip both axes 0 and 1
                if input_strides.len() >= 2 {
                    let mut new_strides = input_strides.to_vec();
                    new_strides[0] = -new_strides[0];
                    new_strides[1] = -new_strides[1];
                    Some(new_strides)
                } else {
                    Some(input_strides.to_vec())
                }
            }
            ViewOp::Rotate270 => {
                // Rotate270: transpose [0,1] then flip axis 0
                // Stride calculation: swap strides[0] and strides[1], then negate strides[0]
                if input_strides.len() >= 2 {
                    let mut new_strides = input_strides.to_vec();
                    new_strides.swap(0, 1);
                    new_strides[0] = -new_strides[0];
                    Some(new_strides)
                } else {
                    Some(input_strides.to_vec())
                }
            }
        }
    }
}
