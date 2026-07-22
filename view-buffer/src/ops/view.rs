//! View operations that perform zero-copy transformations.

use crate::core::dtype::OutputDTypeRule;
use crate::ops::cost::OpCost;
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
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
    /// Extracts a single channel from a multi-channel [H, W, C] buffer,
    /// producing a 2D [H, W] result. Zero-copy via offset + dimension drop.
    ChannelSelect { index: usize },
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
            ViewOp::ChannelSelect { .. } => "ChannelSelect",
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
            ViewOp::ChannelSelect { .. } => {
                // [H, W, C] → [H, W]
                if input_shape.len() == 3 {
                    vec![input_shape[0], input_shape[1]]
                } else {
                    input_shape.to_vec()
                }
            }
        }
    }

    fn output_rank_rule(&self) -> OutputRankRule {
        match self {
            // Selecting a channel drops the trailing channel dimension.
            ViewOp::ChannelSelect { .. } => OutputRankRule::ReduceByOne,
            // Reshape's rank is structural: the *count* of target dims is
            // known at plan time even when individual entries are per-row
            // expressions (bound to placeholder values for introspection).
            ViewOp::Reshape(shape) => OutputRankRule::Fixed(shape.len()),
            // Transpose/flip/crop/rotate all keep the rank.
            ViewOp::Transpose(_)
            | ViewOp::Flip(_)
            | ViewOp::Crop { .. }
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270 => OutputRankRule::PreserveRank,
        }
    }

    fn output_channel_rule(&self) -> OutputChannelRule {
        match self {
            // ChannelSelect collapses to a single 2-D plane (no channel dim).
            ViewOp::ChannelSelect { .. } => OutputChannelRule::NotApplicable,
            // Transpose can move the channel axis; reshape is arbitrary; crop can
            // slice the channel dimension itself — none are declarable up front.
            ViewOp::Transpose(_) | ViewOp::Reshape(_) | ViewOp::Crop { .. } => {
                OutputChannelRule::Unknown
            }
            // Flip/rotate preserve the channel dimension.
            ViewOp::Flip(_) | ViewOp::Rotate90 | ViewOp::Rotate180 | ViewOp::Rotate270 => {
                OutputChannelRule::PreserveChannels
            }
        }
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        // View ops (transpose, reshape, flip, crop, channel_select, rotate90)
        // only rearrange existing elements — the dtype is always preserved.
        OutputDTypeRule::PreserveInput
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
            ViewOp::Flip(axes) => {
                let mut new_strides = input_strides.to_vec();
                for &axis in axes {
                    new_strides[axis] = -new_strides[axis];
                }
                Some(new_strides)
            }
            ViewOp::Crop { .. } => Some(input_strides.to_vec()),
            ViewOp::Rotate90 => {
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
                if input_strides.len() >= 2 {
                    let mut new_strides = input_strides.to_vec();
                    new_strides.swap(0, 1);
                    new_strides[0] = -new_strides[0];
                    Some(new_strides)
                } else {
                    Some(input_strides.to_vec())
                }
            }
            ViewOp::ChannelSelect { .. } => {
                // Drop the last stride dimension (channel axis)
                if input_strides.len() >= 3 {
                    Some(input_strides[..2].to_vec())
                } else {
                    Some(input_strides.to_vec())
                }
            }
        }
    }
}
