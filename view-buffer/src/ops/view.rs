//! View operations that perform zero-copy transformations.

use crate::core::dtype::OutputDTypeRule;
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};

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
    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[crate::DType],
    ) -> Result<(), crate::ops::validation::ValidationError> {
        use crate::ops::validation::{require_axes, ValidationError};
        let shape = input_shapes[0];
        match self {
            ViewOp::Transpose(perm) => {
                let mut seen = vec![false; shape.len()];
                let is_permutation = perm.len() == shape.len()
                    && perm
                        .iter()
                        .all(|&a| a < seen.len() && !std::mem::replace(&mut seen[a], true));
                if is_permutation {
                    Ok(())
                } else {
                    Err(ValidationError::NotAPermutation {
                        axes: perm.clone(),
                        ndim: shape.len(),
                    })
                }
            }
            // A reshape that changes the element count would describe memory
            // the buffer does not own.
            ViewOp::Reshape(new) => {
                let (have, want) = (
                    shape.iter().product::<usize>(),
                    new.iter().product::<usize>(),
                );
                if have == want {
                    Ok(())
                } else {
                    Err(ValidationError::InvalidParameter {
                        param: "shape".to_string(),
                        reason: format!(
                            "cannot reshape {shape:?} ({have} elements) to {new:?} ({want} elements)"
                        ),
                    })
                }
            }
            ViewOp::Flip(axes) => require_axes(shape, axes),
            ViewOp::Crop { start, end } => {
                if start.len() < shape.len() || end.len() < shape.len() {
                    return Err(ValidationError::ShapeRequirement {
                        requirement: "crop bounds for every axis of the input",
                        got: shape.to_vec(),
                    });
                }
                // An `end` of `usize::MAX` is "to the end of this axis". Any
                // other bound past the axis is a window outside the input:
                // rejected rather than clamped, since clamping returns a
                // smaller region than the caller asked for (CR-42).
                for (axis, &dim) in shape.iter().enumerate() {
                    let (s, e) = (start[axis], end[axis]);
                    if s > dim || (e != usize::MAX && e > dim) {
                        let end_text = if e == usize::MAX {
                            "end".to_string()
                        } else {
                            e.to_string()
                        };
                        return Err(ValidationError::InvalidParameter {
                            param: "window".to_string(),
                            reason: format!(
                                "crop window {s}..{end_text} on axis {axis} lies outside the \
                                 input of shape {shape:?}"
                            ),
                        });
                    }
                }
                Ok(())
            }
            // Image rotations: a [H, W] or [H, W, C] buffer. Anything else was
            // returned unchanged or rotated over the wrong axes.
            ViewOp::Rotate90 | ViewOp::Rotate180 | ViewOp::Rotate270 => {
                crate::ops::validation::require_hw_or_hwc(shape)
            }
            ViewOp::ChannelSelect { index } => match shape {
                [_, _, c] if index < c => Ok(()),
                [_, _] if *index == 0 => Ok(()),
                _ => Err(ValidationError::InvalidParameter {
                    param: "index".to_string(),
                    reason: format!("channel {index} of a buffer of shape {shape:?}"),
                }),
            },
        }
    }

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

    fn shape(&self) -> OpShape {
        let known = |v: &[usize]| v.iter().map(|&n| Sym::Known(n)).collect();
        match self {
            ViewOp::Transpose(perm) => OpShape::Transpose(perm.clone()),
            ViewOp::Reshape(new_shape) => OpShape::Fixed(known(new_shape)),
            ViewOp::Flip(_) | ViewOp::Rotate180 => OpShape::Preserve,
            // One entry per input axis, as `ViewBuffer::slice` produces. An
            // `end` of `usize::MAX` is the crop builder's "to the end of this
            // axis" sentinel.
            ViewOp::Crop { start, end } => OpShape::Crop {
                start: known(start),
                len: start
                    .iter()
                    .zip(end)
                    .map(|(&s, &e)| (e != usize::MAX).then(|| Sym::Known(e.saturating_sub(s))))
                    .collect(),
            },
            ViewOp::Rotate90 | ViewOp::Rotate270 => OpShape::SwapHw,
            ViewOp::ChannelSelect { .. } => OpShape::DropChannelAxis,
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

    fn identity_rule(&self) -> IdentityRule {
        match self {
            // A full-frame crop at the origin, or a same-shape reshape (a
            // row-major no-op), moves no data; `OpShape::preserves` decides
            // whether this one is.
            ViewOp::Crop { .. } | ViewOp::Reshape(_) => IdentityRule::WhenShapePreserved,
            // Flip/transpose/rotate/channel-select move pixels or drop an axis
            // even when the shape is preserved (a square transpose, a 180°
            // rotate), so none is ever a no-op.
            ViewOp::Transpose(_)
            | ViewOp::Flip(_)
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270
            | ViewOp::ChannelSelect { .. } => IdentityRule::Never,
        }
    }

    fn is_spatial_window(&self) -> bool {
        match self {
            // A crop is a hoistable spatial window only when it narrows the H/W
            // plane and leaves every further axis at full extent — axes 0/1 may
            // shrink, but each channel/depth axis must keep `start == 0` and the
            // `usize::MAX` full-extent end. That is exactly what the `crop`
            // builder emits (`[top, left, 0] .. [_, _, usize::MAX]`); a crop that
            // slices the channel axis would not commute with a channel-changing
            // pointwise op and is not a window.
            ViewOp::Crop { start, end } => {
                start.iter().skip(2).all(|&s| s == 0)
                    && end.iter().skip(2).all(|&e| e == usize::MAX)
            }
            // Reshape/transpose/flip/rotate/channel-select are not H/W crops.
            ViewOp::Transpose(_)
            | ViewOp::Reshape(_)
            | ViewOp::Flip(_)
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270
            | ViewOp::ChannelSelect { .. } => false,
        }
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        match self {
            // Picks a channel at the same (y, x) — spatially the identity.
            ViewOp::ChannelSelect { .. } => SpatialDependency::Pointwise,
            // Permute/reshape/flip/crop/rotate all remap coordinates (crop
            // offsets the origin; transpose/reshape/rotate move axes).
            ViewOp::Transpose(_)
            | ViewOp::Reshape(_)
            | ViewOp::Flip(_)
            | ViewOp::Crop { .. }
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270 => SpatialDependency::geometric(),
        }
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
