use crate::dtype::DType;
use crate::ops::affine::AffineParams;
use crate::ops::cost::OpCost;
use crate::ops::scalar::FusedKernel;
use crate::ops::validation::{is_2d_like, ValidationError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Legacy memory effect enum - kept for backwards compatibility.
/// Prefer using `Op::intrinsic_cost()` which returns `OpCost`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEffect {
    View,
    StridePreserving,
    RequiresContiguous,
}

impl From<MemoryEffect> for OpCost {
    fn from(effect: MemoryEffect) -> Self {
        match effect {
            MemoryEffect::View => OpCost::ZeroCopy,
            MemoryEffect::StridePreserving => OpCost::Allocating,
            MemoryEffect::RequiresContiguous => OpCost::Allocating,
        }
    }
}

/// Trait for all operations in the pipeline.
///
/// Operations must provide shape/dtype inference, cost information,
/// and optional validation for plan-time error checking.
pub trait Op {
    /// Returns the name of this operation for display/debugging.
    fn name(&self) -> &'static str;

    /// Infers the output shape given input shapes.
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>;

    /// Infers the output dtype given input dtypes.
    fn infer_dtype(&self, inputs: &[DType]) -> DType;

    /// Returns the legacy memory effect. Prefer `intrinsic_cost()`.
    fn memory_effect(&self) -> MemoryEffect;

    /// Returns the intrinsic cost of this operation.
    fn intrinsic_cost(&self) -> OpCost {
        self.memory_effect().into()
    }

    /// Infers output strides given input shape and strides.
    ///
    /// Returns None if strides cannot be inferred or if the operation
    /// requires materialization that makes input strides irrelevant.
    fn infer_strides(&self, input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>>;

    /// Validates the operation at plan time.
    ///
    /// Returns Ok(()) if the operation is valid for the given inputs,
    /// or Err with a description of why validation failed.
    fn validate(
        &self,
        _input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        // Default: no validation requirements
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ViewOp {
    Transpose(Vec<usize>),
    Reshape(Vec<usize>),
    Flip(Vec<usize>),
    Crop { start: Vec<usize>, end: Vec<usize> },
}

impl Op for ViewOp {
    fn name(&self) -> &'static str {
        match self {
            ViewOp::Transpose(_) => "Transpose",
            ViewOp::Reshape(_) => "Reshape",
            ViewOp::Flip(_) => "Flip",
            ViewOp::Crop { .. } => "Crop",
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

    fn infer_strides(&self, input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match self {
            ViewOp::Transpose(perm) => Some(perm.iter().map(|&i| input_strides[i]).collect()),
            ViewOp::Reshape(_new_shape) => {
                let layout_check =
                    crate::layout::LayoutFacts::new(input_shape, input_strides, DType::U8, 0);

                if layout_check.is_contiguous() {
                    None // Defer to runtime/planner checks for now as we lack DType here
                } else {
                    None // Invalid view
                }
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
        }
    }
}

/// Method for normalizing data.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NormalizeMethod {
    /// Scale to [0.0, 1.0] range using min/max.
    MinMax,
    /// Standardize using (x - mean) / std.
    ZScore,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComputeOp {
    Cast(DType),
    Affine(AffineParams),
    Scale(f32),
    Relu,
    Fused(FusedKernel),
    /// Normalize data - requires full buffer scan. Only supports 2D-like shapes (HW or HW1).
    Normalize(NormalizeMethod),
    /// Clamp values to [min, max] range.
    Clamp { min: f32, max: f32 },
}

impl Op for ComputeOp {
    fn name(&self) -> &'static str {
        match self {
            ComputeOp::Cast(_) => "Cast",
            ComputeOp::Affine(_) => "Affine",
            ComputeOp::Scale(_) => "Scale",
            ComputeOp::Relu => "Relu",
            ComputeOp::Fused(_) => "Fused",
            ComputeOp::Normalize(_) => "Normalize",
            ComputeOp::Clamp { .. } => "Clamp",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        inputs[0].to_vec()
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        match self {
            ComputeOp::Cast(target) => *target,
            // All other ops preserve dtype
            ComputeOp::Affine(_) => inputs[0],
            ComputeOp::Scale(_) => inputs[0],
            ComputeOp::Relu => inputs[0],
            ComputeOp::Fused(_) => inputs[0],
            ComputeOp::Normalize(_) => inputs[0],
            ComputeOp::Clamp { .. } => inputs[0],
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            ComputeOp::Cast(_) => MemoryEffect::StridePreserving,
            ComputeOp::Scale(_) => MemoryEffect::StridePreserving,
            ComputeOp::Relu => MemoryEffect::StridePreserving,
            ComputeOp::Fused(_) => MemoryEffect::StridePreserving,
            ComputeOp::Clamp { .. } => MemoryEffect::StridePreserving,
            ComputeOp::Affine(_) => MemoryEffect::RequiresContiguous,
            ComputeOp::Normalize(_) => MemoryEffect::RequiresContiguous,
        }
    }

    fn intrinsic_cost(&self) -> OpCost {
        // All compute ops allocate new buffers
        OpCost::Allocating
    }

    fn infer_strides(&self, _input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match self.memory_effect() {
            MemoryEffect::StridePreserving => Some(input_strides.to_vec()),
            MemoryEffect::RequiresContiguous => None,
            MemoryEffect::View => unreachable!(),
        }
    }

    fn validate(
        &self,
        input_shapes: &[&[usize]],
        input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        match self {
            ComputeOp::Normalize(_) => {
                let shape = input_shapes[0];
                if !is_2d_like(shape) {
                    return Err(ValidationError::ShapeRequirement {
                        requirement: "2D (HW) or single-channel (HW1)",
                        got: shape.to_vec(),
                    });
                }
                if input_dtypes[0] != DType::F32 {
                    return Err(ValidationError::DTypeRequirement {
                        expected: vec![DType::F32],
                        got: input_dtypes[0],
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}