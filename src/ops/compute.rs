//! Compute operations that transform data.

use crate::core::dtype::DType;
use crate::ops::affine::AffineParams;
use crate::ops::cost::OpCost;
use crate::ops::scalar::FusedKernel;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::{is_2d_like, ValidationError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Method for normalizing data.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NormalizeMethod {
    /// Scale to [0.0, 1.0] range using min/max.
    MinMax,
    /// Standardize using (x - mean) / std.
    ZScore,
}

/// Compute operations that process data element-wise or globally.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComputeOp {
    /// Cast to a different data type.
    Cast(DType),
    /// Apply an affine transformation.
    Affine(AffineParams),
    /// Scale by a constant factor.
    Scale(f32),
    /// Apply ReLU activation.
    Relu,
    /// Apply a fused kernel of scalar operations.
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
