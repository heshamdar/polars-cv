//! Compute operations that transform data.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::execution::tiling::TilePolicy;
use crate::ops::affine::AffineParams;
use crate::ops::cost::OpCost;
use crate::ops::scalar::FusedKernel;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::{is_2d_like, ValidationError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Method for normalizing data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NormalizeMethod {
    /// Scale to [0.0, 1.0] range using min/max.
    MinMax,
    /// Standardize using (x - mean) / std (computed per-image).
    ZScore,
    /// Channel-wise normalization with preset mean/std values.
    ///
    /// Used for ImageNet-style normalization where mean and std are
    /// precomputed across the entire dataset.
    ///
    /// For RGB images: `(pixel - mean[c]) / std[c]` for each channel c.
    ///
    /// Example ImageNet values:
    /// - mean: [0.485, 0.456, 0.406]
    /// - std: [0.229, 0.224, 0.225]
    Preset {
        /// Per-channel mean values (typically 3 for RGB).
        mean: Vec<f32>,
        /// Per-channel standard deviation values (typically 3 for RGB).
        std: Vec<f32>,
    },
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
            // Use the new output dtype rules for operations that need promotion
            ComputeOp::Normalize(_) => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::Scale(_) => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::Relu => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::Clamp { .. } => self.output_dtype_rule().resolve(inputs[0], None),
            // Other ops preserve dtype
            ComputeOp::Affine(_) => inputs[0],
            ComputeOp::Fused(_) => inputs[0],
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
            ComputeOp::Normalize(method) => {
                let shape = input_shapes[0];

                // Validate shape requirements based on method
                match method {
                    NormalizeMethod::MinMax | NormalizeMethod::ZScore => {
                        // Per-image normalization: only supports 2D-like shapes (HW or HW1)
                        if !is_2d_like(shape) {
                            return Err(ValidationError::ShapeRequirement {
                                requirement: "2D (HW) or single-channel (HW1)",
                                got: shape.to_vec(),
                            });
                        }
                    }
                    NormalizeMethod::Preset { mean, std } => {
                        // Channel-wise normalization: requires HWC with matching channel count
                        if shape.len() < 2 || shape.len() > 3 {
                            return Err(ValidationError::ShapeRequirement {
                                requirement: "2D (HW) or 3D (HWC)",
                                got: shape.to_vec(),
                            });
                        }
                        // Get number of channels (1 for HW, C for HWC)
                        let channels = if shape.len() == 3 { shape[2] } else { 1 };
                        if mean.len() != channels || std.len() != channels {
                            return Err(ValidationError::ShapeRequirement {
                                requirement: "mean/std length must match channel count",
                                got: vec![mean.len(), std.len(), channels],
                            });
                        }
                    }
                }

                // Validate that input dtype is accepted
                if !self.accepted_input_dtypes().accepts(input_dtypes[0]) {
                    return Err(ValidationError::DTypeRequirement {
                        expected: vec![DType::F32, DType::F64], // Indicate numeric types accepted
                        got: input_dtypes[0],
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // --- Dtype Contract Methods ---

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        match self {
            // These operations accept all numeric types and handle casting internally
            ComputeOp::Normalize(_) => DTypeCategory::Numeric,
            ComputeOp::Scale(_) => DTypeCategory::Numeric,
            ComputeOp::Clamp { .. } => DTypeCategory::Numeric,
            ComputeOp::Relu => DTypeCategory::Numeric,
            // Cast accepts anything
            ComputeOp::Cast(_) => DTypeCategory::Any,
            // Others default to any
            ComputeOp::Affine(_) => DTypeCategory::Any,
            ComputeOp::Fused(_) => DTypeCategory::Any,
        }
    }

    fn working_dtype(&self) -> Option<DType> {
        match self {
            // Normalize needs f32 for numerical stability (accumulator)
            ComputeOp::Normalize(_) => Some(DType::F32),
            // Scale uses f32 for the multiplication
            ComputeOp::Scale(_) => Some(DType::F32),
            // Clamp and Relu work in f32 for safety
            ComputeOp::Clamp { .. } => Some(DType::F32),
            ComputeOp::Relu => Some(DType::F32),
            // Others work with whatever dtype they receive
            _ => None,
        }
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            // Normalize: default to f32, but can be configured
            ComputeOp::Normalize(_) => OutputDTypeRule::Configurable(DType::F32),
            // Scale: promote integers to float, preserve floats
            ComputeOp::Scale(_) => OutputDTypeRule::PromoteToFloat,
            // Clamp: preserve input dtype (user expects same type back)
            ComputeOp::Clamp { .. } => OutputDTypeRule::PromoteToFloat,
            // Relu: promote to float for proper negative handling
            ComputeOp::Relu => OutputDTypeRule::PromoteToFloat,
            // Cast: always outputs the target dtype
            ComputeOp::Cast(target) => OutputDTypeRule::Fixed(*target),
            // Others preserve input
            ComputeOp::Affine(_) => OutputDTypeRule::PreserveInput,
            ComputeOp::Fused(_) => OutputDTypeRule::PreserveInput,
        }
    }

    #[inline]
    fn tile_policy(&self) -> TilePolicy {
        match self {
            // Point-wise operations - no pixel dependencies
            ComputeOp::Scale(_) => TilePolicy::PointWise,
            ComputeOp::Relu => TilePolicy::PointWise,
            ComputeOp::Clamp { .. } => TilePolicy::PointWise,
            ComputeOp::Cast(_) => TilePolicy::PointWise,
            ComputeOp::Fused(_) => TilePolicy::PointWise,

            // Normalize with preset values is point-wise (fixed per-channel params)
            ComputeOp::Normalize(NormalizeMethod::Preset { .. }) => TilePolicy::PointWise,

            // Normalize with minmax/zscore needs global statistics
            ComputeOp::Normalize(NormalizeMethod::MinMax) => TilePolicy::Global,
            ComputeOp::Normalize(NormalizeMethod::ZScore) => TilePolicy::Global,

            // Affine transformation needs global context
            ComputeOp::Affine(_) => TilePolicy::Global,
        }
    }
}
