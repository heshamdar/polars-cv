//! Compute operations that transform data.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::affine::{AffineParams, InterpolationType};
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

impl NormalizeMethod {
    /// Canonical Python-facing method names.
    ///
    /// `Preset` carries payload, so this enum cannot use the `named_variants!`
    /// value table; the parser handles `preset` structurally (it needs the
    /// `mean`/`std` parameters). The exhaustive match below still forces this
    /// list to be revisited when a variant is added.
    pub const NAMES: &'static [&'static str] = &["minmax", "zscore", "preset"];
}

const _: fn(&NormalizeMethod) = |m| match m {
    NormalizeMethod::MinMax | NormalizeMethod::ZScore | NormalizeMethod::Preset { .. } => (),
};

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
    /// Adjust contrast: `(pixel - mean) * factor + mean`.
    /// Requires full buffer scan to compute the mean.
    AdjustContrast(f32),
    /// Adjust gamma (power-law): normalize to [0,1], apply `pixel^gamma`, denormalize.
    AdjustGamma(f32),
    /// Invert pixel values: `max_val - pixel` (255 for u8, 1.0 for float).
    Invert,
    /// Deferred rotation via affine transform. The affine matrix is built at
    /// execution time from the actual buffer dimensions so that the image
    /// center is computed correctly.
    RotateAffine {
        angle_deg: f32,
        expand: bool,
        interpolation: InterpolationType,
        border_value: f64,
    },
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
            ComputeOp::AdjustContrast(_) => "AdjustContrast",
            ComputeOp::AdjustGamma(_) => "AdjustGamma",
            ComputeOp::Invert => "Invert",
            ComputeOp::RotateAffine { .. } => "RotateAffine",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        match self {
            ComputeOp::Affine(params) => {
                let input_shape = inputs[0];
                let mut s = input_shape.to_vec();
                if s.len() >= 2 {
                    s[0] = params.output_height as usize;
                    s[1] = params.output_width as usize;
                }
                s
            }
            ComputeOp::RotateAffine {
                angle_deg, expand, ..
            } => {
                let input_shape = inputs[0];
                if !expand || input_shape.len() < 2 {
                    return input_shape.to_vec();
                }
                let ih = input_shape[0] as f64;
                let iw = input_shape[1] as f64;
                let rad = (*angle_deg as f64) * std::f64::consts::PI / 180.0;
                let abs_cos = rad.cos().abs();
                let abs_sin = rad.sin().abs();
                let new_w = (iw * abs_cos + ih * abs_sin).round() as usize;
                let new_h = (ih * abs_cos + iw * abs_sin).round() as usize;
                let mut s = input_shape.to_vec();
                s[0] = new_h;
                s[1] = new_w;
                s
            }
            _ => inputs[0].to_vec(),
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        match self {
            ComputeOp::Cast(target) => *target,
            ComputeOp::Normalize(_) => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::Scale(_) => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::Relu => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::Clamp { .. } => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::AdjustContrast(_) => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::AdjustGamma(_) => self.output_dtype_rule().resolve(inputs[0], None),
            ComputeOp::Invert => inputs[0],
            ComputeOp::Affine(_) => inputs[0],
            ComputeOp::RotateAffine { .. } => inputs[0],
            // The kernel computes in f32 and converts to its pinned output
            // dtype in the same pass (set to the unfused chain's planned
            // dtype at fusion time).
            ComputeOp::Fused(k) => k.out_dtype,
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            ComputeOp::Cast(_) => MemoryEffect::StridePreserving,
            ComputeOp::Scale(_) => MemoryEffect::StridePreserving,
            ComputeOp::Relu => MemoryEffect::StridePreserving,
            ComputeOp::Fused(_) => MemoryEffect::StridePreserving,
            ComputeOp::Clamp { .. } => MemoryEffect::StridePreserving,
            ComputeOp::AdjustGamma(_) => MemoryEffect::StridePreserving,
            ComputeOp::Invert => MemoryEffect::StridePreserving,
            ComputeOp::Affine(_) => MemoryEffect::RequiresContiguous,
            ComputeOp::RotateAffine { .. } => MemoryEffect::RequiresContiguous,
            ComputeOp::Normalize(_) => MemoryEffect::RequiresContiguous,
            ComputeOp::AdjustContrast(_) => MemoryEffect::RequiresContiguous,
        }
    }

    fn intrinsic_cost(&self) -> OpCost {
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

                match method {
                    NormalizeMethod::MinMax | NormalizeMethod::ZScore => {
                        if !is_2d_like(shape) {
                            return Err(ValidationError::ShapeRequirement {
                                requirement: "2D (HW) or single-channel (HW1)",
                                got: shape.to_vec(),
                            });
                        }
                    }
                    NormalizeMethod::Preset { mean, std } => {
                        if shape.len() < 2 || shape.len() > 3 {
                            return Err(ValidationError::ShapeRequirement {
                                requirement: "2D (HW) or 3D (HWC)",
                                got: shape.to_vec(),
                            });
                        }
                        let channels = if shape.len() == 3 { shape[2] } else { 1 };
                        if mean.len() != channels || std.len() != channels {
                            return Err(ValidationError::ShapeRequirement {
                                requirement: "mean/std length must match channel count",
                                got: vec![mean.len(), std.len(), channels],
                            });
                        }
                    }
                }

                if !self.accepted_input_dtypes().accepts(input_dtypes[0]) {
                    return Err(ValidationError::DTypeRequirement {
                        expected: vec![DType::F32, DType::F64],
                        got: input_dtypes[0],
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        match self {
            ComputeOp::Normalize(_)
            | ComputeOp::Scale(_)
            | ComputeOp::Clamp { .. }
            | ComputeOp::Relu
            | ComputeOp::AdjustContrast(_)
            | ComputeOp::AdjustGamma(_)
            | ComputeOp::Invert => DTypeCategory::Numeric,
            ComputeOp::Cast(_) => DTypeCategory::Any,
            ComputeOp::Affine(_) => DTypeCategory::Any,
            ComputeOp::RotateAffine { .. } => DTypeCategory::Any,
            ComputeOp::Fused(_) => DTypeCategory::Any,
        }
    }

    fn working_dtype(&self) -> Option<DType> {
        match self {
            ComputeOp::Normalize(_) => Some(DType::F32),
            ComputeOp::Scale(_) => Some(DType::F32),
            ComputeOp::Clamp { .. } => Some(DType::F32),
            ComputeOp::Relu => Some(DType::F32),
            ComputeOp::AdjustContrast(_) => Some(DType::F32),
            ComputeOp::AdjustGamma(_) => Some(DType::F32),
            ComputeOp::Invert => None,
            _ => None,
        }
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            ComputeOp::Normalize(_) => OutputDTypeRule::Configurable(DType::F32),
            ComputeOp::Scale(_) => OutputDTypeRule::PromoteToFloat,
            ComputeOp::Clamp { .. } => OutputDTypeRule::PromoteToFloat,
            ComputeOp::Relu => OutputDTypeRule::PromoteToFloat,
            ComputeOp::AdjustContrast(_) => OutputDTypeRule::PromoteToFloat,
            ComputeOp::AdjustGamma(_) => OutputDTypeRule::PromoteToFloat,
            ComputeOp::Invert => OutputDTypeRule::PreserveInput,
            ComputeOp::Cast(target) => OutputDTypeRule::Fixed(*target),
            ComputeOp::Affine(_) => OutputDTypeRule::PreserveInput,
            ComputeOp::RotateAffine { .. } => OutputDTypeRule::PreserveInput,
            ComputeOp::Fused(k) => OutputDTypeRule::Fixed(k.out_dtype),
        }
    }
}
