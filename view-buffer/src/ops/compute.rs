//! Compute operations that transform data.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::affine::{AffineParams, InterpolationType};
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A normalization, with the per-channel statistics a preset carries.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Normalization {
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

/// Which normalization: the user-facing method name, without its payload.
///
/// `Normalization::Preset` carries statistics, so it cannot hold a
/// `named_variants!` table; this fieldless twin does, and
/// [`Normalization::method`] ties the two together exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizeMethod {
    MinMax,
    ZScore,
    Preset,
}

crate::naming::named_variants!(NormalizeMethod: "Normalization methods (``PRESET``: channel-wise with preset mean/std values)." {
    "minmax" => MinMax,
    "zscore" => ZScore,
    "preset" => Preset,
});

impl Normalization {
    /// The method this normalization is.
    pub fn method(&self) -> NormalizeMethod {
        match self {
            Normalization::MinMax => NormalizeMethod::MinMax,
            Normalization::ZScore => NormalizeMethod::ZScore,
            Normalization::Preset { .. } => NormalizeMethod::Preset,
        }
    }
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
    /// A single pure-elementwise scalar op (`abs`, `sqrt`, `min(c)`, …).
    ///
    /// The user-facing bridge for the core math primitives: each such
    /// `Pipeline` method resolves to one `ComputeOp::Scalar(ScalarOp)`. It
    /// carries the same `PromoteToFloat` contract as `Scale`/`Relu`/`Clamp` and
    /// lowers (via `extract_ops`) to its inner `ScalarOp`, so the fused and
    /// unfused paths share one arithmetic authority. `ScalarOp` is the single
    /// source of both the arithmetic and the op identity, so no per-op
    /// `ComputeOp` variant is needed.
    Scalar(ScalarOp),
    /// Normalize data - requires full buffer scan. Only supports 2D-like shapes (HW or HW1).
    ///
    /// Computation always happens in f32; the second field is the output dtype,
    /// folded from the structural `out_dtype` parameter (defaulting to `F32`).
    /// The op reports it via a `Fixed(out_dtype)` rule, so the planner's
    /// `output_dtype_rule().resolve(input)` already yields it; execution casts
    /// the f32 result to it so the produced dtype matches — see the
    /// dtype-contract tests.
    Normalize(Normalization, DType),
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
            ComputeOp::Normalize(..) => "Normalize",
            ComputeOp::Clamp { .. } => "Clamp",
            ComputeOp::AdjustContrast(_) => "AdjustContrast",
            ComputeOp::AdjustGamma(_) => "AdjustGamma",
            ComputeOp::Invert => "Invert",
            ComputeOp::RotateAffine { .. } => "RotateAffine",
            ComputeOp::Scalar(s) => s.name(),
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

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            ComputeOp::Cast(_) => MemoryEffect::StridePreserving,
            ComputeOp::Scale(_) => MemoryEffect::StridePreserving,
            ComputeOp::Relu => MemoryEffect::StridePreserving,
            ComputeOp::Fused(_) => MemoryEffect::StridePreserving,
            ComputeOp::Scalar(_) => MemoryEffect::StridePreserving,
            ComputeOp::Clamp { .. } => MemoryEffect::StridePreserving,
            ComputeOp::AdjustGamma(_) => MemoryEffect::StridePreserving,
            ComputeOp::Invert => MemoryEffect::StridePreserving,
            ComputeOp::Affine(_) => MemoryEffect::RequiresContiguous,
            ComputeOp::RotateAffine { .. } => MemoryEffect::RequiresContiguous,
            ComputeOp::Normalize(..) => MemoryEffect::RequiresContiguous,
            ComputeOp::AdjustContrast(_) => MemoryEffect::RequiresContiguous,
        }
    }

    fn identity_rule(&self) -> IdentityRule {
        match self {
            // A same-dtype cast copies its input; any other target converts.
            ComputeOp::Cast(_) => IdentityRule::WhenDtypePreserved,
            // Every other compute op transforms pixel values (arithmetic ops
            // also promote to float, so they are not even dtype-preserving).
            ComputeOp::Affine(_)
            | ComputeOp::Scale(_)
            | ComputeOp::Relu
            | ComputeOp::Fused(_)
            | ComputeOp::Scalar(_)
            | ComputeOp::Normalize(..)
            | ComputeOp::Clamp { .. }
            | ComputeOp::AdjustGamma(_)
            | ComputeOp::Invert
            | ComputeOp::AdjustContrast(_)
            | ComputeOp::RotateAffine { .. } => IdentityRule::Never,
        }
    }

    fn is_spatial_window(&self) -> bool {
        false // Compute ops transform values, never an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        match self {
            // Per-element: output at (y, x) depends only on input at (y, x).
            ComputeOp::Cast(_)
            | ComputeOp::Scale(_)
            | ComputeOp::Relu
            | ComputeOp::Fused(_)
            | ComputeOp::Scalar(_)
            | ComputeOp::Clamp { .. }
            | ComputeOp::AdjustGamma(_)
            | ComputeOp::Invert => SpatialDependency::Pointwise,
            // Read a global statistic (min/max/mean/std) over all pixels.
            ComputeOp::Normalize(..) | ComputeOp::AdjustContrast(_) => SpatialDependency::Global,
            // Resample onto a transformed coordinate grid.
            ComputeOp::Affine(_) | ComputeOp::RotateAffine { .. } => SpatialDependency::geometric(),
        }
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        // `memory_effect` describes the INPUT side (whether the kernel can
        // consume a strided view without a materialize step). The output
        // side is different: every compute kernel either writes a fresh
        // contiguous buffer or mutates an already-contiguous one in place,
        // so the output layout is contiguous regardless of input strides —
        // and the element size may change (integer gamma -> f32), which
        // input byte strides can never describe.
        //
        // `Cast` is the one op whose same-dtype case preserves the input
        // layout, but that decision needs the source and target dtypes,
        // which are not visible here — so `Cast`'s strides are computed in
        // the `ViewExpr::cast` builder directly and never routed through
        // this method. Returning contiguous (None) for every variant keeps
        // this a truthful, dtype-agnostic default.
        None
    }

    fn validate(
        &self,
        input_shapes: &[&[usize]],
        input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        match self {
            ComputeOp::Normalize(method, _) => {
                let shape = input_shapes[0];

                match method {
                    // Global statistics over every element: any shape.
                    Normalization::MinMax | Normalization::ZScore => {}
                    Normalization::Preset { mean, std } => {
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

                // A shape-only caller (plan-time validation) passes no dtype.
                if let Some(&dtype) = input_dtypes.first() {
                    if !self.accepted_input_dtypes().accepts(dtype) {
                        return Err(ValidationError::DTypeRequirement {
                            expected: vec![DType::F32, DType::F64],
                            got: dtype,
                        });
                    }
                }
                Ok(())
            }
            ComputeOp::Affine(_) | ComputeOp::RotateAffine { .. } => {
                crate::ops::validation::require_hw_or_hwc(input_shapes[0])
            }
            _ => Ok(()),
        }
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        match self {
            ComputeOp::Normalize(..)
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
            ComputeOp::Scalar(_) => DTypeCategory::Numeric,
        }
    }

    fn working_dtype(&self) -> Option<DType> {
        match self {
            ComputeOp::Normalize(..) => Some(DType::F32),
            ComputeOp::Scale(_) => Some(DType::F32),
            ComputeOp::Clamp { .. } => Some(DType::F32),
            ComputeOp::Relu => Some(DType::F32),
            ComputeOp::AdjustContrast(_) => Some(DType::F32),
            ComputeOp::AdjustGamma(_) => Some(DType::F32),
            // No fixed f32 working dtype: these compute in the input dtype
            // (or defer to a nested kernel). Listed rather than `_ => None` so a
            // new ComputeOp must declare its working dtype instead of inheriting
            // `None` silently.
            ComputeOp::Invert
            | ComputeOp::Cast(_)
            | ComputeOp::Affine(_)
            | ComputeOp::RotateAffine { .. }
            | ComputeOp::Fused(_)
            // The kernel reads any dtype and converts internally (like Fused),
            // so there is no external working-dtype pre-cast.
            | ComputeOp::Scalar(_) => None,
        }
    }

    fn output_rank_rule(&self) -> OutputRankRule {
        // Every compute kind is element-wise or a geometric H/W warp
        // (affine/rotate) — the rank is always preserved.
        OutputRankRule::PreserveRank
    }

    fn output_channel_rule(&self) -> OutputChannelRule {
        // Compute kinds operate per element and never add or drop channels.
        OutputChannelRule::PreserveChannels
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            // The op is constructed with its output dtype already resolved
            // (`out_dtype`, defaulting to F32), so the rule is that concrete
            // dtype — no separate override pass is needed once the op exists.
            ComputeOp::Normalize(_, out_dtype) => OutputDTypeRule::Fixed(*out_dtype),
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
            ComputeOp::Scalar(_) => OutputDTypeRule::PromoteToFloat,
        }
    }
}
