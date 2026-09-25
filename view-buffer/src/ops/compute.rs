//! Compute operations that transform data.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::affine::{AffineParams, InterpolationType};
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;

use crate::mode::{Exec, Mode};
use polars_cv_macros::{Ops, Resolve};

/// A normalization, with the per-channel statistics a preset carries.
#[derive(Debug, Clone, PartialEq)]
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

/// The compute ops: one variant per wire op (see `crate::mode`), plus the
/// engine-internal variants that lowering and fusion produce. Each wire
/// variant's doc comment is its Python docstring.
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum ComputeOp<M: Mode = Exec> {
    /// Cast to a different data type.
    #[op(name = "cast", sample = {"dtype": "f32"})]
    Cast {
        /// Target data type (e.g., "f32", "u8").
        dtype: M::L<DType>,
    },
    /// Multiply all values by a factor.
    ///
    /// The public `Pipeline.scale` is sugar over this op that adds
    /// `out_dtype`/`preserve_dtype` (a trailing cast).
    #[op(name = "scale", visibility = "internal", sample = {"factor": 2.0})]
    Scale {
        /// Scale factor.
        factor: M::V<f32>,
    },
    /// Apply ReLU activation (max(0, x)): negative values become zero.
    #[op(name = "relu", sample = {})]
    Relu,
    /// Normalize values to a standard range.
    ///
    /// Example:
    ///     >>> Pipeline().source().normalize(method="minmax")
    ///     >>> Pipeline().source().normalize(
    ///     ...     method="preset",
    ///     ...     mean=[0.485, 0.456, 0.406],
    ///     ...     std=[0.229, 0.224, 0.225],
    ///     ... )
    #[op(name = "normalize", sample = {"method": "preset", "mean": [0.5], "std": [0.25],
                                       "out_dtype": "f32"})]
    Normalize {
        /// Normalization method: "minmax" scales values to [0, 1] using
        /// per-element min/max; "zscore" standardizes to mean=0, std=1 using
        /// per-element statistics; "preset" applies ImageNet-style channel-wise
        /// normalization, `(x - mean[c]) / std[c]`, with the given `mean` and `std`.
        #[param(default = "minmax")]
        method: M::L<NormalizeMethod>,
        /// Per-channel mean values; required for, and only valid with,
        /// method="preset" (e.g. ImageNet `[0.485, 0.456, 0.406]`). Each element may
        /// be a literal float or a Polars expression; the list length is the channel
        /// count.
        mean: Option<Vec<M::V<f32>>>,
        /// Per-channel standard deviation values; required for, and only valid
        /// with, method="preset" (e.g. ImageNet `[0.229, 0.224, 0.225]`). Each
        /// element accepts an expression, as with `mean`.
        std: Option<Vec<M::V<f32>>>,
        /// Output dtype (default f32). Normalization computes in f32 and the result
        /// is cast to this dtype at execution. For half precision use the sink
        /// dtype instead (`.sink("numpy", dtype="f16")`).
        out_dtype: Option<M::L<DType>>,
    },
    /// Clamp values to a range.
    ///
    /// The public `Pipeline.clamp` is sugar over this op that adds
    /// `out_dtype`/`preserve_dtype` (a trailing cast).
    #[op(name = "clamp", visibility = "internal", sample = {"min": 0.0, "max": 1.0})]
    Clamp {
        /// Minimum value (literal or expression).
        min: M::V<f32>,
        /// Maximum value (literal or expression).
        max: M::V<f32>,
    },
    /// Adjust image contrast: `(pixel - mean) * factor + mean`.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").adjust_contrast(factor=1.5)
    #[op(name = "adjust_contrast", sample = {"factor": 1.5})]
    AdjustContrast {
        /// Contrast factor. 1.0 = no change, >1 = more contrast, <1 = less.
        factor: M::V<f32>,
    },
    /// Apply gamma (power-law) correction: normalize to [0,1], raise to `gamma`,
    /// denormalize.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").adjust_gamma(gamma=0.5)
    #[op(name = "adjust_gamma", sample = {"gamma": 0.5})]
    AdjustGamma {
        /// Gamma value. <1 = brighter, >1 = darker, 1.0 = no change.
        gamma: M::V<f32>,
    },
    /// Invert pixel values: `255 - pixel` for u8, `1.0 - pixel` for float [0,1].
    #[op(name = "invert", sample = {})]
    Invert,
    /// Negate every value (`-x`). Domain: buffer → buffer.
    #[op(name = "neg", sample = {})]
    Neg,
    /// Absolute value (`|x|`). Domain: buffer → buffer.
    #[op(name = "abs", sample = {})]
    Abs,
    /// Square root (`sqrt(x)`; NaN for negative input). Domain: buffer → buffer.
    #[op(name = "sqrt", sample = {})]
    Sqrt,
    /// Square (`x * x`). Domain: buffer → buffer.
    #[op(name = "square", sample = {})]
    Square,
    /// Reciprocal (`1 / x`; ±inf at zero). Domain: buffer → buffer.
    #[op(name = "reciprocal", sample = {})]
    Reciprocal,
    /// Sign: `-1`/`0`/`+1` (`0` for ±0, NaN for NaN). Domain: buffer → buffer.
    #[op(name = "sign", sample = {})]
    Sign,
    /// Round toward negative infinity. Domain: buffer → buffer.
    #[op(name = "floor", sample = {})]
    Floor,
    /// Round toward positive infinity. Domain: buffer → buffer.
    #[op(name = "ceil", sample = {})]
    Ceil,
    /// Round to nearest, ties to even (matches Polars/numpy). Domain: buffer → buffer.
    #[op(name = "round", sample = {})]
    Round,
    /// Round toward zero (drop the fractional part). Domain: buffer → buffer.
    #[op(name = "trunc", sample = {})]
    Trunc,
    /// Floor values at `value` (`max(x, value)`); one-sided clamp.
    #[op(name = "clamp_min", sample = {"value": 0.0})]
    ClampMin {
        /// Lower bound (literal or per-row expression).
        value: M::V<f32>,
    },
    /// Cap values at `value` (`min(x, value)`); one-sided clamp.
    #[op(name = "clamp_max", sample = {"value": 1.0})]
    ClampMax {
        /// Upper bound (literal or per-row expression).
        value: M::V<f32>,
    },
    /// Add a constant to every value (`x + value`).
    #[op(name = "add_constant", sample = {"value": 1.0})]
    AddConstant {
        /// Constant addend (literal or per-row expression).
        value: M::V<f32>,
    },
    /// Subtract a constant from every value (`x - value`).
    #[op(name = "subtract_constant", sample = {"value": 1.0})]
    SubtractConstant {
        /// Constant subtrahend (literal or per-row expression).
        value: M::V<f32>,
    },
    // --- Engine-internal: produced by lowering and fusion, never on the wire.
    /// Apply an affine transformation.
    Affine(AffineParams),
    /// Apply a fused kernel of scalar operations.
    Fused(FusedKernel),
    /// A single pure-elementwise scalar op, as fusion's lowering and the
    /// engine's own builders express one (`ScalarOp` is the one arithmetic
    /// authority; the wire's scalar ops lower to it through [`scalar`]).
    ///
    /// [`scalar`]: ComputeOp::scalar
    Scalar(ScalarOp),
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

impl<M: Mode> ComputeOp<M> {
    /// How this op's output shape follows from its input — the one
    /// definition: every wire compute op is element-wise or a global rescale.
    pub fn shape(&self) -> OpShape {
        match self {
            ComputeOp::Affine(params) => OpShape::SetHw {
                h: Sym::Known(params.output_height as usize),
                w: Sym::Known(params.output_width as usize),
            },
            ComputeOp::RotateAffine {
                angle_deg,
                expand: true,
                ..
            } => OpShape::RotateExpand(Sym::Known(*angle_deg)),
            _ => OpShape::Preserve,
        }
    }

    /// Refuse a parameter combination no row can execute, from the
    /// parameters alone: `mean`/`std` belong to `method="preset"` only, and
    /// it needs both, of one length. Checked when the op is planned.
    pub fn check(&self) -> Result<(), String> {
        let ComputeOp::Normalize {
            method, mean, std, ..
        } = self
        else {
            return Ok(());
        };
        match M::lit(method) {
            NormalizeMethod::MinMax | NormalizeMethod::ZScore => {
                if mean.is_some() || std.is_some() {
                    return Err("mean/std parameters are only valid for method='preset'".into());
                }
            }
            NormalizeMethod::Preset => {
                let (Some(mean), Some(std)) = (mean, std) else {
                    return Err("method='preset' requires both 'mean' and 'std' parameters".into());
                };
                if mean.len() != std.len() {
                    return Err(format!(
                        "mean length ({}) must match std length ({})",
                        mean.len(),
                        std.len()
                    ));
                }
            }
        }
        Ok(())
    }
}

impl ComputeOp {
    /// The pure-elementwise scalar op this is, when it is one: the one
    /// arithmetic authority fusion and the unfused path both run.
    pub fn scalar(&self) -> Option<ScalarOp> {
        Some(match *self {
            ComputeOp::Neg => ScalarOp::Neg,
            ComputeOp::Abs => ScalarOp::Abs,
            ComputeOp::Sqrt => ScalarOp::Sqrt,
            ComputeOp::Square => ScalarOp::Square,
            ComputeOp::Reciprocal => ScalarOp::Recip,
            ComputeOp::Sign => ScalarOp::Sign,
            ComputeOp::Floor => ScalarOp::Floor,
            ComputeOp::Ceil => ScalarOp::Ceil,
            ComputeOp::Round => ScalarOp::Round,
            ComputeOp::Trunc => ScalarOp::Trunc,
            ComputeOp::ClampMin { value } => ScalarOp::Max(value),
            ComputeOp::ClampMax { value } => ScalarOp::Min(value),
            ComputeOp::AddConstant { value } => ScalarOp::Add(value),
            ComputeOp::SubtractConstant { value } => ScalarOp::Sub(value),
            ComputeOp::Scalar(ref s) => s.clone(),
            _ => return None,
        })
    }

    /// A `Normalize` op performing `normalization`, emitting `out_dtype`.
    pub fn from_normalization(normalization: Normalization, out_dtype: DType) -> Self {
        let method = normalization.method();
        let (mean, std) = match normalization {
            Normalization::Preset { mean, std } => (Some(mean), Some(std)),
            Normalization::MinMax | Normalization::ZScore => (None, None),
        };
        ComputeOp::Normalize {
            method,
            mean,
            std,
            out_dtype: Some(out_dtype),
        }
    }

    /// The normalization a `Normalize` op performs (its parameters having
    /// passed [`check`](Self::check)).
    pub fn normalization(
        method: NormalizeMethod,
        mean: &Option<Vec<f32>>,
        std: &Option<Vec<f32>>,
    ) -> Normalization {
        match (method, mean, std) {
            (NormalizeMethod::MinMax, ..) => Normalization::MinMax,
            (NormalizeMethod::ZScore, ..) => Normalization::ZScore,
            (NormalizeMethod::Preset, mean, std) => Normalization::Preset {
                mean: mean.clone().unwrap_or_default(),
                std: std.clone().unwrap_or_default(),
            },
        }
    }
}

impl Op for ComputeOp {
    fn name(&self) -> &'static str {
        // A scalar op is named by the one arithmetic authority it lowers to.
        if let Some(s) = self.scalar() {
            return s.name();
        }
        match self {
            ComputeOp::Cast { .. } => "Cast",
            ComputeOp::Affine(_) => "Affine",
            ComputeOp::Scale { .. } => "Scale",
            ComputeOp::Relu => "Relu",
            ComputeOp::Fused(_) => "Fused",
            ComputeOp::Normalize { .. } => "Normalize",
            ComputeOp::Clamp { .. } => "Clamp",
            ComputeOp::AdjustContrast { .. } => "AdjustContrast",
            ComputeOp::AdjustGamma { .. } => "AdjustGamma",
            ComputeOp::Invert => "Invert",
            ComputeOp::RotateAffine { .. } => "RotateAffine",
            _ => unreachable!("every other variant is a scalar op"),
        }
    }

    fn shape(&self) -> OpShape {
        ComputeOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            ComputeOp::Affine(_) | ComputeOp::RotateAffine { .. } => {
                MemoryEffect::RequiresContiguous
            }
            ComputeOp::Normalize { .. } | ComputeOp::AdjustContrast { .. } => {
                MemoryEffect::RequiresContiguous
            }
            _ => MemoryEffect::StridePreserving,
        }
    }

    fn identity_rule(&self) -> IdentityRule {
        match self {
            // A same-dtype cast copies its input; any other target converts.
            ComputeOp::Cast { .. } => IdentityRule::WhenDtypePreserved,
            // Every other compute op transforms pixel values (arithmetic ops
            // also promote to float, so they are not even dtype-preserving).
            _ => IdentityRule::Never,
        }
    }

    fn is_spatial_window(&self) -> bool {
        false // Compute ops transform values, never an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        match self {
            // Read a global statistic (min/max/mean/std) over all pixels.
            ComputeOp::Normalize { .. } | ComputeOp::AdjustContrast { .. } => {
                SpatialDependency::Global
            }
            // Resample onto a transformed coordinate grid.
            ComputeOp::Affine(_) | ComputeOp::RotateAffine { .. } => SpatialDependency::geometric(),
            // Per-element: output at (y, x) depends only on input at (y, x).
            _ => SpatialDependency::Pointwise,
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
            ComputeOp::Normalize {
                method, mean, std, ..
            } => {
                self.check()
                    .map_err(|message| ValidationError::Generic { message })?;
                let shape = input_shapes[0];
                if let (NormalizeMethod::Preset, Some(mean), Some(std)) = (method, mean, std) {
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
            ComputeOp::Cast { .. }
            | ComputeOp::Affine(_)
            | ComputeOp::RotateAffine { .. }
            | ComputeOp::Fused(_) => DTypeCategory::Any,
            _ => DTypeCategory::Numeric,
        }
    }

    fn working_dtype(&self) -> Option<DType> {
        match self {
            ComputeOp::Normalize { .. }
            | ComputeOp::Scale { .. }
            | ComputeOp::Clamp { .. }
            | ComputeOp::Relu
            | ComputeOp::AdjustContrast { .. }
            | ComputeOp::AdjustGamma { .. } => Some(DType::F32),
            // No fixed f32 working dtype: these compute in the input dtype
            // (or defer to a nested kernel, or — the scalar ops — read any
            // dtype and convert internally, like Fused).
            _ => None,
        }
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            // Computation happens in f32; the result is cast to `out_dtype`.
            ComputeOp::Normalize { out_dtype, .. } => {
                OutputDTypeRule::Fixed(out_dtype.unwrap_or(DType::F32))
            }
            ComputeOp::Cast { dtype } => OutputDTypeRule::Fixed(*dtype),
            ComputeOp::Fused(k) => OutputDTypeRule::Fixed(k.out_dtype),
            ComputeOp::Invert | ComputeOp::Affine(_) | ComputeOp::RotateAffine { .. } => {
                OutputDTypeRule::PreserveInput
            }
            _ => OutputDTypeRule::PromoteToFloat,
        }
    }
}
