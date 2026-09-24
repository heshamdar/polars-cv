//! Compute ops: element-wise arithmetic, casts and normalization.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::{ComputeOp, DType, Normalization, NormalizeMethod, ScalarOp, ViewDto};

use super::{Literal, OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

fn compute(op: ComputeOp) -> PolarsResult<GraphStep> {
    Ok(GraphStep::Buffer(ViewDto::Compute(op)))
}

/// Cast to a different data type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Cast {
    /// Target data type (e.g., "f32", "u8").
    #[param(positional)]
    pub dtype: Literal<DType>,
}

impl OpDef for Cast {
    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Cast { dtype } = self;
        compute(ComputeOp::Cast(dtype.get()))
    }
}

/// Multiply all values by a factor.
///
/// The public `Pipeline.scale` is sugar over this op that adds
/// `out_dtype`/`preserve_dtype` (a trailing cast).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "internal")]
pub struct Scale {
    /// Scale factor.
    #[param(positional)]
    pub factor: Param<f32>,
}

impl OpDef for Scale {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Scale { factor } = self;
        compute(ComputeOp::Scale(factor.resolve(row, ctx)?))
    }
}

/// Clamp values to a range.
///
/// The public `Pipeline.clamp` is sugar over this op that adds
/// `out_dtype`/`preserve_dtype` (a trailing cast).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "internal")]
pub struct Clamp {
    /// Minimum value (literal or expression).
    #[param(positional)]
    pub min: Param<f32>,
    /// Maximum value (literal or expression).
    #[param(positional)]
    pub max: Param<f32>,
}

impl OpDef for Clamp {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Clamp { min, max } = self;
        compute(ComputeOp::Clamp {
            min: min.resolve(row, ctx)?,
            max: max.resolve(row, ctx)?,
        })
    }
}

/// Normalize values to a standard range.
///
/// Example:
///     >>> Pipeline().source().normalize(method="minmax")
///     >>> Pipeline().source().normalize(
///     ...     method="preset",
///     ...     mean=[0.485, 0.456, 0.406],
///     ...     std=[0.229, 0.224, 0.225],
///     ... )
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Normalize {
    /// Normalization method: "minmax" scales values to [0, 1] using
    /// per-element min/max; "zscore" standardizes to mean=0, std=1 using
    /// per-element statistics; "preset" applies ImageNet-style channel-wise
    /// normalization, `(x - mean[c]) / std[c]`, with the given `mean` and `std`.
    #[param(positional, default = "minmax")]
    pub method: Literal<NormalizeMethod>,
    /// Per-channel mean values; required for, and only valid with,
    /// method="preset" (e.g. ImageNet `[0.485, 0.456, 0.406]`). Each element may
    /// be a literal float or a Polars expression; the list length is the channel
    /// count.
    #[param(positional)]
    pub mean: Option<Vec<Param<f32>>>,
    /// Per-channel standard deviation values; required for, and only valid
    /// with, method="preset" (e.g. ImageNet `[0.229, 0.224, 0.225]`). Each
    /// element accepts an expression, as with `mean`.
    #[param(positional)]
    pub std: Option<Vec<Param<f32>>>,
    /// Output dtype (default f32). Normalization computes in f32 and the result
    /// is cast to this dtype at execution. For half precision use the sink
    /// dtype instead (`.sink("numpy", dtype="f16")`).
    #[param(positional)]
    pub out_dtype: Option<Literal<DType>>,
}

impl OpDef for Normalize {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Normalize {
            method,
            mean,
            std,
            out_dtype,
        } = self;
        let values = |list: &[Param<f32>]| -> PolarsResult<Vec<f32>> {
            list.iter().map(|p| p.resolve(row, ctx)).collect()
        };
        // `mean`/`std` belong to one method only; for the others they would be
        // accepted and ignored, so they are refused instead. (The flat field
        // list follows the frozen Python signature; the API phase, P8, can make
        // the preset statistics part of the method itself.)
        let without_stats = |n: Normalization| -> PolarsResult<Normalization> {
            if mean.is_some() || std.is_some() {
                polars_bail!(ComputeError:
                    "mean/std parameters are only valid for method='preset'");
            }
            Ok(n)
        };
        let normalization = match method.get() {
            NormalizeMethod::MinMax => without_stats(Normalization::MinMax)?,
            NormalizeMethod::ZScore => without_stats(Normalization::ZScore)?,
            NormalizeMethod::Preset => {
                let (Some(mean), Some(std)) = (mean, std) else {
                    polars_bail!(ComputeError:
                        "method='preset' requires both 'mean' and 'std' parameters");
                };
                if mean.len() != std.len() {
                    polars_bail!(ComputeError:
                        "mean length ({}) must match std length ({})", mean.len(), std.len());
                }
                Normalization::Preset {
                    mean: values(mean)?,
                    std: values(std)?,
                }
            }
        };
        let out_dtype = out_dtype.map_or(DType::F32, |d| d.get());
        compute(ComputeOp::Normalize(normalization, out_dtype))
    }
}

/// Declare a parameterless element-wise op: `name => struct, doc, ComputeOp`.
macro_rules! unary_ops {
    ($($ty:ident, $doc:literal, $op:expr;)+) => {$(
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        pub struct $ty {}

        impl OpDef for $ty {
            fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty {} = self;
                compute($op)
            }
        }
    )+};
}

unary_ops! {
    Relu, "Apply ReLU activation (max(0, x)): negative values become zero.",
        ComputeOp::Relu;
    Invert, "Invert pixel values: `255 - pixel` for u8, `1.0 - pixel` for float [0,1].",
        ComputeOp::Invert;
    Neg, "Negate every value (`-x`). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Neg);
    Abs, "Absolute value (`|x|`). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Abs);
    Sqrt, "Square root (`sqrt(x)`; NaN for negative input). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Sqrt);
    Square, "Square (`x * x`). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Square);
    Reciprocal, "Reciprocal (`1 / x`; ±inf at zero). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Recip);
    Sign, "Sign: `-1`/`0`/`+1` (`0` for ±0, NaN for NaN). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Sign);
    Floor, "Round toward negative infinity. Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Floor);
    Ceil, "Round toward positive infinity. Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Ceil);
    Round, "Round to nearest, ties to even (matches Polars/numpy). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Round);
    Trunc, "Round toward zero (drop the fractional part). Domain: buffer → buffer.",
        ComputeOp::Scalar(ScalarOp::Trunc);
}

/// Declare a one-value element-wise op taking a per-row `value`.
macro_rules! value_ops {
    ($($ty:ident, $doc:literal, $value_doc:literal, $op:path;)+) => {$(
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        pub struct $ty {
            #[doc = $value_doc]
            #[param(positional)]
            pub value: Param<f32>,
        }

        impl OpDef for $ty {
            fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty { value } = self;
                compute(ComputeOp::Scalar($op(value.resolve(row, ctx)?)))
            }
        }
    )+};
}

value_ops! {
    ClampMin, "Floor values at `value` (`max(x, value)`); one-sided clamp.",
        "Lower bound (literal or per-row expression).", ScalarOp::Max;
    ClampMax, "Cap values at `value` (`min(x, value)`); one-sided clamp.",
        "Upper bound (literal or per-row expression).", ScalarOp::Min;
    AddConstant, "Add a constant to every value (`x + value`).",
        "Constant addend (literal or per-row expression).", ScalarOp::Add;
    SubtractConstant, "Subtract a constant from every value (`x - value`).",
        "Constant subtrahend (literal or per-row expression).", ScalarOp::Sub;
}

/// Adjust image contrast: `(pixel - mean) * factor + mean`.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").adjust_contrast(factor=1.5)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct AdjustContrast {
    /// Contrast factor. 1.0 = no change, >1 = more contrast, <1 = less.
    pub factor: Param<f32>,
}

impl OpDef for AdjustContrast {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let AdjustContrast { factor } = self;
        compute(ComputeOp::AdjustContrast(factor.resolve(row, ctx)?))
    }
}

/// Apply gamma (power-law) correction: normalize to [0,1], raise to `gamma`,
/// denormalize.
///
/// Example:
///     >>> pipe = Pipeline().source("image_bytes").adjust_gamma(gamma=0.5)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct AdjustGamma {
    /// Gamma value. <1 = brighter, >1 = darker, 1.0 = no change.
    pub gamma: Param<f32>,
}

impl OpDef for AdjustGamma {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let AdjustGamma { gamma } = self;
        compute(ComputeOp::AdjustGamma(gamma.resolve(row, ctx)?))
    }
}
