//! Compute operations that transform data.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::affine::{AffineParams, InterpolationType};
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;

use crate::mode::{size, Exec, Mode};
use crate::ops::view::ViewOp;
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
    #[op(name = "scale", visibility = Internal, sample = {"factor": 2.0})]
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
    #[op(name = "clamp", visibility = Internal, sample = {"min": 0.0, "max": 1.0})]
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
    /// Invert values, keeping the dtype: `MAX + MIN - x` for an integer
    /// (`255 - x` for u8, `-1 - x` for a signed dtype), `1 - x` for a float.
    #[op(name = "invert", sample = {})]
    Invert,
    /// Negate every value (`-x`).
    #[op(name = "neg", sample = {})]
    Neg,
    /// Absolute value (`|x|`).
    #[op(name = "abs", sample = {})]
    Abs,
    /// Square root (`sqrt(x)`; NaN for negative input).
    #[op(name = "sqrt", sample = {})]
    Sqrt,
    /// Square (`x * x`).
    #[op(name = "square", sample = {})]
    Square,
    /// Reciprocal (`1 / x`; ±inf at zero).
    #[op(name = "reciprocal", sample = {})]
    Reciprocal,
    /// Sign: `-1`/`0`/`+1` (`0` for ±0, NaN for NaN).
    #[op(name = "sign", sample = {})]
    Sign,
    /// Round toward negative infinity.
    #[op(name = "floor", sample = {})]
    Floor,
    /// Round toward positive infinity.
    #[op(name = "ceil", sample = {})]
    Ceil,
    /// Round to nearest, ties to even (matches Polars/numpy).
    #[op(name = "round", sample = {})]
    Round,
    /// Round toward zero (drop the fractional part).
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
    /// Apply a 2x3 affine transformation matrix.
    ///
    /// The matrix ``[a, b, tx, c, d, ty]`` is a **forward** mapping from
    /// source to destination (same convention as OpenCV ``warpAffine``):
    ///
    /// ```text
    /// x_dst = a * x_src + b * y_src + tx
    /// y_dst = c * x_src + d * y_src + ty
    /// ```
    ///
    /// The kernel inverts this matrix internally for interpolation.
    ///
    /// Example:
    ///     ```python
    ///     >>> # Translate image by (50, 30)
    ///     >>> pipe = Pipeline().source("image_bytes").warp_affine(
    ///     ...     matrix=[1.0, 0.0, 50.0, 0.0, 1.0, 30.0],
    ///     ...     output_size=(224, 224),
    ///     ... )
    ///     >>>
    ///     >>> # Per-sample random affine: each row uses its own matrix columns
    ///     >>> pipe = Pipeline().source("image_bytes").warp_affine(
    ///     ...     matrix=[pl.col("a"), pl.col("b"), pl.col("tx"),
    ///     ...             pl.col("c"), pl.col("d"), pl.col("ty")],
    ///     ...     output_size=(224, 224),
    ///     ... )
    ///     ```
    #[op(name = "warp_affine", sample = {"matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                         "output_size": [4, 4],
                                         "interpolation": "bilinear",
                                         "border_value": 0.0})]
    WarpAffine {
        /// Six-element sequence representing the 2x3 affine matrix
        /// ``[a, b, tx, c, d, ty]`` (forward mapping). **Each element may be a
        /// literal float or a Polars expression**, so a batch can apply a
        /// different (e.g. random) affine per row in one call — the matrix is
        /// resolved per row at execution.
        matrix: [M::V<f64>; 6],
        /// ``(height, width)`` of the output image. Each element accepts a Polars
        /// expression for per-row dynamic values.
        output_size: [M::V<u32>; 2],
        /// Interpolation method -- ``"bilinear"`` (default) or ``"nearest"``.
        #[param(default = "bilinear")]
        interpolation: M::V<InterpolationType>,
        /// Pixel value for out-of-bounds regions (default 0).
        #[param(default = 0.0)]
        border_value: M::V<f64>,
    },
    /// Combined rotation and scaling around a center point.
    ///
    /// Warps with OpenCV's ``getRotationMatrix2D(center, -angle, scale)``
    /// matrix, built per row from the values the row resolves: the matrix
    /// is computed in one place (the engine), so a literal and the same
    /// value as an expression give identical output.
    ///
    /// Example:
    ///     ```python
    ///     >>> pipe = Pipeline().source("image_bytes").rotate_and_scale(
    ///     ...     angle=45.0, scale=1.2, center=(112, 112), output_size=(224, 224)
    ///     ... )
    ///     >>> # Per-row angle from a column
    ///     >>> pipe = Pipeline().source("image_bytes").rotate_and_scale(
    ///     ...     angle=pl.col("theta"), center=(112, 112), output_size=(224, 224)
    ///     ... )
    ///     ```
    #[op(name = "rotate_and_scale", sample = {"angle": 30.0, "center": [2.0, 2.0],
                                              "output_size": [4, 4], "scale": 1.0})]
    RotateAndScale {
        /// Rotation angle in degrees (positive = clockwise). Accepts a Polars
        /// expression for a per-row angle.
        angle: M::V<f64>,
        /// ``(cx, cy)`` center of rotation. Required: an image source's
        /// height/width are not known until execution, so there is no
        /// plan-time centre to default to. Each element accepts an expression.
        center: [M::V<f64>; 2],
        /// ``(height, width)`` of the output. Required, because the output
        /// shape is part of the plan-time schema. Each element accepts an
        /// expression.
        output_size: [M::V<u32>; 2],
        /// Scale factor (default 1.0). Accepts an expression.
        #[param(default = 1.0)]
        scale: M::V<f64>,
    },
    /// Rotate image by specified angle.
    ///
    /// For angles of 90, 180, or 270 degrees, this uses zero-copy view operations
    /// (``interpolation`` and ``border_value`` do not apply: nothing is resampled
    /// and no out-of-bounds region is exposed). For arbitrary angles, the rotation
    /// is performed via an affine transformation using the specified
    /// interpolation and border value. For combined rotation + scale or explicit
    /// output sizing, use :meth:`rotate_and_scale` or :meth:`warp_affine`.
    ///
    /// Example:
    ///     ```python
    ///     >>> pipe = Pipeline().source("image_bytes").rotate(90)
    ///     >>> pipe = Pipeline().source("image_bytes").rotate(45, expand=True)
    ///     >>> pipe = Pipeline().source("image_bytes").rotate(pl.col("angle"))
    ///     >>> pipe = Pipeline().source("image_bytes").rotate(30, interpolation="nearest")
    ///     ```
    #[op(name = "rotate", sample = {"angle": 30.0, "expand": true, "interpolation": "nearest",
                                    "border_value": 0.0})]
    Rotate {
        /// Rotation angle in degrees (positive = clockwise). Can be a literal float
        /// or Polars expression.
        angle: M::V<f32>,
        /// If True, expand output dimensions to fit rotated image. If False
        /// (default), keep original dimensions (corners may be cropped).
        #[param(default = false)]
        expand: M::L<bool>,
        /// Interpolation method for arbitrary angles -- ``"bilinear"`` (default) or
        /// ``"nearest"``. Not applicable to 90/180/270 degree rotations.
        #[param(default = "bilinear")]
        interpolation: M::V<InterpolationType>,
        /// Fill value for out-of-bounds pixels (default 0). Not applicable to
        /// 90/180/270 degree rotations.
        #[param(default = 0.0)]
        border_value: M::V<f64>,
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
    /// definition, read on the `Wire` op at plan time and on the `Exec` op
    /// at execution.
    pub fn shape(&self) -> OpShape {
        match self {
            ComputeOp::WarpAffine {
                output_size: [h, w],
                ..
            }
            | ComputeOp::RotateAndScale {
                output_size: [h, w],
                ..
            } => OpShape::SetHw {
                h: size::<M>(h),
                w: size::<M>(w),
            },
            ComputeOp::Rotate { angle, expand, .. } => match (M::sym(angle), M::lit(expand)) {
                // A per-row angle is a lattice rotation (swapping H/W) on some
                // rows and a resampling on others.
                (Sym::PerRow, false) => OpShape::MaybeSwapHw,
                (Sym::PerRow, true) => OpShape::RotateExpand(Sym::PerRow),
                (Sym::Known(angle), expand) => match Rotation::of(angle) {
                    Rotation::Lattice(view) => view.shape(),
                    Rotation::Identity => OpShape::Preserve,
                    Rotation::Resample(angle) if expand => OpShape::RotateExpand(Sym::Known(angle)),
                    Rotation::Resample(_) => OpShape::Preserve,
                },
            },
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
        // Warping is inverse mapping — for each output pixel, ask where it
        // came from — so a matrix that collapses the plane onto a line or a
        // point has no answer. Refused where every coefficient is known: at
        // plan time for a literal matrix, per row for a per-row one.
        if let ComputeOp::WarpAffine { matrix, .. } = self {
            let known: Option<Vec<f64>> = matrix.iter().map(|c| M::sym(c).known()).collect();
            if let Some(known) = known {
                let m: [f64; 6] = [known[0], known[1], known[2], known[3], known[4], known[5]];
                if !AffineParams::matrix_is_invertible(&m) {
                    let [a, b, _, c, d, _] = m;
                    return Err(format!(
                        "warp_affine: matrix {m:?} is singular or not finite \
                         (determinant {}), so it has no inverse and the warp is undefined. A non-finite coefficient, a row of \
                         zeros, a zero scale factor on an axis, or two proportional \
                         rows will do this.",
                        a * d - b * c
                    ));
                }
            }
            return Ok(());
        }
        // The rotation matrix's determinant is scale**2: a zero scale is
        // singular, and a non-finite angle, centre or scale makes it NaN. Each
        // known value is checked as soon as it is known (a literal at plan
        // time), and the whole matrix — the same rule `warp_affine` reads —
        // once every value is.
        if let ComputeOp::RotateAndScale {
            angle,
            center: [cx, cy],
            scale,
            ..
        } = self
        {
            let named = [
                ("angle", M::sym(angle).known()),
                ("center x", M::sym(cx).known()),
                ("center y", M::sym(cy).known()),
                ("scale", M::sym(scale).known()),
            ];
            for (name, value) in named {
                if let Some(v) = value.filter(|v| !v.is_finite()) {
                    return Err(format!("rotate_and_scale: {name} {v} is not finite"));
                }
            }
            if let Some(scale) = named[3].1 {
                if (scale * scale).abs() < AffineParams::SINGULAR_EPSILON {
                    return Err(format!(
                        "rotate_and_scale: scale {scale} collapses the image to a \
                         point, so the warp has no inverse and is undefined."
                    ));
                }
            }
            if let [Some(angle), Some(cx), Some(cy), Some(scale)] = named.map(|(_, v)| v) {
                let m = AffineParams::rotation_matrix_2d(angle, cx, cy, scale);
                if !AffineParams::matrix_is_invertible(&m) {
                    return Err(format!(
                        "rotate_and_scale: angle {angle}, centre ({cx}, {cy}) and scale \
                         {scale} give the matrix {m:?}, which has no inverse."
                    ));
                }
            }
            return Ok(());
        }
        // A rotation by a non-finite angle has no matrix (and no expanded
        // canvas: `AffineParams::expanded_size` of NaN is 0).
        if let ComputeOp::Rotate { angle, .. } = self {
            if let Some(angle) = M::sym(angle).known().filter(|a| !a.is_finite()) {
                return Err(format!("rotate: angle {angle} is not finite"));
            }
            return Ok(());
        }
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

/// What a rotation by `angle` degrees is: the one classification the
/// planner's shape and execution's lowering both read.
enum Rotation {
    /// A multiple of 90° other than 0: a zero-copy view.
    Lattice(ViewOp),
    /// 0° (mod 360): nothing moves.
    Identity,
    /// Any other angle (normalized to `[0, 360)`): resampled.
    Resample(f32),
}

impl Rotation {
    fn of(angle: f32) -> Rotation {
        const EPSILON: f32 = 0.001;
        let angle = angle.rem_euclid(360.0);
        let near = |target: f32| (angle - target).abs() < EPSILON;
        if near(90.0) {
            Rotation::Lattice(ViewOp::Rotate90)
        } else if near(180.0) {
            Rotation::Lattice(ViewOp::Rotate180)
        } else if near(270.0) {
            Rotation::Lattice(ViewOp::Rotate270)
        } else if near(0.0) || near(360.0) {
            Rotation::Identity
        } else {
            Rotation::Resample(angle)
        }
    }
}

impl ComputeOp {
    /// The engine step this op executes as. A rotation by a lattice angle is
    /// a zero-copy view and any other a resampling warp, chosen from the
    /// (resolved) angle; a warp's matrix becomes its `AffineParams`. Every
    /// other op executes as itself.
    pub fn lowered(self) -> crate::ops::dto::ViewDto {
        use crate::ops::dto::ViewDto;
        match self {
            ComputeOp::Rotate {
                angle,
                expand,
                interpolation,
                border_value,
            } => match Rotation::of(angle) {
                Rotation::Lattice(view) => ViewDto::View(view),
                // The lattice rotations and the 0° no-op are exact
                // permutations of the input pixels; `interpolation` and
                // `border_value` apply only to the resampling branch. A
                // `RotateAffine` of exactly 0° is how the identity is spelled:
                // it returns its (packed) input, sharing its data, and never
                // warps (`execution::warp::rotate`).
                Rotation::Identity => ViewDto::Compute(ComputeOp::RotateAffine {
                    angle_deg: 0.0,
                    expand: false,
                    interpolation: InterpolationType::Bilinear,
                    border_value: 0.0,
                }),
                // The matrix is built at execution from the buffer's size.
                Rotation::Resample(angle) => ViewDto::Compute(ComputeOp::RotateAffine {
                    angle_deg: angle,
                    expand,
                    interpolation,
                    border_value,
                }),
            },
            ComputeOp::WarpAffine {
                matrix,
                output_size: [output_height, output_width],
                interpolation,
                border_value,
            } => ViewDto::Compute(ComputeOp::Affine(AffineParams {
                matrix,
                output_height,
                output_width,
                interpolation,
                border_value,
            })),
            // The matrix comes from the one rotation authority, on the row's
            // own values.
            ComputeOp::RotateAndScale {
                angle,
                center: [cx, cy],
                output_size: [output_height, output_width],
                scale,
            } => ViewDto::Compute(ComputeOp::Affine(AffineParams {
                matrix: AffineParams::rotation_matrix_2d(angle, cx, cy, scale),
                output_height,
                output_width,
                interpolation: InterpolationType::Bilinear,
                border_value: 0.0,
            })),
            other => ViewDto::Compute(other),
        }
    }

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

impl<M: Mode> Op for ComputeOp<M> {
    fn name(&self) -> &'static str {
        // A scalar op is named by the one arithmetic authority it lowers to
        // (`scalar()`; `wire_scalars_are_named_as_their_scalar_op` pins it).
        match self {
            ComputeOp::Scalar(s) => s.name(),
            ComputeOp::Neg => "Neg",
            ComputeOp::Abs => "Abs",
            ComputeOp::Sqrt => "Sqrt",
            ComputeOp::Square => "Square",
            ComputeOp::Reciprocal => "Recip",
            ComputeOp::Sign => "Sign",
            ComputeOp::Floor => "Floor",
            ComputeOp::Ceil => "Ceil",
            ComputeOp::Round => "Round",
            ComputeOp::Trunc => "Trunc",
            ComputeOp::ClampMin { .. } => "Max",
            ComputeOp::ClampMax { .. } => "Min",
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
            ComputeOp::WarpAffine { .. } => "WarpAffine",
            ComputeOp::RotateAndScale { .. } => "RotateAndScale",
            ComputeOp::Rotate { .. } => "Rotate",
            ComputeOp::AddConstant { .. } => "Add",
            ComputeOp::SubtractConstant { .. } => "Sub",
        }
    }

    fn shape(&self) -> OpShape {
        ComputeOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            ComputeOp::Affine(_)
            | ComputeOp::RotateAffine { .. }
            | ComputeOp::WarpAffine { .. }
            | ComputeOp::RotateAndScale { .. }
            | ComputeOp::Rotate { .. } => MemoryEffect::RequiresContiguous,
            // Every per-value op, `normalize` and `adjust_contrast` included
            // (their statistics too), reads a view where it lies
            // (`core::map`), so a planned pack would only copy it.
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
            ComputeOp::Affine(_)
            | ComputeOp::RotateAffine { .. }
            | ComputeOp::WarpAffine { .. }
            | ComputeOp::RotateAndScale { .. }
            | ComputeOp::Rotate { .. } => SpatialDependency::geometric(),
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
        input_shapes: &[&[crate::ops::Dim]],
        input_dtypes: &[crate::PlannedDType],
    ) -> Result<(), ValidationError> {
        match self {
            ComputeOp::Normalize {
                method, mean, std, ..
            } => {
                self.check()
                    .map_err(|message| ValidationError::Generic { message })?;
                let shape = input_shapes[0];
                if let (NormalizeMethod::Preset, Some(mean), Some(std)) =
                    (M::lit(method), mean, std)
                {
                    if shape.len() < 2 || shape.len() > 3 {
                        return Err(ValidationError::ShapeRequirement {
                            requirement: "2D (HW) or 3D (HWC)",
                            got: shape.to_vec(),
                        });
                    }
                    // An unknown channel count is checked per row.
                    let channels = if shape.len() == 3 {
                        shape[2].known()
                    } else {
                        Some(1)
                    };
                    if let Some(channels) = channels {
                        if mean.len() != channels || std.len() != channels {
                            return Err(ValidationError::ShapeRequirement {
                                requirement: "mean/std length must match channel count",
                                got: shape.to_vec(),
                            });
                        }
                    }
                }

                // An unknown dtype is checked per row.
                if let Some(crate::PlannedDType::Known(dtype)) = input_dtypes.first().copied() {
                    if !self.accepted_input_dtypes().accepts(dtype) {
                        return Err(ValidationError::DTypeRequirement {
                            expected: vec![DType::F32, DType::F64],
                            got: dtype,
                        });
                    }
                }
                Ok(())
            }
            ComputeOp::WarpAffine { .. } | ComputeOp::RotateAndScale { .. } => {
                self.check()
                    .map_err(|message| ValidationError::Generic { message })?;
                crate::ops::validation::require_hw_or_hwc(input_shapes[0])
            }
            ComputeOp::Affine(_) | ComputeOp::RotateAffine { .. } | ComputeOp::Rotate { .. } => {
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
            | ComputeOp::WarpAffine { .. }
            | ComputeOp::RotateAndScale { .. }
            | ComputeOp::Rotate { .. }
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
                OutputDTypeRule::Fixed(out_dtype.as_ref().map_or(DType::F32, M::lit))
            }
            ComputeOp::Cast { dtype } => OutputDTypeRule::Fixed(M::lit(dtype)),
            ComputeOp::Fused(k) => OutputDTypeRule::Fixed(k.out_dtype),
            ComputeOp::Invert
            | ComputeOp::Affine(_)
            | ComputeOp::RotateAffine { .. }
            | ComputeOp::WarpAffine { .. }
            | ComputeOp::RotateAndScale { .. }
            | ComputeOp::Rotate { .. } => OutputDTypeRule::PreserveInput,
            _ => OutputDTypeRule::PromoteToFloat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode::{Literals, Resolve, Wire};

    /// A geometric parameter that is not finite has no warp: a NaN or
    /// infinite angle, centre or scale makes the matrix NaN, and it used to
    /// pass (`NaN < eps` is false) and warp every pixel to the border. Each
    /// warp's matrix is checked by the one rule, `AffineParams::is_invertible`
    /// (finite and non-singular), per row when a value is per-row.
    #[test]
    fn a_warp_with_a_non_finite_parameter_is_refused() {
        let rs = |angle: f64, cx: f64, scale: f64| ComputeOp::<Exec>::RotateAndScale {
            angle,
            center: [cx, 2.0],
            output_size: [4, 4],
            scale,
        };
        let warp = |m0: f64| ComputeOp::<Exec>::WarpAffine {
            matrix: [m0, 0.0, 0.0, 0.0, 1.0, 0.0],
            output_size: [4, 4],
            interpolation: InterpolationType::Bilinear,
            border_value: 0.0,
        };
        let rotate = |angle: f32| ComputeOp::<Exec>::Rotate {
            angle,
            expand: true,
            interpolation: InterpolationType::Bilinear,
            border_value: 0.0,
        };
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(rs(bad, 2.0, 1.0).check().is_err(), "angle {bad}");
            assert!(rs(30.0, bad, 1.0).check().is_err(), "centre {bad}");
            assert!(rs(30.0, 2.0, bad).check().is_err(), "scale {bad}");
            assert!(warp(bad).check().is_err(), "matrix {bad}");
            assert!(rotate(bad as f32).check().is_err(), "rotate {bad}");
        }
        assert!(rs(30.0, 2.0, 1.0).check().is_ok());
        assert!(warp(1.0).check().is_ok());
        assert!(rotate(30.0).check().is_ok());
    }

    /// `rotate_and_scale` executes as a warp that turns the image clockwise
    /// (y down) about its centre and scales it. Checked against the geometry
    /// rather than the function that builds the matrix: the centre stays put,
    /// and a point one unit right of it lands `scale` along the turned axis.
    #[test]
    fn rotate_and_scale_lowers_to_the_rotation_matrix() {
        let cases = [
            // (angle, centre, scale, where (cx + 1, cy) must land)
            (90.0, [1.0, 3.0], 2.0, [1.0, 5.0]),
            (180.0, [0.0137, 0.0137], 0.5, [0.0137 - 0.5, 0.0137]),
            (30.0, [4.0, -2.0], 1.0, [4.0 + 0.75f64.sqrt(), -2.0 + 0.5]),
        ];
        for (angle, [cx, cy], scale, [ex, ey]) in cases {
            let op: ComputeOp<Exec> = ComputeOp::RotateAndScale {
                angle,
                center: [cx, cy],
                output_size: [5, 6],
                scale,
            };
            let crate::ops::dto::ViewDto::Compute(ComputeOp::Affine(params)) = op.lowered() else {
                panic!("rotate_and_scale must lower to an affine warp");
            };
            let m = params.matrix;
            let at = |x: f64, y: f64| (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
            let close = |(x, y): (f64, f64), (u, v): (f64, f64)| {
                (x - u).abs() < 1e-12 && (y - v).abs() < 1e-12
            };
            assert!(close(at(cx, cy), (cx, cy)), "{angle}: the centre moved");
            assert!(
                close(at(cx + 1.0, cy), (ex, ey)),
                "{angle}: (cx + 1, cy) -> {:?}, want ({ex}, {ey})",
                at(cx + 1.0, cy)
            );
            assert_eq!((params.output_height, params.output_width), (5, 6));
        }
    }

    #[test]
    fn a_zero_scale_is_refused() {
        let op: ComputeOp<Exec> = ComputeOp::RotateAndScale {
            angle: 10.0,
            center: [1.0, 1.0],
            output_size: [4, 4],
            scale: 0.0,
        };
        assert!(op.check().unwrap_err().contains("scale 0"));
    }

    /// The generic `name()` spells each wire scalar op's `ScalarOp` name
    /// (a per-row value cannot build the `ScalarOp`); the executed op's
    /// `scalar()` is the authority it must match.
    #[test]
    fn wire_scalars_are_named_as_their_scalar_op() {
        let mut scalars = 0;
        for wire in ComputeOp::<Wire>::samples() {
            let exec: ComputeOp = wire.resolve(&Literals).unwrap();
            assert_eq!(wire.name(), exec.name());
            if let Some(scalar) = exec.scalar() {
                assert_eq!(exec.name(), scalar.name(), "{exec:?}");
                scalars += 1;
            }
        }
        assert!(scalars >= 14, "only {scalars} scalar ops compared");
    }

    /// An op executes as the step it lowers to — `rotate` by a lattice angle
    /// as a zero-copy view, `warp_affine` as `Affine` — whose shape is that
    /// step's own; the plan reads the op's. The two must agree.
    #[test]
    fn a_lowered_op_keeps_its_shape() {
        let mut ops: Vec<ComputeOp> = ComputeOp::<Wire>::samples()
            .iter()
            .map(|op| op.resolve(&Literals).unwrap())
            .collect();
        // Every rotation class: the lattice angles, the identity, a resample.
        for angle in [0.0, 90.0, 180.0, 270.0, -90.0, 30.0] {
            for expand in [false, true] {
                ops.push(ComputeOp::Rotate {
                    angle,
                    expand,
                    interpolation: InterpolationType::Bilinear,
                    border_value: 0.0,
                });
            }
        }
        let input = [7usize, 5, 3];
        for op in ops {
            let lowered = op.clone().lowered();
            assert_eq!(
                op.shape().concrete(&[&input]),
                lowered.as_op().shape().concrete(&[&input]),
                "{op:?} lowers to {lowered:?}"
            );
        }
    }
}
