//! Affine ops: warps by a 2x3 matrix.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::{AffineParams, ComputeOp, InterpolationType, ViewDto, ViewOp};

use super::{Literal, OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

/// Apply a 2x3 affine transformation matrix.
///
/// The matrix ``[a, b, tx, c, d, ty]`` is a **forward** mapping from
/// source to destination (same convention as OpenCV ``warpAffine``)::
///
///     x_dst = a * x_src + b * y_src + tx
///     y_dst = c * x_src + d * y_src + ty
///
/// The kernel inverts this matrix internally for interpolation.
///
/// Domain: buffer → buffer
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct WarpAffine {
    /// Six-element sequence representing the 2x3 affine matrix
    /// ``[a, b, tx, c, d, ty]`` (forward mapping). **Each element may be a
    /// literal float or a Polars expression**, so a batch can apply a
    /// different (e.g. random) affine per row in one call — the matrix is
    /// resolved per row at execution.
    #[param(positional)]
    pub matrix: [Param<f64>; 6],
    /// ``(height, width)`` of the output image. Each element accepts a Polars
    /// expression for per-row dynamic values.
    #[param(positional)]
    pub output_size: [Param<u32>; 2],
    /// Interpolation method -- ``"bilinear"`` (default) or ``"nearest"``.
    #[param(default = "bilinear")]
    pub interpolation: Param<InterpolationType>,
    /// Pixel value for out-of-bounds regions (default 0).
    #[param(default = 0.0)]
    pub border_value: Param<f64>,
}

impl OpDef for WarpAffine {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let WarpAffine {
            matrix,
            output_size: [output_height, output_width],
            interpolation,
            border_value,
        } = self;
        let mut coefficients = [0.0; 6];
        for (out, p) in coefficients.iter_mut().zip(matrix) {
            *out = p.resolve(row, ctx)?;
        }
        let affine = AffineParams {
            matrix: coefficients,
            output_height: output_height.resolve(row, ctx)?,
            output_width: output_width.resolve(row, ctx)?,
            interpolation: interpolation.resolve(row, ctx)?,
            border_value: border_value.resolve(row, ctx)?,
        };
        // Warping is inverse mapping — for each output pixel, ask where it
        // came from — so a matrix that collapses the plane onto a line or a
        // point has no answer. Reject it here, where the user supplied it, and
        // name the determinant so the offending coefficients are findable.
        //
        // Not under a plan-time probe: every expression parameter is bound to
        // the *same* placeholder there, so a per-row matrix arrives as six
        // equal coefficients and is singular by construction. Its real values
        // only exist per row, where this same code runs for a dynamic op.
        if !ctx.is_probe() && !affine.is_invertible() {
            return Err(polars_err!(ComputeError:
                "warp_affine: matrix {:?} is singular (determinant {}), so it \
                 has no inverse and the warp is undefined. A row of zeros, a \
                 zero scale factor on an axis, or two proportional rows will \
                 do this.",
                affine.matrix, affine.determinant()));
        }
        Ok(GraphStep::Buffer(ViewDto::Compute(ComputeOp::Affine(
            affine,
        ))))
    }
}

/// Rotate image by specified angle.
///
/// For angles of 90, 180, or 270 degrees, this uses zero-copy view operations
/// (``interpolation`` and ``border_value`` do not apply: nothing is resampled
/// and no out-of-bounds region is exposed). For arbitrary angles, the rotation
/// is performed via an affine transformation using the specified
/// interpolation and border value. For combined rotation + scale or explicit
/// output sizing, use :meth:`rotate_and_scale` or :meth:`warp_affine`.
///
/// Domain: buffer -> buffer
///
/// Example:
///     ```python
///     >>> pipe = Pipeline().source("image_bytes").rotate(90)
///     >>> pipe = Pipeline().source("image_bytes").rotate(45, expand=True)
///     >>> pipe = Pipeline().source("image_bytes").rotate(pl.col("angle"))
///     >>> pipe = Pipeline().source("image_bytes").rotate(30, interpolation="nearest")
///     ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Rotate {
    /// Rotation angle in degrees (positive = clockwise). Can be a literal float
    /// or Polars expression.
    #[param(positional)]
    pub angle: Param<f32>,
    /// If True, expand output dimensions to fit rotated image. If False
    /// (default), keep original dimensions (corners may be cropped).
    #[param(default = false)]
    pub expand: Literal<bool>,
    /// Interpolation method for arbitrary angles -- ``"bilinear"`` (default) or
    /// ``"nearest"``. Not applicable to 90/180/270 degree rotations.
    #[param(default = "bilinear")]
    pub interpolation: Param<InterpolationType>,
    /// Fill value for out-of-bounds pixels (default 0). Not applicable to
    /// 90/180/270 degree rotations.
    #[param(default = 0.0)]
    pub border_value: Param<f64>,
}

impl OpDef for Rotate {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Rotate {
            angle,
            expand,
            interpolation,
            border_value,
        } = self;
        const EPSILON: f32 = 0.001;
        let angle = angle.resolve(row, ctx)?.rem_euclid(360.0);
        let near = |target: f32| (angle - target).abs() < EPSILON;
        // The lattice rotations and the 0° no-op are exact permutations of the
        // input pixels; `interpolation`/`border_value` apply only to the
        // resampling branch.
        let step = if near(90.0) {
            ViewDto::View(ViewOp::Rotate90)
        } else if near(180.0) {
            ViewDto::View(ViewOp::Rotate180)
        } else if near(270.0) {
            ViewDto::View(ViewOp::Rotate270)
        } else if near(0.0) || near(360.0) {
            ViewDto::Compute(ComputeOp::RotateAffine {
                angle_deg: 0.0,
                expand: false,
                interpolation: InterpolationType::Bilinear,
                border_value: 0.0,
            })
        } else {
            // The matrix is built at execution from the buffer's dimensions.
            ViewDto::Compute(ComputeOp::RotateAffine {
                angle_deg: angle,
                expand: expand.get(),
                interpolation: interpolation.resolve(row, ctx)?,
                border_value: border_value.resolve(row, ctx)?,
            })
        };
        Ok(GraphStep::Buffer(step))
    }
}
