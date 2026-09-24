//! Affine ops: warps by a 2x3 matrix.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::{AffineParams, ComputeOp, InterpolationType, ViewDto};

use super::{OpDef, Param};
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
