//! Convolution.

#[allow(unused_imports)]
use crate::ops::ParamExt as _;
use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::ops::filter::{BorderMode, ConvolveOp};
use view_buffer::ViewDto;

use super::{OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::OpShape;

/// Apply generic 2D convolution with an arbitrary kernel.
///
/// Domain: buffer → buffer
///
/// Example:
///     ```python
///     >>> edge = Pipeline().source("image_bytes").convolve2d(
///     ...     kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
///     ...     ksize=3,
///     ... )
///     ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Convolve2d {
    /// Flattened kernel values (row-major, ``ksize × ksize``). **Each
    /// coefficient may be a literal float or a Polars expression**, so a batch
    /// can convolve with a different kernel per row. The kernel *length* is
    /// structural and must be a literal odd square.
    pub kernel: Vec<Param<f32>>,
    /// Kernel dimension (must be odd; kernel is ``ksize × ksize``). Accepts a
    /// Polars expression for per-row dynamic values.
    pub ksize: Param<u32>,
    /// If True, divide output by the sum of absolute kernel values.
    #[param(default = false)]
    pub normalize: Param<bool>,
    /// Border handling mode (``"replicate"``, ``"zero"``, ``"reflect"``).
    #[param(default = "replicate")]
    pub border: Param<BorderMode>,
}

impl OpDef for Convolve2d {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Convolve2d {
            kernel,
            ksize,
            normalize,
            border,
        } = self;
        let len = kernel.len();
        let side = match *ksize {
            Param::Lit(k) => {
                if k % 2 == 0 {
                    polars_bail!(ComputeError: "convolve2d ksize must be odd, got {}", k);
                }
                k as usize
            }
            // A per-row `ksize` is only known per row, but the kernel length
            // is structural and checkable now: it fixes the side. At plan time
            // there is no row to read, so the side comes from the kernel.
            Param::Slot(_) => {
                let side = len.isqrt();
                if side * side != len || side % 2 == 0 {
                    polars_bail!(ComputeError:
                        "convolve2d kernel length {} must be the square of an odd \
                         number (9 for 3x3, 25 for 5x5, ...)", len);
                }
                if ctx.is_planning() {
                    side
                } else {
                    ksize.resolve(row, ctx)? as usize
                }
            }
        };
        if len != side * side {
            polars_bail!(ComputeError:
                "kernel length {} doesn't match ksize²={}", len, side * side);
        }
        Ok(GraphStep::Buffer(ViewDto::Filter(ConvolveOp {
            kernel: kernel
                .iter()
                .map(|c| c.resolve(row, ctx))
                .collect::<PolarsResult<_>>()?,
            ksize: side,
            normalize: normalize.resolve(row, ctx)?,
            border: border.resolve(row, ctx)?,
        })))
    }
}
