//! Spatial filtering (convolution) operations.
//!
//! Provides generic 2D convolution with arbitrary kernels and configurable
//! border handling. Higher-level operations like Sobel, Laplacian, and
//! sharpen are implemented in the Python layer as predefined kernel wrappers.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::shape_rule::OpShape;
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};

use crate::kernels::filter::convolve_in;
use crate::mode::{Exec, Mode};
use polars_cv_macros::{Ops, Resolve};

/// Border handling mode for convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Replicate the nearest edge pixel.
    Replicate,
    /// Treat out-of-bounds pixels as zero.
    Zero,
    /// Reflect pixels around the edge (dcba|abcd|dcba).
    Reflect,
}

crate::naming::named_variants!(BorderMode: "Border-handling mode for 2D convolution (``convolve2d``).\n\n- REPLICATE: Replicate the nearest edge pixel.\n- ZERO: Treat out-of-bounds pixels as zero.\n- REFLECT: Reflect pixels around the edge (dcba|abcd|dcba)." {
    "replicate" => Replicate,
    "zero" => Zero,
    "reflect" => Reflect,
});

/// Apply generic 2D convolution with an arbitrary kernel.
///
/// Example:
///     ```python
///     >>> edge = Pipeline().source("image_bytes").convolve2d(
///     ...     [-1, -1, -1, -1, 8, -1, -1, -1, -1]
///     ... )
///     ```
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
#[op(name = "convolve2d", sample = {"kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    "normalize": false, "border": "replicate"})]
pub struct ConvolveOp<M: Mode = Exec> {
    /// Flattened square kernel, row-major: its length is the square of an odd
    /// side (9 for 3×3, 25 for 5×5, ...), which is the kernel's size. **Each
    /// coefficient may be a literal float or a Polars expression**, so a batch
    /// can convolve with a different kernel per row; the length is structural.
    pub kernel: Vec<M::V<f32>>,
    /// If True, divide output by the sum of absolute kernel values.
    #[param(default = false)]
    pub normalize: M::V<bool>,
    /// Border handling mode (``"replicate"``, ``"zero"``, ``"reflect"``).
    #[param(default = "replicate")]
    pub border: M::V<BorderMode>,
}

impl<M: Mode> ConvolveOp<M> {
    /// The kernel's side, from its length (structural, so known at plan time).
    pub fn side(&self) -> usize {
        self.kernel.len().isqrt()
    }

    /// Refuse a kernel no row can run: its length must be the square of an
    /// odd side.
    pub fn check(&self) -> Result<(), String> {
        let len = self.kernel.len();
        let side = self.side();
        if side * side != len || side.is_multiple_of(2) {
            return Err(format!(
                "convolve2d: kernel length {len} must be the square of an odd \
                 number (9 for 3x3, 25 for 5x5, ...)"
            ));
        }
        Ok(())
    }

    /// Same-size convolution (padded to keep the dimensions).
    pub fn shape(&self) -> OpShape {
        OpShape::Preserve
    }
}

impl<M: Mode> Op for ConvolveOp<M> {
    fn validate(
        &self,
        input_shapes: &[&[crate::ops::Dim]],
        _input_dtypes: &[crate::PlannedDType],
    ) -> Result<(), crate::ops::validation::ValidationError> {
        self.check()
            .map_err(|message| crate::ops::validation::ValidationError::Generic { message })?;
        crate::ops::validation::require_hw_or_hwc(input_shapes[0])
    }

    fn name(&self) -> &'static str {
        "Convolve2D"
    }

    fn shape(&self) -> OpShape {
        ConvolveOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Reads any layout: the input in the accumulator's dtype (itself, or
        // its conversion, which walks a view once) is read a row at a time,
        // in place where the rows are packed, else each packed once into a
        // ring of `ksize` rows (`Rows`). A planned pack would copy the view.
        MemoryEffect::StridePreserving
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false // A convolution is not an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // A side×side kernel: output at (y, x) depends on input within
        // side / 2 pixels of (y, x).
        SpatialDependency::neighborhood(self.side() / 2)
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        None // Produces contiguous output
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Numeric
    }

    fn working_dtype(&self) -> Option<DType> {
        None // Accumulates in the input's `DType::accumulator`
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        OutputDTypeRule::PromoteToFloat
    }
}

/// Apply 2D convolution to a buffer.
///
/// Accumulates in the input's [`DType::accumulator`] (f32, or f64 for f64 and
/// 32/64-bit integer input) and stores the declared output dtype
/// (`PromoteToFloat`: f64 for f64, f32 otherwise). For multi-channel images,
/// each channel is convolved independently.
///
/// Structured for auto-vectorization (interior/border split, same pattern as
/// the separable Gaussian blur in `kernels/blur.rs`):
/// - interior pixels accumulate each kernel tap as a contiguous shifted-slice
///   multiply-add over the row segment — no bounds handling, no per-tap
///   border dispatch, vectorizes for any channel count;
/// - the border ring keeps the original per-pixel clamped/reflected gather.
///
/// Taps are visited in the same `ky`-outer/`kx`-inner order as the original
/// per-pixel loop and `norm_factor` is applied as a final multiply, so every
/// element accumulates in the identical floating-point order: the f32 result
/// is bit-exact with the pre-split implementation (see tests/it/convolve_ref.rs).
pub fn apply_convolve2d(buf: &ViewBuffer, op: &ConvolveOp) -> ViewBuffer {
    let out_dtype = op.output_dtype_rule().resolve(buf.dtype());
    let out = match buf.dtype().accumulator() {
        DType::F64 => convolve_in::<f64>(buf, op),
        _ => convolve_in::<f32>(buf, op),
    };
    if out.dtype() == out_dtype {
        out
    } else {
        out.cast(out_dtype)
    }
}
