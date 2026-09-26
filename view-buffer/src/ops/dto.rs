use crate::mode::{Exec, Mode};
use crate::ops::color::ColorConvertOp;
use crate::ops::compute::ComputeOp;
use crate::ops::filter::ConvolveOp;
use crate::ops::image::ImageOp;
use crate::ops::traits::{IdentityRule, Op};
use crate::ops::view::ViewOp;
use crate::ops::Domain;
use polars_cv_macros::Resolve;

/// A pure Data Transfer Object (DTO) for single-buffer operation plans.
///
/// Every variant is backed by an [`Op`] implementation and is executable via
/// [`ViewExpr::apply_op`](crate::expr::ViewExpr::apply_op) — the enum contains
/// exactly what the engine can run, nothing more (guarded by the
/// `apply_op_executes_every_view_dto_variant` coverage test). Graph-level
/// concerns — multi-input operations, node references, expression columns,
/// domain transitions — live in the polars-cv plugin's `GraphStep`, not here.
///
/// Generic over the [`Mode`] like its ops: the `Wire` form is what a plan
/// holds and reads its rules from, the `Exec` form what runs.
#[derive(Debug, Clone, PartialEq, Resolve)]
pub enum ViewDto<M: Mode = Exec> {
    /// Zero-copy layout operation (transpose, reshape, flip, crop, …).
    View(ViewOp<M>),
    /// Element-wise compute operation (cast, scale, clamp, affine, …).
    Compute(ComputeOp<M>),
    /// Image-processing operation (resize family, blur, pad family, …).
    Image(ImageOp<M>),
    /// Color space conversion (RGB ↔ HSV, LAB, YCbCr, BGR, Gray).
    Color(ColorConvertOp<M>),
    /// Generic 2D convolution with arbitrary kernel.
    Filter(ConvolveOp<M>),
}

// PadMode/PadPosition live with the padding kernels; re-exported here
// for the existing `ops::dto::PadMode` import paths.
pub use crate::ops::pad::{PadMode, PadPosition};

impl<M: Mode> ViewDto<M> {
    /// The backing [`Op`] implementation — the single delegation point for
    /// every per-op contract (name, shape, dtype, rank, channel rules).
    /// Adding a variant without an `Op` impl fails to compile here.
    pub fn as_op(&self) -> &dyn Op {
        match self {
            ViewDto::View(op) => op,
            ViewDto::Compute(op) => op,
            ViewDto::Image(op) => op,
            ViewDto::Color(op) => op,
            ViewDto::Filter(op) => op,
        }
    }

    /// Refuse a parameter combination no row can run, from the values this
    /// op knows (all of them once resolved, the literals on the wire).
    pub fn check(&self) -> Result<(), String> {
        match self {
            ViewDto::View(op) => op.check(),
            ViewDto::Compute(op) => op.check(),
            ViewDto::Image(op) => op.kind.check(),
            ViewDto::Color(op) => op.check(),
            ViewDto::Filter(op) => op.check(),
        }
    }

    /// Every single-buffer op consumes a buffer.
    pub fn input_domain(&self) -> Domain {
        Domain::Buffer
    }

    /// Every single-buffer op produces a buffer (domain-changing steps are
    /// graph-level concerns in the plugin's `GraphStep`).
    pub fn output_domain(&self) -> Domain {
        Domain::Buffer
    }

    /// The rule that determines this operation's output dtype.
    pub fn output_dtype_rule(&self) -> crate::core::dtype::OutputDTypeRule {
        self.as_op().output_dtype_rule()
    }

    /// How this operation's output depends on the spatial extent of its input.
    pub fn spatial_dependency(&self) -> crate::ops::spatial_rule::SpatialDependency {
        self.as_op().spatial_dependency()
    }

    /// Under what condition this operation is a removable no-op.
    pub fn identity_rule(&self) -> IdentityRule {
        self.as_op().identity_rule()
    }

    /// Whether this operation is a hoistable H/W spatial window (a crop/ROI).
    pub fn is_spatial_window(&self) -> bool {
        self.as_op().is_spatial_window()
    }

    /// Get the name of this operation for error messages.
    pub fn name(&self) -> &'static str {
        self.as_op().name()
    }
}
