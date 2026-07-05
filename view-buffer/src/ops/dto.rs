#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ops::color::ColorConvertOp;
use crate::ops::compute::ComputeOp;
use crate::ops::filter::ConvolveOp;
use crate::ops::image::ImageOp;
use crate::ops::phash::PerceptualHashOp;
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::traits::Op;
use crate::ops::view::ViewOp;
use crate::ops::Domain;

/// A pure Data Transfer Object (DTO) for single-buffer operation plans.
///
/// Every variant is backed by an [`Op`] implementation and is executable via
/// [`ViewExpr::apply_op`](crate::expr::ViewExpr::apply_op) — the enum contains
/// exactly what the engine can run, nothing more (guarded by the
/// `apply_op_executes_every_view_dto_variant` coverage test). Graph-level
/// concerns — multi-input operations, node references, expression columns,
/// domain transitions — live in the polars-cv plugin's `GraphStep`, not here.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ViewDto {
    /// Zero-copy layout operation (transpose, reshape, flip, crop, …).
    View(ViewOp),
    /// Element-wise compute operation (cast, scale, clamp, affine, …).
    Compute(ComputeOp),
    /// Image-processing operation (resize family, blur, pad family, …).
    Image(ImageOp),
    /// Color space conversion (RGB ↔ HSV, LAB, YCbCr, BGR, Gray).
    Color(ColorConvertOp),
    /// Generic 2D convolution with arbitrary kernel.
    Filter(ConvolveOp),
    /// Perceptual hash — computes an image fingerprint buffer.
    PerceptualHash(PerceptualHashOp),
}

// PadMode/PadPosition live with the padding kernels; re-exported here
// for the existing `ops::dto::PadMode` import paths.
pub use crate::ops::pad::{PadMode, PadPosition};

impl ViewDto {
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
            ViewDto::PerceptualHash(op) => op,
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

    /// The rule that determines how this operation transforms the input rank.
    pub fn output_rank_rule(&self) -> OutputRankRule {
        self.as_op().output_rank_rule()
    }

    /// The rule that determines how this operation transforms the channel
    /// count (the trailing dimension of an `[H, W, C]` buffer).
    pub fn output_channel_rule(&self) -> OutputChannelRule {
        self.as_op().output_channel_rule()
    }

    /// Get the name of this operation for error messages.
    pub fn name(&self) -> &'static str {
        self.as_op().name()
    }

    /// Validate that this operation can receive input from the given domain.
    ///
    /// Returns an error with a helpful message if the domains are incompatible.
    pub fn validate_input_domain(&self, input_domain: Domain) -> Result<(), String> {
        let expected = self.input_domain();
        if expected.accepts(input_domain) {
            Ok(())
        } else {
            Err(format!(
                "{}() expects {} input but pipeline is currently in {} domain. \
                 Add a domain-converting operation (e.g., rasterize() or extract_contours()).",
                self.name(),
                expected.name(),
                input_domain.name()
            ))
        }
    }
}
