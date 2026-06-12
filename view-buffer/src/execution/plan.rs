//! Execution plan types and the executor.

use crate::core::buffer::ViewBuffer;
use crate::execution::runner::{
    apply_compute_inner, apply_image_inner, apply_perceptual_hash, apply_view,
};
use crate::ops::phash::PerceptualHashOp;
use crate::ops::{ComputeOp, ImageOp, ViewOp};

/// A single step in a flat execution plan.
#[derive(Debug, Clone)]
pub enum PlanStep {
    View(ViewOp),
    Compute(ComputeOp),
    Image(ImageOp),
    PerceptualHash(PerceptualHashOp),
    /// Ensure the buffer is contiguous before passing to the next op.
    MaterializeContiguous,
}

/// A compiled execution plan: a source buffer and an ordered list of steps.
#[derive(Debug)]
pub struct ExecutionPlan {
    pub source: ViewBuffer,
    pub steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    /// Executes the plan and returns the resulting [`ViewBuffer`].
    pub fn execute(self) -> ViewBuffer {
        let mut current = self.source;
        for step in self.steps {
            current = apply_step(current, step);
        }
        current
    }
}

/// Applies a single plan step to the full buffer.
pub(crate) fn apply_step(buf: ViewBuffer, step: PlanStep) -> ViewBuffer {
    match step {
        PlanStep::View(op) => apply_view(buf, op),
        PlanStep::Compute(op) => apply_compute_inner(buf, op),
        PlanStep::Image(op) => apply_image_inner(buf, op),
        PlanStep::PerceptualHash(op) => apply_perceptual_hash(buf, op),
        PlanStep::MaterializeContiguous => buf.to_contiguous(),
    }
}
