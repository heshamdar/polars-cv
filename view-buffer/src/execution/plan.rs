//! Execution plan types and the executor.

use crate::core::buffer::ViewBuffer;
use crate::execution::runner::{apply_compute_inner, apply_image_inner, apply_view};
use crate::ops::color::{apply_color_convert, ColorConvertOp};
use crate::ops::filter::{apply_convolve2d, ConvolveOp};
use crate::ops::{ComputeOp, ImageOp, ViewOp};

/// A single step in a flat execution plan.
#[derive(Debug, Clone)]
pub enum PlanStep {
    View(ViewOp),
    Compute(ComputeOp),
    Image(ImageOp),
    /// Color space conversion.
    Color(ColorConvertOp),
    /// 2D convolution.
    Filter(ConvolveOp),
    /// Ensure the buffer is contiguous before passing to the next op.
    MaterializeContiguous,
    /// Ensure each row of the buffer is packed (any row stride) before
    /// passing it to the next op.
    MaterializeDenseRows,
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

    /// Executes `steps` on `source` without owning them: a plan replayed on
    /// every row that shares its source's layout. Each step is cloned only
    /// as it is applied, and most own nothing on the heap.
    pub fn execute_steps(source: ViewBuffer, steps: &[PlanStep]) -> ViewBuffer {
        let mut current = source;
        for step in steps {
            current = apply_step(current, step.clone());
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
        PlanStep::Color(op) => apply_color_convert(&buf, &op),
        PlanStep::Filter(op) => apply_convolve2d(&buf, &op),
        PlanStep::MaterializeContiguous => buf.to_contiguous(),
        PlanStep::MaterializeDenseRows => buf.to_dense_rows(),
    }
}
