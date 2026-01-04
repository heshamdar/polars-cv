//! Execution plan types.

use crate::core::buffer::ViewBuffer;
use crate::execution::runner::{apply_compute, apply_image, apply_perceptual_hash, apply_view};
use crate::ops::phash::PerceptualHashOp;
use crate::ops::{ComputeOp, ImageOp, ViewOp};

/// A step in the execution plan.
#[derive(Debug, Clone)]
pub enum PlanStep {
    View(ViewOp),
    Compute(ComputeOp),
    Image(ImageOp),
    PerceptualHash(PerceptualHashOp),
    MaterializeContiguous,
}

/// An execution plan built from a ViewExpr graph.
#[derive(Debug)]
pub struct ExecutionPlan {
    pub source: ViewBuffer,
    pub steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    /// Executes the plan and returns the resulting ViewBuffer.
    pub fn execute(self) -> ViewBuffer {
        let mut current_buffer = self.source;

        for step in self.steps {
            match step {
                PlanStep::View(op) => {
                    current_buffer = apply_view(current_buffer, op);
                }
                PlanStep::Compute(op) => {
                    current_buffer = apply_compute(current_buffer, op);
                }
                PlanStep::Image(op) => {
                    current_buffer = apply_image(current_buffer, op);
                }
                PlanStep::PerceptualHash(op) => {
                    current_buffer = apply_perceptual_hash(current_buffer, op);
                }
                PlanStep::MaterializeContiguous => {
                    current_buffer = current_buffer.to_contiguous();
                }
            }
        }
        current_buffer
    }
}
