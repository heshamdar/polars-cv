//! Execution plan types and the executor.

use crate::core::buffer::ViewBuffer;
use crate::ops::ViewDto;

/// A single step in a flat execution plan.
#[derive(Debug, Clone)]
pub enum PlanStep {
    /// An engine op of any family, run by [`ViewDto::execute`].
    Op(ViewDto),
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
        PlanStep::Op(op) => op.execute(buf),
        PlanStep::MaterializeContiguous => buf.to_contiguous(),
        PlanStep::MaterializeDenseRows => buf.to_dense_rows(),
    }
}
