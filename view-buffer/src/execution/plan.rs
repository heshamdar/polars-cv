//! Execution plan types and the strategy-driven executor.

use crate::core::buffer::ViewBuffer;
use crate::execution::strategy::ExecutionStrategy;
use crate::execution::tiling::{apply_step_full, execute_segmented_tiled};
use crate::ops::{ComputeOp, ImageOp, ViewOp};
use crate::ops::phash::PerceptualHashOp;

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
///
/// The plan carries an [`ExecutionStrategy`] that was captured from the
/// thread-local default when [`ViewExpr::plan`] was called.  Call
/// [`ExecutionPlan::with_strategy`] to override it before executing.
#[derive(Debug)]
pub struct ExecutionPlan {
    pub source: ViewBuffer,
    pub steps: Vec<PlanStep>,
    /// Strategy captured at plan-build time (from thread-local default).
    pub strategy: ExecutionStrategy,
}

impl ExecutionPlan {
    /// Override the strategy stored in this plan.
    ///
    /// Useful when you want to run the same compiled plan multiple times
    /// with different strategies (e.g. A/B benchmarking).
    pub fn with_strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Executes the plan and returns the resulting [`ViewBuffer`].
    ///
    /// The execution path is chosen based on the plan's strategy and the
    /// actual byte footprint of the source buffer:
    ///
    /// * Small images (or `FullImage` strategy) → one sequential pass per op.
    /// * Large images with a tiling strategy → segment-level outer-loop
    ///   tiling where consecutive tileable ops share each tile.
    pub fn execute(self) -> ViewBuffer {
        let image_bytes = self.source.size_bytes();
        let tile_size = self.strategy.resolve(image_bytes);

        match tile_size {
            None => execute_full(self.source, self.steps),
            Some(ts) => execute_segmented_tiled(self.source, self.steps, ts),
        }
    }
}

/// Executes `steps` on `source` with a single sequential pass over the full
/// buffer for each step (the original behaviour, optimal for small images).
pub(crate) fn execute_full(source: ViewBuffer, steps: Vec<PlanStep>) -> ViewBuffer {
    let mut current = source;
    for step in steps {
        current = apply_step_full(current, step);
    }
    current
}
