//! Execution planning, strategy selection, and tiled execution.
//!
//! # Key types
//!
//! * [`ExecutionPlan`] / [`PlanStep`] — compiled plan returned by
//!   [`ViewExpr::plan`](crate::expr::ViewExpr::plan).
//! * [`ExecutionStrategy`] — controls full-image vs. segment-level tiling.
//! * [`TilePolicy`] — declares how an op interacts with spatial locality
//!   (used by op implementations to declare tileability).
//!
//! # Controlling the strategy
//!
//! ```ignore
//! use view_buffer::{ExecutionStrategy, set_execution_strategy, with_execution_strategy};
//!
//! // Force full-image (no tiling) for this thread:
//! set_execution_strategy(ExecutionStrategy::FullImage);
//!
//! // Use explicit tile size with a custom threshold:
//! set_execution_strategy(ExecutionStrategy::Tiled {
//!     tile_size: 128,
//!     threshold_bytes: 256 * 1024,
//! });
//!
//! // Restore after a scoped region:
//! let result = with_execution_strategy(ExecutionStrategy::FullImage, || {
//!     expr.plan().execute()
//! });
//! ```

pub mod plan;
pub mod runner;
pub mod strategy;
pub mod tiling;

pub use plan::{ExecutionPlan, PlanStep};
pub use runner::{apply_channel_merge, apply_channel_swap, execute_plan};
pub use strategy::{
    get_execution_strategy, set_execution_strategy, with_execution_strategy, ExecutionStrategy,
    DEFAULT_TILE_SIZE, ADAPTIVE_THRESHOLD_BYTES,
};
pub use tiling::TilePolicy;
