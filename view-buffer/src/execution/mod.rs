//! Execution planning and sequential op dispatch.
//!
//! # Key types
//!
//! * [`ExecutionPlan`] / [`PlanStep`] — compiled plan returned by
//!   [`ViewExpr::plan`](crate::expr::ViewExpr::plan).

pub(crate) mod distance;
pub mod plan;
mod resample;
pub mod runner;
mod warp;

pub use plan::{ExecutionPlan, PlanStep};
pub use runner::{apply_channel_merge, apply_channel_swap, execute_plan, validate_channel_merge};
