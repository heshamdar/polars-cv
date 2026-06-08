//! Execution planning and sequential op dispatch.
//!
//! # Key types
//!
//! * [`ExecutionPlan`] / [`PlanStep`] — compiled plan returned by
//!   [`ViewExpr::plan`](crate::expr::ViewExpr::plan).

pub mod plan;
pub mod runner;

pub use plan::{ExecutionPlan, PlanStep};
pub use runner::{apply_channel_merge, apply_channel_swap, execute_plan};
