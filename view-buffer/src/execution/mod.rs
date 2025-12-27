//! Execution planning and running for ViewExpr graphs.
//!
//! This module contains:
//! - [`ExecutionPlan`] - A plan built from a ViewExpr graph
//! - [`PlanStep`] - Individual steps in an execution plan
//! - [`execute_plan`] - High-level entry point for executing plans

mod plan;
mod runner;

pub use plan::{ExecutionPlan, PlanStep};
pub use runner::execute_plan;
