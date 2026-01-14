//! Execution planning and running for ViewExpr graphs.
//!
//! This module contains:
//! - [`ExecutionPlan`] - A plan built from a ViewExpr graph
//! - [`PlanStep`] - Individual steps in an execution plan
//! - [`execute_plan`] - High-level entry point for executing plans
//! - [`TileConfig`], [`TilePolicy`] - Tiled execution configuration
//! - [`with_tile_config`] - Scoped tile configuration

mod plan;
mod runner;
pub mod tiling;

pub use plan::{ExecutionPlan, PlanStep};
pub use runner::execute_plan;
pub use tiling::{
    configure_tiling, get_tile_config, is_tiling_enabled, set_tile_config, with_tile_config,
    TileConfig, TilePolicy,
};
