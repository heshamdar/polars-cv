//! Unified pipeline graph module.
//!
//! This module provides the graph-based pipeline execution system, including:
//! - `UnifiedGraph`: The main graph structure for multi-output pipelines
//! - `GraphNode`: Individual nodes in the pipeline graph
//! - Source decoding and output encoding utilities

// Internal modules with descriptive names
pub(crate) mod node_output;
#[path = "types.rs"]
pub(crate) mod types;

#[path = "compiled.rs"]
pub(crate) mod compiled;
pub(crate) mod row;
pub(crate) mod step;

pub(crate) mod sink_kind;

#[cfg(test)]
mod copy_census;

pub(crate) mod sink;
pub(crate) mod source;

// Re-exports for crate-level access
pub(crate) use compiled::{get_or_compile, resolved_output_specs};
pub(crate) use sink::dtype_for_output;
pub(crate) use types::RowErrorPolicy;
