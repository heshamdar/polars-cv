//! Unified pipeline graph module.
//!
//! This module provides the graph-based pipeline execution system, including:
//! - `UnifiedGraph`: The main graph structure for multi-output pipelines
//! - `GraphNode`: Individual nodes in the pipeline graph
//! - Source decoding and output encoding utilities

// Internal modules with descriptive names
#[path = "types.rs"]
pub(crate) mod types;

#[path = "decode.rs"]
pub(crate) mod decode;

#[path = "encode.rs"]
pub(crate) mod encode;

// Re-exports for crate-level access
pub(crate) use decode::dtype_for_output;
pub use types::UnifiedGraph;
