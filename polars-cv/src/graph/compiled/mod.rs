//! Compiled, cacheable form of a pipeline graph.
//!
//! [`CompiledGraph`] is a pure function of the plugin kwargs (`graph_json`):
//! JSON parsing, topological ordering, nested-parameter hoisting, and static
//! (all-literal) op resolution
//! all happen once at compile time. Because the plugin is registered as
//! elementwise, the streaming engine invokes it once **per morsel** — the
//! process-wide cache ([`get_or_compile`]) makes repeat invocations pay only
//! a hash lookup instead of re-parsing and re-compiling the graph.
//!
//! # Cache-safety invariants
//!
//! A `CompiledGraph` must contain **nothing derived from the data**:
//!
//! - no input dtypes (the `"auto"` dtype/ndim resolution from the input
//!   column type happens per call in [`resolved_output_specs`]),
//! - no shapes, row counts, null masks, or any first-row-derived facts.
//!
//! Per-row decode, per-row shapes, and per-row nulls remain exactly per-row,
//! so one cached graph is valid for inputs of any size, dtype mix, or null
//! pattern. A cache hit additionally requires **full string equality** of the
//! kwargs — the hash alone is never trusted.

mod cache;
mod compile;
mod run;
#[cfg(test)]
mod tests;

pub(crate) use cache::get_or_compile;
pub(crate) use compile::resolved_output_specs;
#[cfg(test)]
pub(crate) use compile::CompiledGraph;
