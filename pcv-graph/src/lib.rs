//! `pcv-graph`: typed IR, op registry, and executor for polars-cv pipelines.
//!
//! This crate is the single source of truth for the polars-cv pipeline graph.
//! It owns:
//!
//! - [`contract`]: plan-time op contracts (dtype/ndim/alpha effects).
//! - [`registry`]: inventory-based registration of ops, sources, and sinks.
//!
//! Higher layers (`polars-cv` PyO3 plugin, codegen for the Python contract
//! table) are driven by what is registered here.

pub mod contract;
pub mod registry;

pub use contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
pub use registry::{OpRegistration, OpSchemaDescriptor, ParamKind, ParamSchema};

/// Wire format version stamped on every encoded graph payload.
///
/// Increment when the on-wire `Graph` representation changes in a way that
/// breaks decoders built against an older version.
pub const WIRE_VERSION: u32 = 2;
