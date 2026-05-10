//! `pcv-graph`: typed IR, op registry, and executor for polars-cv pipelines.
//!
//! This crate is the single source of truth for the polars-cv pipeline graph.
//! It owns:
//!
//! - [`contract`]: plan-time op contracts (dtype/ndim/alpha effects).
//! - [`op`]: the [`Operation`](op::Operation) trait + helper types.
//! - [`source`] / [`sink`]: per-row source and sink traits.
//! - [`params`]: parameter map type used as the bridge between the wire
//!   format and op factories.
//! - [`registry`]: inventory-based registration of ops, sources, and sinks.
//! - [`ops`]: built-in op adapters (one file per op).
//!
//! Higher layers (`polars-cv` PyO3 plugin, codegen for the Python contract
//! table) are driven by what is registered here.

pub mod contract;
pub mod op;
pub mod ops;
pub mod params;
pub mod registry;
pub mod sink;
pub mod source;

pub use contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
pub use op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
pub use params::{ParamMap, ParamValue};
pub use registry::{
    find_op, iter_ops, OpRegistration, OpSchemaDescriptor, ParamKind, ParamSchema,
};
pub use sink::{find_sink, iter_sinks, Sink, SinkRegistration, SinkRowOutput};
pub use source::{find_source, iter_sources, Source, SourceRegistration};

pub use view_buffer::ops::{Domain, NodeOutput};

/// Wire format version stamped on every encoded graph payload.
///
/// Increment when the on-wire `Graph` representation changes in a way that
/// breaks decoders built against an older version.
pub const WIRE_VERSION: u32 = 2;
