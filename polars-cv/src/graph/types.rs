//! Core types for the unified pipeline graph.
//!
//! This module contains the data structures for representing vision pipeline
//! graphs: `UnifiedGraph`, `GraphNode`, `OutputSpec`, etc. Execution lives in
//! [`super::compiled`].

use polars::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use view_buffer::geometry::Contour;
use view_buffer::ViewBuffer;

use crate::params::NullParamPolicy;
use crate::pipeline::{SinkSpec, SourceSpec};

use super::encode::{default_domain, default_dtype};

/// Output specification for a single output in the graph.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
    /// Expected output domain for validation and type inference.
    #[serde(default = "default_domain")]
    pub expected_domain: String,
    /// Expected output dtype for list/array sinks.
    #[serde(default = "default_dtype")]
    pub expected_dtype: String,
    /// Expected output shape for list/array sinks.
    #[serde(default)]
    pub expected_shape: Option<Vec<usize>>,
    /// Expected number of dimensions for list sinks.
    #[serde(default)]
    pub expected_ndim: Option<usize>,
    /// Optional sink encoding selector, independent of the output domain.
    ///
    /// Some outputs share a domain but need a distinct Polars schema. For
    /// example histogram buckets are a `vector`-domain output, but are encoded
    /// as `List(Struct[lower_edge, upper_edge, count, normalized])`. Python sets
    /// this to `"histogram_buckets"` for that case; `None` means encode by the
    /// (domain, format) pair as usual.
    #[serde(default)]
    pub expected_encoding: Option<String>,
}
/// Result type for individual row execution.
///
/// Each variant holds the typed data for a single row output.
/// The Option allows null handling - None represents null input or error.
#[derive(Clone)]
pub(crate) enum RowResult {
    /// Binary data (images, blobs, etc.)
    Binary(Option<Vec<u8>>),
    /// Scalar value (reduce operations)
    Scalar(Option<f64>),
    /// Vector of f64 values
    Vector(Option<Vec<f64>>),
    /// Contour geometry data
    Contours(Option<Vec<Contour>>),
    /// Typed list for "list" sink (variable length, preserves dtype).
    TypedList(Option<(TypedBufferData, Vec<usize>)>),
    /// Typed fixed-size array for "array" sink (fixed shape, preserves dtype).
    TypedArray(Option<(TypedBufferData, Vec<usize>)>),
    /// Numpy/Torch struct output (zero-copy ViewBuffer ownership transfer).
    NumpyStruct(Option<ViewBuffer>),
    /// Histogram buckets data [lower_edge, upper_edge, count, normalized] flattened
    HistogramBuckets(Option<Vec<f64>>),
}

/// Per-row error policy for graph execution.
///
/// Applies to `Result`-level errors while producing a row (source decode,
/// op resolution/execution, output encode). Panics are not covered — they
/// abort the batch via the executor's `catch_unwind` backstop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowErrorPolicy {
    /// Propagate the first error and fail the whole expression (default).
    #[default]
    Raise,
    /// A failing row yields null for all of its outputs; other rows proceed.
    Null,
    /// As `Null`, plus a reserved `_error: String` field in the output
    /// struct carrying the failure message for bad rows.
    NullWithMessage,
}

/// Unified pipeline graph specification.
///
/// This struct handles all cases:
/// - Single output: `outputs` contains only "_output" key, returns Binary
/// - Multi output: `outputs` contains multiple keys, returns Struct
#[derive(Debug, Deserialize)]
pub struct UnifiedGraph {
    /// Graph wire-format version. Version 0 (absent) and 1 are identical;
    /// the field exists so future format changes can be detected instead of
    /// misparsed (also relevant for any persisted graph JSON).
    #[serde(default)]
    pub version: u32,
    /// Per-row error policy for the whole graph.
    #[serde(default)]
    pub on_error: RowErrorPolicy,
    /// What a null in a per-row expression parameter means for the affected
    /// rows. Independent of [`on_error`](Self::on_error): under
    /// [`NullParamPolicy::Null`] a null parameter is not an error at all, so it
    /// yields a null result without weakening error reporting for anything else.
    #[serde(default)]
    pub on_null_param: NullParamPolicy,
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// Output specifications (alias -> spec).
    /// Single output uses "_output" as key.
    pub outputs: HashMap<String, OutputSpec>,
    /// Mapping from node IDs to input column indices.
    /// Only root nodes (no upstream) have bindings.
    #[serde(default)]
    pub column_bindings: HashMap<String, usize>,
    /// Cached topological order (computed once during parsing).
    /// Not serialized - computed on load.
    #[serde(skip)]
    cached_order: Vec<String>,
}

/// The newest graph wire-format version this build understands.
const SUPPORTED_GRAPH_VERSION: u32 = 1;

impl UnifiedGraph {
    /// Parse a graph from JSON.
    ///
    /// This also computes and caches the topological order for efficient
    /// repeated execution.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        let mut graph: Self = serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError : "Failed to parse pipeline graph: {}", e))?;
        if graph.version > SUPPORTED_GRAPH_VERSION {
            polars_bail!(ComputeError:
                "Pipeline graph format version {} is newer than this polars-cv build supports ({}); \
                 upgrade polars-cv",
                graph.version, SUPPORTED_GRAPH_VERSION
            );
        }
        graph.cached_order = graph.compute_topological_order()?;
        Ok(graph)
    }
    /// Check if this is a single-output graph (returns Binary instead of Struct).
    pub fn is_single_output(&self) -> bool {
        self.outputs.len() == 1 && self.outputs.contains_key("_output")
    }
    /// Get cached topological order.
    /// The order is computed once during parsing and reused for all executions.
    pub(crate) fn topological_order(&self) -> &[String] {
        &self.cached_order
    }
    /// Compute nodes in topological order (dependencies first).
    /// Includes all nodes reachable from any output.
    fn compute_topological_order(&self) -> PolarsResult<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut in_stack: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();
        fn dfs(
            node_id: &str,
            nodes: &HashMap<String, GraphNode>,
            visited: &mut HashSet<String>,
            in_stack: &mut HashSet<String>,
            order: &mut Vec<String>,
        ) -> PolarsResult<()> {
            if visited.contains(node_id) {
                return Ok(());
            }
            if in_stack.contains(node_id) {
                polars_bail!(ComputeError: "Cycle detected in graph at node '{}'", node_id);
            }
            in_stack.insert(node_id.to_string());
            if let Some(node) = nodes.get(node_id) {
                for upstream_id in &node.upstream {
                    dfs(upstream_id, nodes, visited, in_stack, order)?;
                }
            }
            in_stack.remove(node_id);
            visited.insert(node_id.to_string());
            order.push(node_id.to_string());
            Ok(())
        }
        for spec in self.outputs.values() {
            dfs(
                &spec.node,
                &self.nodes,
                &mut visited,
                &mut in_stack,
                &mut order,
            )?;
        }
        Ok(order)
    }
}
/// Typed buffer data for dtype-preserving list/array outputs.
#[derive(Debug, Clone)]
pub(crate) enum TypedBufferData {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}
impl TypedBufferData {
    /// Get the number of elements in this typed buffer.
    pub(crate) fn len(&self) -> usize {
        match self {
            TypedBufferData::U8(v) => v.len(),
            TypedBufferData::I8(v) => v.len(),
            TypedBufferData::U16(v) => v.len(),
            TypedBufferData::I16(v) => v.len(),
            TypedBufferData::U32(v) => v.len(),
            TypedBufferData::I32(v) => v.len(),
            TypedBufferData::U64(v) => v.len(),
            TypedBufferData::I64(v) => v.len(),
            TypedBufferData::F32(v) => v.len(),
            TypedBufferData::F64(v) => v.len(),
        }
    }
    /// Extract typed data from a buffer that is already contiguous.
    ///
    /// This avoids the redundant `to_contiguous()` call when the caller
    /// has already materialized the buffer.
    ///
    /// # Panics
    /// Panics if the buffer is not contiguous (via `as_slice` assertion).
    pub(crate) fn from_contiguous_buffer(buf: &ViewBuffer) -> Self {
        // as_slice asserts contiguity internally
        match buf.dtype() {
            view_buffer::DType::U8 => TypedBufferData::U8(buf.as_slice::<u8>().to_vec()),
            view_buffer::DType::I8 => TypedBufferData::I8(buf.as_slice::<i8>().to_vec()),
            view_buffer::DType::U16 => TypedBufferData::U16(buf.as_slice::<u16>().to_vec()),
            view_buffer::DType::I16 => TypedBufferData::I16(buf.as_slice::<i16>().to_vec()),
            view_buffer::DType::U32 => TypedBufferData::U32(buf.as_slice::<u32>().to_vec()),
            view_buffer::DType::I32 => TypedBufferData::I32(buf.as_slice::<i32>().to_vec()),
            view_buffer::DType::U64 => TypedBufferData::U64(buf.as_slice::<u64>().to_vec()),
            view_buffer::DType::I64 => TypedBufferData::I64(buf.as_slice::<i64>().to_vec()),
            view_buffer::DType::F32 => TypedBufferData::F32(buf.as_slice::<f32>().to_vec()),
            view_buffer::DType::F64 => TypedBufferData::F64(buf.as_slice::<f64>().to_vec()),
        }
    }
    /// Get the Polars DataType for this typed data.
    pub(crate) fn polars_dtype(&self) -> DataType {
        match self {
            TypedBufferData::U8(_) => DataType::UInt8,
            TypedBufferData::I8(_) => DataType::Int8,
            TypedBufferData::U16(_) => DataType::UInt16,
            TypedBufferData::I16(_) => DataType::Int16,
            TypedBufferData::U32(_) => DataType::UInt32,
            TypedBufferData::I32(_) => DataType::Int32,
            TypedBufferData::U64(_) => DataType::UInt64,
            TypedBufferData::I64(_) => DataType::Int64,
            TypedBufferData::F32(_) => DataType::Float32,
            TypedBufferData::F64(_) => DataType::Float64,
        }
    }
    /// The view-buffer dtype this variant holds.
    pub(crate) fn dtype(&self) -> view_buffer::DType {
        use view_buffer::DType;
        match self {
            TypedBufferData::U8(_) => DType::U8,
            TypedBufferData::I8(_) => DType::I8,
            TypedBufferData::U16(_) => DType::U16,
            TypedBufferData::I16(_) => DType::I16,
            TypedBufferData::U32(_) => DType::U32,
            TypedBufferData::I32(_) => DType::I32,
            TypedBufferData::U64(_) => DType::U64,
            TypedBufferData::I64(_) => DType::I64,
            TypedBufferData::F32(_) => DType::F32,
            TypedBufferData::F64(_) => DType::F64,
        }
    }

    /// Get the dtype string for this typed data.
    ///
    /// Spelled by `dtype_table!`, not here: this maps a variant to a dtype and
    /// lets that dtype name itself.
    pub(crate) fn dtype_str(&self) -> &'static str {
        self.dtype().short_name()
    }
}
/// Output value from encoding - can be binary, contour struct, scalar, or array.
#[derive(Debug, Clone)]
pub(crate) enum OutputValue {
    Binary(Vec<u8>),
    Contours(Arc<Vec<Contour>>),
    Scalar(f64),
    Vector(Arc<Vec<f64>>),
    /// Typed list representation for "list" sink - preserves buffer dtype.
    TypedList {
        /// Typed data preserving original buffer dtype.
        data: TypedBufferData,
        /// Original shape of the buffer.
        shape: Vec<usize>,
    },
    /// Typed fixed-size array representation for "array" sink.
    TypedArray {
        /// Typed data preserving original buffer dtype.
        data: TypedBufferData,
        /// Fixed shape (validated against buffer).
        shape: Vec<usize>,
    },
    /// Numpy/Torch struct output (zero-copy ViewBuffer for struct encoding).
    NumpyStruct(ViewBuffer),
    /// Histogram buckets data [lower_edge, upper_edge, count, normalized] flattened
    HistogramBuckets(Vec<f64>),
}
/// A node in the pipeline graph.
///
/// `deny_unknown_fields` closes this end of the wire format. It was permissive,
/// so a stale or misspelled key was silently dropped — which is how node-level
/// `shape_hints` went on being serialized long after the last reader was
/// removed. Anything Python sends must be declared here, including the fields
/// only Python consumes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphNode {
    /// Source specification for this node's input.
    pub source: SourceSpec,
    /// Operations to apply.
    #[serde(default)]
    pub ops: Vec<crate::pipeline::OpSpec>,
    /// Upstream node IDs this node depends on.
    #[serde(default)]
    pub upstream: Vec<String>,
    /// Optional user-defined alias for multi-output.
    /// Note: Used for deserialization; alias becomes the key in outputs map.
    #[serde(default)]
    #[allow(dead_code)]
    pub alias: Option<String>,
    /// Planner metadata for graph visualization only; the executor computes
    /// its own schema from `ops`. Declared so the node stays closed.
    #[serde(default)]
    #[allow(dead_code)]
    pub domain: Option<String>,
    /// See [`GraphNode::domain`].
    #[serde(default, rename = "output_dtype")]
    #[allow(dead_code)]
    pub output_dtype: Option<String>,
}
