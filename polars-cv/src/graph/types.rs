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

use crate::formats::sink::Sink;
use crate::formats::source::Source;
use crate::params::NullParamPolicy;

use crate::plan::State;

/// One requested output as it crosses the wire: the node and its sink.
/// Everything else about the output is planned from the graph.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputRequest {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: Sink,
}

/// An output as planned: its sink plus the facts the planner derived for it
/// ([`resolved_output_specs`](super::compiled::resolved_output_specs)).
/// Never deserialized: nothing about an output's schema is taken from the
/// wire.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: Sink,
    /// Planned output domain.
    pub expected_domain: view_buffer::ops::Domain,
    /// Planned output element dtype.
    pub expected_dtype: view_buffer::PlannedDType,
    /// Planned `[H, W, C]` shape, when the output is rank 3 and all three
    /// sizes are known.
    pub expected_shape: Option<Vec<usize>>,
    /// Planned rank.
    pub expected_ndim: Option<usize>,
    /// The output is histogram buckets: a `vector`-domain output encoded as
    /// `List(Struct[lower_edge, upper_edge, count, normalized])` rather than by
    /// its (domain, format) pair.
    pub histogram_buckets: bool,
}

impl OutputSpec {
    /// The spec for `out`, whose node the planner left in `planned`.
    pub(crate) fn planned(out: &OutputRequest, planned: &State, histogram_buckets: bool) -> Self {
        // A shape is published only for a rank-3 `[H, W, C]` output whose three
        // sizes are all known: the state tracks H/W/C, so at any other rank it
        // cannot describe the shape — publishing `[H, W, C]` for a rank-2
        // output is how `channel_select` once declared a schema execution
        // could not produce.
        let expected_shape = (planned.ndim == Some(3))
            .then(|| planned.dims.iter().copied().collect::<Option<Vec<_>>>())
            .flatten();
        OutputSpec {
            node: out.node.clone(),
            sink: out.sink.clone(),
            expected_domain: planned.domain,
            expected_dtype: planned.dtype,
            expected_shape,
            expected_ndim: planned.ndim,
            histogram_buckets,
        }
    }
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

impl RowResult {
    /// The variant's name, for internal-error messages.
    pub(crate) fn variant_name(&self) -> &'static str {
        match self {
            RowResult::Binary(_) => "Binary",
            RowResult::Scalar(_) => "Scalar",
            RowResult::Vector(_) => "Vector",
            RowResult::Contours(_) => "Contours",
            RowResult::TypedList(_) => "TypedList",
            RowResult::TypedArray(_) => "TypedArray",
            RowResult::NumpyStruct(_) => "NumpyStruct",
            RowResult::HistogramBuckets(_) => "HistogramBuckets",
        }
    }
}

/// Per-row error policy for graph execution.
///
/// Applies to `Result`-level errors while producing a row (source decode,
/// op resolution/execution, output encode), including engine panics, which the
/// executor catches per row and treats as that row's error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

view_buffer::naming::named_variants!(RowErrorPolicy: "What a failing row does to a graph query.\n\nApplies to errors raised while producing a row — source decode, op\nexecution, output encode:\n- RAISE: propagate the first error, failing the whole expression.\n- NULL: a failing row yields null; other rows proceed.\n- NULL_WITH_MESSAGE: as NULL, plus an `_error` field." {
    "raise" => Raise,
    "null" => Null,
    "null_with_message" => NullWithMessage,
});

#[cfg(test)]
mod row_error_policy_tests {
    use super::{OutputSpec, UnifiedGraph};

    fn graph_with(field: &str, value: &str) -> String {
        UnifiedGraph::from_json(&format!(
            r#"{{"nodes": {{}}, "outputs": {{}}, "{field}": "{value}"}}"#
        ))
        .map(|_| String::new())
        .unwrap_or_else(|e| e.to_string())
    }

    /// The one output of a single `image_bytes` node running `ops`, as planned.
    fn output(ops: &str) -> Result<OutputSpec, String> {
        let graph = UnifiedGraph::from_json(&format!(
            r#"{{"nodes": {{"n0": {{"source": {{"format": "image_bytes"}}, "ops": {ops}}}}},
                "outputs": {{"_output": {{"node": "n0", "sink": {{"format": "numpy"}}}}}},
                "column_bindings": {{"n0": 0}}}}"#
        ))
        .map_err(|e| e.to_string())?;
        crate::graph::resolved_output_specs(&graph, &[])
            .map(|mut specs| specs.remove(0).1)
            .map_err(|e| e.to_string())
    }

    /// The output facts are planned from the graph's own ops: a shape only for
    /// rank 3 with all three sizes known.
    #[test]
    fn output_facts_are_planned_from_the_ops() {
        let full = output(
            r#"[{"op": "cast", "dtype": "u8"},
                {"op": "assert_shape", "rank": null, "dims": [4, 5, 3]}]"#,
        )
        .unwrap();
        assert_eq!(
            (full.expected_domain.name(), full.expected_dtype.as_str()),
            ("buffer", "u8")
        );
        assert_eq!(full.expected_shape, Some(vec![4, 5, 3]));
        assert_eq!(full.expected_ndim, Some(3));
        // Channels unknown: no shape, though H and W are known.
        let partial =
            output(r#"[{"op": "resize", "height": 4, "width": 5, "filter": "nearest"}]"#).unwrap();
        assert_eq!(partial.expected_shape, None);
        // Rank 2: the H/W/C state cannot describe the shape.
        let rank2 = output(
            r#"[{"op": "assert_shape", "rank": null, "dims": [4, 5, 3]},
                {"op": "channel_select", "index": 0}]"#,
        )
        .unwrap();
        assert_eq!((rank2.expected_shape, rank2.expected_ndim), (None, Some(2)));
    }

    /// Nothing about an output's schema is taken from the wire: the old
    /// planned-state and expected-* fields are refused by name.
    #[test]
    fn an_output_carries_only_its_node_and_sink() {
        for extra in [
            r#""planned": {"domain": "buffer", "dtype": "u8"}"#,
            r#""expected_dtype": "u8""#,
        ] {
            let err = UnifiedGraph::from_json(&format!(
                r#"{{"nodes": {{}}, "outputs": {{"_output": {{"node": "n0",
                    "sink": {{"format": "numpy"}}, {extra}}}}}}}"#
            ))
            .unwrap_err()
            .to_string();
            let field = extra.split('"').nth(1).unwrap();
            assert!(err.contains(&format!("unknown field `{field}`")), "{err}");
        }
    }

    /// Histogram buckets are recognised from the node's own last step (or its
    /// lineage's, through an op-less node), not from a wire field Python had to
    /// remember to set.
    #[test]
    fn histogram_buckets_are_read_off_the_ops() {
        let graph = |ops: &str| {
            let g = UnifiedGraph::from_json(&format!(
                r#"{{"nodes": {{"n0": {{"source": {{"format": "image_bytes"}}, "ops": {ops}}},
                     "n1": {{"source": {{"format": "blob"}}, "ops": [], "upstream": ["n0"]}}}},
                    "outputs": {{"a": {{"node": "n0", "sink": {{"format": "native"}}}},
                                 "b": {{"node": "n1", "sink": {{"format": "native"}}}}}},
                    "column_bindings": {{"n0": 0}}}}"#
            ))
            .unwrap();
            crate::graph::resolved_output_specs(&g, &[]).unwrap()
        };
        let buckets = r#"[{"op": "histogram", "bins": 4, "range": null, "closed": "left", "output": "buckets"}]"#;
        let specs = graph(buckets);
        assert!(specs.iter().all(|(_, s)| s.histogram_buckets));
        let counts = buckets.replace("buckets", "counts");
        assert!(!graph(&counts)[0].1.histogram_buckets);
    }

    /// An engine toggle the engine does not have is refused, not ignored:
    /// `OptConfig` and the Python `OptFlags` come from one list
    /// (`engine_passes!`), and a stray key means the two builds disagree.
    #[test]
    fn an_unknown_engine_toggle_is_refused() {
        let graph = |opt: &str| {
            UnifiedGraph::from_json(&format!(
                r#"{{"nodes": {{}}, "outputs": {{}}, "opt": {opt}}}"#
            ))
            .map(|_| String::new())
            .unwrap_or_else(|e| e.to_string())
        };
        assert_eq!(graph(r#"{"scalar_fusion": false}"#), "");
        let err = graph(r#"{"scalar_fusoin": false}"#);
        assert!(err.contains("unknown field `scalar_fusoin`"), "{err}");
    }

    /// The graph's policies parse through their `NAMED` tables, the same
    /// spellings the generated Python enums send: one vocabulary, no serde
    /// `rename_all` beside it to keep in step.
    #[test]
    fn graph_policies_parse_through_their_named_tables() {
        for (field, enum_name, good, bad) in [
            (
                "on_error",
                "RowErrorPolicy",
                "null_with_message",
                "NullWithMessage",
            ),
            ("on_null_param", "NullParamPolicy", "null", "Null"),
        ] {
            assert_eq!(graph_with(field, good), "", "{field}={good}");
            let err = graph_with(field, bad);
            let named = format!("unknown {enum_name} \"{bad}\", expected one of");
            assert!(err.contains(&named), "{field}={bad}: {err}");
        }
    }
}

/// Unified pipeline graph specification.
///
/// This struct handles all cases:
/// - Single output: `outputs` contains only "_output" key, returns Binary
/// - Multi output: `outputs` contains multiple keys, returns Struct
///
/// Closed with `deny_unknown_fields` for the same reason `GraphNode` is: a
/// misspelled top-level key (`on_eror`) otherwise took the policy default
/// with nothing said.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnifiedGraph {
    /// Graph wire-format version. Version 0 (absent) and 1 are identical;
    /// the field exists so future format changes can be detected instead of
    /// misparsed (also relevant for any persisted graph JSON).
    #[serde(default)]
    pub version: u32,
    /// Per-row error policy for the whole graph.
    #[serde(default, deserialize_with = "crate::ops::param::literal_field")]
    pub on_error: RowErrorPolicy,
    /// What a null in a per-row expression parameter means for the affected
    /// rows. Independent of [`on_error`](Self::on_error): under
    /// [`NullParamPolicy::Null`] a null parameter is not an error at all, so it
    /// yields a null result without weakening error reporting for anything else.
    #[serde(default, deserialize_with = "crate::ops::param::literal_field")]
    pub on_null_param: NullParamPolicy,
    /// Which engine-tier (Tier-2) optimizations to apply when executing buffer-op
    /// chains. Absent (older specs) or partially specified means all enabled, via
    /// [`OptConfig`](view_buffer::OptConfig)'s serde default — the historical
    /// behavior. The Python planner emits it from the engine-tier `OptFlags`.
    #[serde(default)]
    pub opt: view_buffer::OptConfig,
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// The requested outputs (alias -> node and sink).
    /// Single output uses "_output" as key.
    pub outputs: HashMap<String, OutputRequest>,
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

    /// Whether `node_id`'s buffer is histogram buckets: its last step, or, for
    /// a node with no ops, its primary upstream's.
    pub(crate) fn ends_in_histogram_buckets(&self, node_id: &str) -> bool {
        let Some(node) = self.nodes.get(node_id) else {
            return false;
        };
        match node.ops.last() {
            Some(op) => matches!(
                op,
                crate::graph::step::GraphStep::Histogram(h)
                    if h.output.get() == view_buffer::ops::HistogramOutput::Buckets
            ),
            None => node
                .upstream
                .first()
                .is_some_and(|up| self.ends_in_histogram_buckets(up)),
        }
    }
    /// What the input column will supply to `node_id`'s lineage, when it
    /// starts at a root whose source takes facts from the column; `None`
    /// when the plan holds everything already.
    pub(crate) fn column_facts_pending(
        &self,
        node_id: &str,
    ) -> Option<crate::graph::decode::ColumnFacts> {
        let node = self.nodes.get(node_id)?;
        match node.upstream.first() {
            Some(up) if !self.column_bindings.contains_key(node_id) => {
                self.column_facts_pending(up)
            }
            _ => node.source.resolves_from_column().then(|| {
                crate::graph::decode::ColumnFacts::Pending {
                    sizes: node.source.column_may_fix_sizes(),
                }
            }),
        }
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
    pub source: Source,
    /// Operations to apply.
    #[serde(default)]
    pub ops: Vec<crate::ops::TypedOp>,
    /// Upstream node IDs this node depends on.
    #[serde(default)]
    pub upstream: Vec<String>,
}
