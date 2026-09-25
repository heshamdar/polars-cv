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

/// Output specification for a single output in the graph.
///
/// Deserialized from [`WireOutput`]: the wire carries the output node's final
/// planned [`State`], and the `expected_*` facts below are read off it here,
/// in one place, rather than each projected by the Python planner.
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "WireOutput")]
pub struct OutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: Sink,
    /// Expected output domain for validation and type inference.
    pub expected_domain: String,
    /// Expected output dtype for list/array sinks.
    pub expected_dtype: String,
    /// Expected output shape for list/array sinks.
    pub expected_shape: Option<Vec<usize>>,
    /// Did any dimension of the plan come from a user `assert_shape`?
    ///
    /// Decides who [`validate_output_schema`](super::compiled) reports a
    /// plan/exec divergence against. An inferred shape that execution
    /// contradicts is a contract bug — a rule lying about its transform. An
    /// *asserted* one is a claim about the caller's data, and reporting it as
    /// "the planner's shape contract disagrees with the Rust implementation"
    /// sent people to read plugin source over their own typo.
    pub shape_asserted: bool,
    /// Expected number of dimensions for list sinks.
    pub expected_ndim: Option<usize>,
    /// The output is histogram buckets: a `vector`-domain output encoded as
    /// `List(Struct[lower_edge, upper_edge, count, normalized])` rather than by
    /// its (domain, format) pair. Read off the output node's ops at load
    /// ([`UnifiedGraph::from_json`]), never sent on the wire.
    pub histogram_buckets: bool,
}

/// An output as it crosses the wire. Closed like `GraphNode`:
/// `deny_unknown_fields` does not descend, so this sibling needed its own.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOutput {
    node: String,
    sink: Sink,
    /// The output node's final planned state.
    planned: State,
}

impl From<WireOutput> for OutputSpec {
    fn from(wire: WireOutput) -> Self {
        let WireOutput {
            node,
            sink,
            planned,
        } = wire;
        // A shape is published only for a rank-3 `[H, W, C]` output whose three
        // sizes are all known: the state tracks H/W/C, so at any other rank it
        // cannot describe the shape — publishing `[H, W, C]` for a rank-2
        // output is how `channel_select` once declared a schema execution
        // could not produce.
        let expected_shape = (planned.ndim == Some(3))
            .then(|| {
                planned
                    .dims
                    .iter()
                    .map(|d| d.and_then(|n| usize::try_from(n).ok()))
                    .collect::<Option<Vec<_>>>()
            })
            .flatten();
        OutputSpec {
            node,
            sink,
            expected_domain: planned.domain,
            expected_dtype: planned.dtype,
            expected_shape,
            shape_asserted: planned.asserted.iter().any(|a| *a),
            expected_ndim: planned.ndim,
            histogram_buckets: false,
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

    /// One output over a single `image_bytes` node, planned as `planned`.
    fn output(planned: &str) -> Result<OutputSpec, String> {
        UnifiedGraph::from_json(&format!(
            r#"{{"nodes": {{"n0": {{"source": {{"format": "image_bytes"}}, "ops": []}}}},
                "outputs": {{"_output": {{"node": "n0", "sink": {{"format": "numpy"}}{planned}}}}}}}"#
        ))
        .map(|g| g.outputs["_output"].clone())
        .map_err(|e| e.to_string())
    }

    /// The output facts are read off the node's planned state: a shape only
    /// for rank 3 with all three sizes known, "asserted" if any dimension was.
    #[test]
    fn output_facts_are_read_off_the_planned_state() {
        let spec = |state: &str| output(&format!(r#", "planned": {state}"#)).unwrap();
        let full = spec(r#"{"domain": "buffer", "dtype": "u8", "ndim": 3, "dims": [4, 5, 3]}"#);
        assert_eq!(
            (full.expected_domain.as_str(), full.expected_dtype.as_str()),
            ("buffer", "u8")
        );
        assert_eq!(full.expected_shape, Some(vec![4, 5, 3]));
        assert_eq!(full.expected_ndim, Some(3));
        assert!(!full.shape_asserted);
        // Channels unknown: no shape, though H and W are known.
        let partial =
            spec(r#"{"domain": "buffer", "dtype": "u8", "ndim": 3, "dims": [4, 5, null]}"#);
        assert_eq!(partial.expected_shape, None);
        // Rank 2: the H/W/C state cannot describe the shape.
        let rank2 = spec(r#"{"domain": "buffer", "dtype": "u8", "ndim": 2, "dims": [4, 5, 3]}"#);
        assert_eq!((rank2.expected_shape, rank2.expected_ndim), (None, Some(2)));
        let asserted = spec(
            r#"{"domain": "buffer", "dtype": "f32", "ndim": 3, "dims": [8, 8, 3],
                "asserted": [false, true, false], "declared": true}"#,
        );
        assert!(asserted.shape_asserted);
    }

    /// An output without its planned state is refused rather than defaulted
    /// (the Python side once fell back to `"buffer"`/`"u8"` for it).
    #[test]
    fn an_output_without_its_planned_state_is_refused() {
        let err = output("").unwrap_err();
        assert!(err.contains("missing field `planned`"), "{err}");
        let err = output(r#", "planned": {"dtype": "u8"}"#).unwrap_err();
        assert!(err.contains("missing field `domain`"), "{err}");
        let err =
            output(r#", "planned": {"domain": "buffer", "dtype": "u8"}, "expected_dtype": "u8""#)
                .unwrap_err();
        assert!(err.contains("unknown field `expected_dtype`"), "{err}");
    }

    /// Histogram buckets are recognised from the node's own last step (or its
    /// lineage's, through an op-less node), not from a wire field Python had to
    /// remember to set.
    #[test]
    fn histogram_buckets_are_read_off_the_ops() {
        let planned = r#""planned": {"domain": "vector", "dtype": "f64"}"#;
        let graph = |ops: &str, extra: &str| {
            UnifiedGraph::from_json(&format!(
                r#"{{"nodes": {{"n0": {{"source": {{"format": "image_bytes"}}, "ops": {ops}}},
                     "n1": {{"source": {{"format": "blob"}}, "ops": [], "upstream": ["n0"]}}}},
                    "outputs": {{"a": {{"node": "n0", "sink": {{"format": "native"}}, {planned}{extra}}},
                                 "b": {{"node": "n1", "sink": {{"format": "native"}}, {planned}}}}}}}"#
            ))
        };
        let buckets = r#"[{"op": "histogram", "bins": 4, "range": null, "closed": "left", "output": "buckets"}]"#;
        let g = graph(buckets, "").unwrap();
        assert!(g.outputs["a"].histogram_buckets && g.outputs["b"].histogram_buckets);
        let counts = buckets.replace("buckets", "counts");
        assert!(!graph(&counts, "").unwrap().outputs["a"].histogram_buckets);
        let err = graph(buckets, r#", "expected_encoding": "histogram_buckets""#).unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown field `expected_encoding`"),
            "{err}"
        );
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
        let buckets: Vec<(String, bool)> = graph
            .outputs
            .iter()
            .map(|(alias, spec)| (alias.clone(), graph.ends_in_histogram_buckets(&spec.node)))
            .collect();
        for (alias, flag) in buckets {
            if let Some(spec) = graph.outputs.get_mut(&alias) {
                spec.histogram_buckets = flag;
            }
        }
        Ok(graph)
    }

    /// Whether `node_id`'s buffer is histogram buckets: its last step, or, for
    /// a node with no ops, its primary upstream's.
    fn ends_in_histogram_buckets(&self, node_id: &str) -> bool {
        let Some(node) = self.nodes.get(node_id) else {
            return false;
        };
        match node.ops.last() {
            Some(op) => {
                let json = serde_json::to_string(op).expect("an op always serializes");
                matches!(
                    crate::resolve_op_from_json(&json),
                    Ok(crate::graph::step::GraphStep::Histogram(h))
                        if h.output == view_buffer::ops::HistogramOutput::Buckets
                )
            }
            None => node
                .upstream
                .first()
                .is_some_and(|up| self.ends_in_histogram_buckets(up)),
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
