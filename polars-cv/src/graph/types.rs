//! Core types for the unified pipeline graph.
//!
//! This module contains the data structures for representing vision pipeline
//! graphs: `UnifiedGraph`, `GraphNode`, `OutputSpec`, etc. Execution lives in
//! [`super::compiled`].

use polars::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use view_buffer::geometry::contour::Outline;
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
    /// Planned channel count: 1 for an `[H, W]` buffer, axis 2's size for
    /// an `[H, W, C]` one, when known (with or without the other sizes).
    pub expected_channels: Option<usize>,
    /// The output is histogram buckets: a `vector`-domain output encoded as
    /// `List(Struct[lower_edge, upper_edge, count, normalized])` rather than by
    /// its (domain, format) pair.
    pub histogram_buckets: bool,
    /// The `long` sink's layout, parsed once for the call rather than per
    /// row (`None` for every other sink).
    pub long_layout: Option<Result<crate::formats::sink::LongLayout, String>>,
}

impl OutputSpec {
    /// The spec for `out`, whose node the planner left in `planned`.
    pub(crate) fn planned(out: &OutputRequest, planned: &State, histogram_buckets: bool) -> Self {
        // A buffer's shape is published whenever its rank and every size are
        // known, at any rank. Other domains ride a raw buffer whose shape is
        // not the logical one (histogram buckets behind a rank-1 vector).
        let expected_shape = match planned.domain {
            view_buffer::ops::Domain::Buffer => planned.shape.concrete(),
            view_buffer::ops::Domain::Vector
            | view_buffer::ops::Domain::Scalar
            | view_buffer::ops::Domain::Contour => None,
        };
        OutputSpec {
            node: out.node.clone(),
            sink: out.sink.clone(),
            expected_domain: planned.domain,
            expected_dtype: planned.dtype,
            expected_shape,
            expected_ndim: planned.shape.rank(),
            expected_channels: match (planned.domain, planned.shape.rank()) {
                (view_buffer::ops::Domain::Buffer, Some(2)) => Some(1),
                (view_buffer::ops::Domain::Buffer, Some(3)) => planned.shape.sizes()[2],
                _ => None,
            },
            histogram_buckets,
            long_layout: out.sink.long_layout(),
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
    Contours(Option<Vec<Outline>>),
    /// Typed list for "list" sink (variable length, preserves dtype).
    TypedList(Option<ViewBuffer>),
    /// Typed fixed-size array for "array" sink (fixed shape, preserves dtype).
    TypedArray(Option<ViewBuffer>),
    /// Numpy/Torch struct output (zero-copy ViewBuffer ownership transfer),
    /// as `OutputValue::NumpyStruct`.
    NumpyStruct(Option<ViewBuffer>),
    /// Histogram buckets data [lower_edge, upper_edge, count, normalized] flattened
    HistogramBuckets(Option<Vec<f64>>),
    /// A row of the "long" sink, as `OutputValue::Long`.
    Long(Option<LongRow>),
}

/// A row of the "long" sink: each kept pixel's position, and one 1-D
/// buffer per value field (the buffer's dtype), in the layout's order.
#[derive(Debug, Clone)]
pub(crate) struct LongRow {
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    pub fields: Vec<ViewBuffer>,
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
            RowResult::Long(_) => "Long",
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
            r#"{{"version": 2, "nodes": {{}}, "outputs": {{}}, "{field}": "{value}"}}"#
        ))
        .map(|_| String::new())
        .unwrap_or_else(|e| e.to_string())
    }

    /// The one output of a single `image_bytes` node running `ops`, as planned.
    fn output(ops: &str) -> Result<OutputSpec, String> {
        let graph = UnifiedGraph::from_json(&format!(
            r#"{{"version": 2, "nodes": {{"n0": {{"input": {{"decode": {{"column": 0, "source": {{"format": "image_bytes"}}}}}}, "ops": {ops}}}}},
                "outputs": {{"_output": {{"node": "n0", "sink": {{"format": "numpy"}}}}}}}}"#
        ))
        .map_err(|e| e.to_string())?;
        crate::graph::resolved_output_specs(&graph, &[])
            .map(|mut specs| specs.remove(0).1)
            .map_err(|e| e.to_string())
    }

    /// The output facts are planned from the graph's own ops: a buffer's shape
    /// whenever its rank and every size are known.
    #[test]
    fn output_facts_are_planned_from_the_ops() {
        let full = output(
            r#"[{"op": "cast", "dtype": "u8"},
                {"op": "assert_shape", "dims": [4, 5, 3], "exact": false}]"#,
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
        // Rank 2 with every size known: the shape is published at any rank.
        let rank2 = output(
            r#"[{"op": "assert_shape", "dims": [4, 5, 3], "exact": false},
                {"op": "channel_select", "index": 0, "keepdims": false}]"#,
        )
        .unwrap();
        assert_eq!(
            (rank2.expected_shape, rank2.expected_ndim),
            (Some(vec![4, 5]), Some(2))
        );
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
                r#"{{"version": 2, "nodes": {{}}, "outputs": {{"_output": {{"node": "n0",
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
                r#"{{"version": 2, "nodes": {{"n0": {{"input": {{"decode": {{"column": 0, "source": {{"format": "image_bytes"}}}}}}, "ops": {ops}}},
                     "n1": {{"ops": [], "input": {{"node": "n0"}}}}}},
                    "outputs": {{"a": {{"node": "n0", "sink": {{"format": "native"}}}},
                                 "b": {{"node": "n1", "sink": {{"format": "native"}}}}}}}}"#
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
                r#"{{"version": 2, "nodes": {{}}, "outputs": {{}}, "opt": {opt}}}"#
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
    /// Graph wire-format version, so a format change is refused by name
    /// instead of misparsed (also relevant for any persisted graph JSON).
    /// Required; checked by [`from_json`](Self::from_json) before the rest of
    /// the graph is read.
    #[serde(rename = "version")]
    _version: u32,
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
    /// Cached topological order (computed once during parsing).
    /// Not serialized - computed on load.
    #[serde(skip)]
    cached_order: Vec<String>,
}

/// The graph wire-format version this build reads. Version 2 gave each node
/// one `input` (CR-117); a version-1 graph's node fields are not read.
pub(crate) const GRAPH_VERSION: u32 = 2;

impl UnifiedGraph {
    /// Parse a graph from JSON.
    ///
    /// This also computes and caches the topological order for efficient
    /// repeated execution.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        // The version first, so an older graph is refused by its version
        // rather than by whichever of its fields the parser meets first.
        let version = serde_json::from_str::<serde_json::Value>(json)
            .ok()
            .and_then(|graph| graph.get("version")?.as_u64());
        if let Some(v) = version {
            if v != u64::from(GRAPH_VERSION) {
                polars_bail!(ComputeError:
                    "Pipeline graph format version {} is not the version this polars-cv \
                     build reads ({}); rebuild the pipeline with this polars-cv",
                    v, GRAPH_VERSION
                );
            }
        }
        let mut graph: Self = serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError : "Failed to parse pipeline graph: {}", e))?;
        graph.check_references()?;
        graph.cached_order = graph.compute_topological_order()?;
        Ok(graph)
    }

    /// Refuse a reference to a node the graph does not have: a node's input,
    /// a node an op reads (whatever kind of operand it is), an output's node.
    /// The one structural check, run on every parse, so execution and schema
    /// inference alike can index nodes by the ids the graph names.
    fn check_references(&self) -> PolarsResult<()> {
        for (node_id, node) in &self.nodes {
            if let Some(input) = node.input.node() {
                if !self.nodes.contains_key(input) {
                    polars_bail!(ComputeError:
                        "node '{}' reads node '{}', which is not in the graph",
                        node_id, input
                    );
                }
            }
            for op in &node.ops {
                for operand in op.operands() {
                    if !self.nodes.contains_key(&operand.0) {
                        polars_bail!(ComputeError:
                            "node '{}': {}() reads node '{}', which is not in the graph",
                            node_id, op.name(), operand.0
                        );
                    }
                }
            }
        }
        for (alias, spec) in &self.outputs {
            if !self.nodes.contains_key(&spec.node) {
                polars_bail!(ComputeError:
                    "Output '{}' references unknown node '{}'",
                    alias, spec.node
                );
            }
        }
        Ok(())
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
                .input
                .node()
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
        match &self.nodes.get(node_id)?.input {
            NodeInput::Node(up) => self.column_facts_pending(up),
            NodeInput::Decode { source, .. } => {
                source
                    .resolves_from_column()
                    .then(|| crate::graph::decode::ColumnFacts::Pending {
                        sizes: source.column_may_fix_sizes(),
                    })
            }
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
                for dependency in node.dependencies() {
                    dfs(dependency, nodes, visited, in_stack, order)?;
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
/// Output value from encoding - can be binary, contour struct, scalar, or array.
#[derive(Debug, Clone)]
pub(crate) enum OutputValue {
    Binary(Vec<u8>),
    Contours(Arc<Vec<Outline>>),
    Scalar(f64),
    Vector(Arc<Vec<f64>>),
    /// A row of the "list" sink: its buffer, in any layout and shape. The
    /// elements are copied once, straight into the column's values
    /// (`ViewBuffer::append_to`); the buffer's shape is the row's.
    TypedList(ViewBuffer),
    /// A row of the "array" sink, its shape already checked against the
    /// sink's (see `TypedList`).
    TypedArray(ViewBuffer),
    /// Numpy/Torch struct output (zero-copy ViewBuffer for struct encoding):
    /// the row's buffer, or for a half-precision sink its f16 bits
    /// (`ViewBuffer::to_f16_bits`).
    NumpyStruct(ViewBuffer),
    /// Histogram buckets data [lower_edge, upper_edge, count, normalized] flattened
    HistogramBuckets(Vec<f64>),
    /// A row of the "long" sink.
    Long(LongRow),
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
    /// What this node reads: one declaration, so a node cannot both decode a
    /// column and read a node, or do neither.
    pub input: NodeInput,
    /// Operations to apply.
    #[serde(default)]
    pub ops: Vec<crate::ops::TypedOp>,
}

/// A node's input.
///
/// The version-1 form spread this over three places — a `source` on every
/// node (a dummy `{"format": "blob"}` on a node that read another), a
/// top-level `column_bindings` map saying which nodes were roots, and a
/// positional `upstream` list whose first entry was the input and whose rest
/// restated the nodes the ops read — and needed three checks to reconcile
/// them (CR-117). A node's dependencies are now its input and its ops'
/// [`operands`](crate::graph::step::GraphStep::operands).
#[derive(Debug, Clone, serde::Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum NodeInput {
    /// A root: decodes plugin input `column` through `source`.
    Decode {
        column: usize,
        /// Written as every plan value is ([`crate::plan::wire`]): an unset
        /// setting is left out, so the graph JSON (the compiled-graph cache
        /// key) carries only what was set.
        #[serde(serialize_with = "wire_source")]
        source: Source,
    },
    /// Reads another node's output.
    Node(String),
}

fn wire_source<S: serde::Serializer>(source: &Source, s: S) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(&crate::plan::wire(source), s)
}

impl NodeInput {
    /// The node this one reads, for a continuation.
    pub fn node(&self) -> Option<&str> {
        match self {
            NodeInput::Decode { .. } => None,
            NodeInput::Node(id) => Some(id),
        }
    }
}

impl GraphNode {
    /// Every node this one depends on: its input, then each node its ops read.
    pub fn dependencies(&self) -> impl Iterator<Item = &str> {
        self.input.node().into_iter().chain(
            self.ops
                .iter()
                .flat_map(|op| op.operands().into_iter().map(|n| n.0.as_str())),
        )
    }
}

#[cfg(test)]
mod node_wire_tests {
    //! The node wire form (version 2): a node's input is one declaration,
    //! `{"decode": {"column", "source"}}` or `{"node": id}`, and a node's
    //! dependencies are that input plus the nodes its ops read (CR-117).
    use super::UnifiedGraph;
    use crate::graph::compiled::CompiledGraph;
    use polars::prelude::*;
    use view_buffer::ViewBuffer;

    fn parse_err(json: &str) -> String {
        UnifiedGraph::from_json(json)
            .expect_err("the graph must be refused")
            .to_string()
    }

    const OUTPUT: &str = r#""outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}}"#;

    /// The version-1 form (`column_bindings`, a positional `upstream`, a
    /// `source` on every node) is refused, by its version and by each field
    /// it carried, rather than read with any of them ignored.
    #[test]
    fn the_version_1_node_form_is_refused() {
        let v1 = format!(
            r#"{{"version": 1, "nodes": {{"n0": {{"source": {{"format": "blob"}}}}}},
                {OUTPUT}, "column_bindings": {{"n0": 0}}}}"#
        );
        let err = parse_err(&v1);
        assert!(err.contains("version 1"), "{err}");
        assert!(err.contains("rebuild"), "{err}");

        let node = |fields: &str| {
            format!(
                r#"{{"version": 2, "nodes": {{"n0": {{"input": {{"decode": {{"column": 0,
                    "source": {{"format": "blob"}}}}}}, {fields}}}}}, {OUTPUT}}}"#
            )
        };
        for (field, json) in [
            ("upstream", node(r#""upstream": []"#)),
            ("source", node(r#""source": {"format": "blob"}"#)),
            (
                "column_bindings",
                format!(
                    r#"{{"version": 2, "nodes": {{"n0": {{"input": {{"decode": {{"column": 0,
                        "source": {{"format": "blob"}}}}}}}}}}, {OUTPUT},
                        "column_bindings": {{"n0": 0}}}}"#
                ),
            ),
        ] {
            let err = parse_err(&json);
            assert!(
                err.contains(&format!("unknown field `{field}`")),
                "{field}: {err}"
            );
        }
    }

    /// A node reads a column or a node: both at once, or neither, cannot be
    /// written.
    #[test]
    fn a_node_reads_a_column_or_a_node_not_both_and_not_neither() {
        let graph = |input: &str| {
            format!(r#"{{"version": 2, "nodes": {{"n0": {{"input": {input}}}}}, {OUTPUT}}}"#)
        };
        let both = parse_err(&graph(
            r#"{"decode": {"column": 0, "source": {"format": "blob"}}, "node": "n1"}"#,
        ));
        assert!(both.contains("Failed to parse pipeline graph"), "{both}");
        let neither = parse_err(&graph("{}"));
        assert!(
            neither.contains("Failed to parse pipeline graph"),
            "{neither}"
        );
        let missing = parse_err(&format!(
            r#"{{"version": 2, "nodes": {{"n0": {{"ops": []}}}}, {OUTPUT}}}"#
        ));
        assert!(missing.contains("missing field `input`"), "{missing}");
    }

    /// A node's dependencies are its input and the nodes its ops read, and
    /// nothing else: an operand named only in an op is executed first.
    #[test]
    fn an_operand_named_only_in_its_op_is_a_dependency() {
        let graph = r#"{
            "version": 2,
            "nodes": {
                "sum": {"input": {"node": "a"}, "ops": [{"op": "add", "other": "b"}]},
                "a": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                "b": {"input": {"decode": {"column": 1, "source": {"format": "blob"}}}}
            },
            "outputs": {"_output": {"node": "sum", "sink": {"format": "blob"}}}
        }"#;
        let blob = |v: Vec<f32>| {
            let n = v.len();
            ViewBuffer::from_vec_with_shape(v, vec![n]).to_blob()
        };
        let a = Series::new("a".into(), &[blob(vec![1.0, 2.0])]);
        let b = Series::new("b".into(), &[blob(vec![10.0, 20.0])]);
        let out = CompiledGraph::compile(graph)
            .unwrap()
            .execute(&[a, b])
            .unwrap();
        let sum = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(sum.as_slice::<f32>(), &[11.0, 22.0]);
    }
}
