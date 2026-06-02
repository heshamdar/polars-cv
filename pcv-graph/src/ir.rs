//! Typed IR for a polars-cv pipeline.
//!
//! The [`Graph`] is the single representation used by both the wire format
//! (bincode/JSON) and the executor — there is no separate `PipelineSpec` like
//! in the v1 graph subsystem. A "single op" pipeline is a graph with one
//! node.
//!
//! Plan-time inference fills in [`OutputBinding::planned`] before execution
//! so the Polars `output_type_func_with_kwargs` callback can answer with the
//! concrete dtype/shape.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::params::{ParamMap, ParamValue};

/// Stable identifier for a node within a graph.
///
/// The bridge layer assigns these monotonically as it builds the IR; they're
/// kept as opaque strings (not integers) so JSON wire payloads are readable
/// and so concatenating sub-graphs stays simple.
pub type NodeId = String;

/// Top-level graph payload sent across the bridge boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Nodes in topological order (`IndexMap` preserves insertion order so
    /// debug dumps are stable).
    pub nodes: IndexMap<NodeId, Node>,
    /// Bindings from named output aliases to (node, sink) pairs.
    pub outputs: Vec<OutputBinding>,
    /// Columns expected as the root inputs (typed, named).
    pub root_columns: Vec<RootBinding>,
    /// Names of expression-bound columns referenced from any `ParamValue::Expr`.
    pub expr_columns: Vec<String>,
    /// Wire-format version stamp (matches `pcv_graph::WIRE_VERSION`).
    pub wire_version: u32,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: IndexMap::new(),
            outputs: Vec::new(),
            root_columns: Vec::new(),
            expr_columns: Vec::new(),
            wire_version: crate::WIRE_VERSION,
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// A graph node — either a source or an op.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Node {
    Source {
        spec: SourceSpec,
    },
    Op {
        op_id: String,
        params: SerializedParams,
        inputs: Inputs,
    },
}

/// Per-node input wiring.
///
/// Single-input ops are the common case; named multi-input ops (binary,
/// `apply_mask`, `channel_merge`) use `Named`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Inputs {
    Single { node: NodeId },
    Named { ports: BTreeMap<String, NodeId> },
}

impl Inputs {
    /// All upstream node ids referenced, in stable order.
    pub fn upstreams(&self) -> Vec<&NodeId> {
        match self {
            Inputs::Single { node } => vec![node],
            Inputs::Named { ports } => ports.values().collect(),
        }
    }
}

/// Source specification — `name` is the registry key, `params` carries the
/// configuration payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpec {
    pub name: String,
    pub params: SerializedParams,
    /// Column the source reads its row data from (if any). Sources like
    /// `image_bytes` set this; constant generators leave it `None`.
    #[serde(default)]
    pub input_column: Option<String>,
}

/// Sink specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkSpec {
    pub name: String,
    pub params: SerializedParams,
}

/// Param map as it travels on the wire — preserves the literal-vs-expr
/// distinction. The bridge resolves `Expr` entries to literals before
/// invoking op factories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SerializedParams(pub IndexMap<String, ParamValue>);

impl SerializedParams {
    pub fn new() -> Self {
        Self(IndexMap::new())
    }

    /// Fast path: extract a `ParamMap` of literals only. Returns `None` if
    /// any entry is expression-bound (the bridge then falls back to per-row
    /// resolution).
    pub fn try_into_literal_map(&self) -> Option<ParamMap> {
        let mut out = ParamMap::with_capacity(self.0.len());
        for (k, v) in &self.0 {
            match v {
                ParamValue::Literal { value } => {
                    out.insert(k.clone(), value.clone());
                }
                ParamValue::Expr { .. } => return None,
            }
        }
        Some(out)
    }
}

/// Output alias → (node, sink, planned schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputBinding {
    /// Alias for the output column (for single-output pipelines, the
    /// convention is `"_output"`).
    pub alias: String,
    /// Node whose output feeds this sink.
    pub node: NodeId,
    /// Sink configuration.
    pub sink: SinkSpec,
    /// Inferred schema, filled in by [`crate::plan`] before execution.
    #[serde(default)]
    pub planned: Option<PlannedSchema>,
}

/// Plan-time inferred schema for an output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlannedSchema {
    pub dtype: Option<String>,
    pub ndim: Option<u8>,
    pub shape: Option<Vec<i64>>,
}

/// Root column binding — an input column the pipeline reads from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootBinding {
    pub name: String,
    /// The node id that consumes this column (typically a source node).
    pub node: NodeId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_roundtrips_through_json() {
        let mut g = Graph::new();
        g.nodes.insert(
            "n0".into(),
            Node::Source {
                spec: SourceSpec {
                    name: "image_bytes".into(),
                    params: SerializedParams::new(),
                    input_column: Some("img".into()),
                },
            },
        );
        g.nodes.insert(
            "n1".into(),
            Node::Op {
                op_id: "identity".into(),
                params: SerializedParams::new(),
                inputs: Inputs::Single { node: "n0".into() },
            },
        );
        g.outputs.push(OutputBinding {
            alias: "_output".into(),
            node: "n1".into(),
            sink: SinkSpec {
                name: "numpy".into(),
                params: SerializedParams::new(),
            },
            planned: None,
        });

        let json = serde_json::to_string(&g).unwrap();
        let g2: Graph = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.nodes.len(), 2);
        assert_eq!(g2.outputs.len(), 1);
        assert_eq!(g2.wire_version, crate::WIRE_VERSION);
    }
}
