//! Unified pipeline graph execution engine.
//!
//! This module handles the execution of pipeline graphs (DAGs) where multiple
//! pipelines can be composed and executed as a single fused operation.
//!
//! The graph executor:
//! - Parses a JSON graph specification
//! - Executes nodes in topological order
//! - Passes ViewBuffers between nodes without serialization
//! - Supports binary operations between nodes
//! - Returns Binary for single output ("_output") or Struct for multiple outputs
//!
//! # Optimization Boundaries
//!
//! Each node in the graph represents an optimization boundary. Operations
//! within a node may be fused by view-buffer's optimizer (e.g., scalar ops),
//! but operations across different nodes are never fused. This ensures:
//!
//! - Output nodes produce exactly the buffer state at their alias point
//! - Shared subexpressions are computed once and reused
//! - No mutation safety issues since each node produces a new buffer

use polars::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

use view_buffer::{ViewBuffer, ViewExpr};

use crate::execute::{decode_source, resolve_op};
use crate::pipeline::{PipelineSpec, SinkSpec, SourceSpec};

/// A node in the pipeline graph.
#[derive(Debug, Deserialize)]
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
    #[serde(default)]
    pub alias: Option<String>,
}

/// Output specification for a single output in the graph.
#[derive(Debug, Deserialize)]
pub struct OutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
}

/// Unified pipeline graph specification.
///
/// This struct handles all cases:
/// - Single output: `outputs` contains only "_output" key, returns Binary
/// - Multi output: `outputs` contains multiple keys, returns Struct
#[derive(Debug, Deserialize)]
pub struct UnifiedGraph {
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// Output specifications (alias -> spec).
    /// Single output uses "_output" as key.
    pub outputs: HashMap<String, OutputSpec>,
    /// Mapping from node IDs to input column indices.
    /// Only root nodes (no upstream) have bindings.
    #[serde(default)]
    pub column_bindings: HashMap<String, usize>,
}

impl UnifiedGraph {
    /// Parse a graph from JSON.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError: "Failed to parse pipeline graph: {}", e))
    }

    /// Check if this is a single-output graph (returns Binary instead of Struct).
    pub fn is_single_output(&self) -> bool {
        self.outputs.len() == 1 && self.outputs.contains_key("_output")
    }

    /// Get all output node IDs.
    pub fn output_node_ids(&self) -> HashSet<String> {
        self.outputs.values().map(|s| s.node.clone()).collect()
    }

    /// Get nodes in topological order (dependencies first).
    /// Includes all nodes reachable from any output.
    pub fn topological_order(&self) -> PolarsResult<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();

        fn dfs(
            node_id: &str,
            nodes: &HashMap<String, GraphNode>,
            visited: &mut HashSet<String>,
            order: &mut Vec<String>,
        ) -> PolarsResult<()> {
            if visited.contains(node_id) {
                return Ok(());
            }

            visited.insert(node_id.to_string());

            if let Some(node) = nodes.get(node_id) {
                for upstream_id in &node.upstream {
                    dfs(upstream_id, nodes, visited, order)?;
                }
            }

            order.push(node_id.to_string());
            Ok(())
        }

        // Start from all output nodes
        for spec in self.outputs.values() {
            dfs(&spec.node, &self.nodes, &mut visited, &mut order)?;
        }

        Ok(order)
    }

    /// Execute the graph on input series.
    ///
    /// Returns:
    /// - Binary column if single output ("_output" only)
    /// - Struct column with named Binary fields if multiple outputs
    pub fn execute(
        &self,
        inputs: &[Series],
        _expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        // Get topological order
        let order = self.topological_order()?;

        // Get length from first input
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError: "No input columns provided"));
        };

        // Get output aliases in deterministic order
        let mut output_aliases: Vec<&String> = self.outputs.keys().collect();
        output_aliases.sort();

        // Prepare result vectors for each output
        let mut results: HashMap<String, Vec<Option<Vec<u8>>>> = HashMap::new();
        for alias in &output_aliases {
            results.insert((*alias).clone(), Vec::with_capacity(len));
        }

        for row_idx in 0..len {
            // Buffer cache for this row
            let mut buffers: HashMap<String, ViewBuffer> = HashMap::new();

            // Execute nodes in order
            for node_id in &order {
                let node = self.nodes.get(node_id).ok_or_else(
                    || polars_err!(ComputeError: "Node '{}' not found in graph", node_id),
                )?;

                // Determine input source for this node
                let buffer = if node.upstream.is_empty() {
                    // Root node: get input from column binding
                    let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                    if col_idx >= inputs.len() {
                        return Err(
                            polars_err!(ComputeError: "Column index {} out of bounds for node '{}'", col_idx, node_id),
                        );
                    }

                    let input_series = &inputs[col_idx];
                    let input_ca = input_series.binary().map_err(
                        |_| polars_err!(ComputeError: "Expected Binary column for node '{}'", node_id),
                    )?;

                    match input_ca.get(row_idx) {
                        Some(bytes) => {
                            // Create temp spec for decoding
                            let first_output = self.outputs.values().next().unwrap();
                            let temp_spec = PipelineSpec {
                                source: node.source.clone(),
                                shape_hints: None,
                                ops: vec![],
                                sink: first_output.sink.clone(),
                            };
                            Some(decode_source(bytes, &temp_spec)?)
                        }
                        None => None,
                    }
                } else {
                    // Non-root node: get input from upstream node's buffer
                    // For now, we use the first upstream node's buffer
                    // The source should be "blob" for these nodes
                    let upstream_id = &node.upstream[0];
                    buffers.get(upstream_id).cloned()
                };

                if let Some(input_buffer) = buffer {
                    // Resolve and apply operations
                    let mut view_dtos = Vec::with_capacity(node.ops.len());
                    for op_spec in &node.ops {
                        let view_dto = resolve_op(op_spec, row_idx, _expr_columns)?;
                        view_dtos.push(view_dto);
                    }

                    // Build expression and execute
                    let result_buffer =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let mut expr = ViewExpr::new_source(input_buffer);
                            for view_dto in view_dtos {
                                expr = expr.apply_op(view_dto);
                            }
                            expr.plan().execute()
                        }))
                        .map_err(|e| {
                            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                                (*s).to_string()
                            } else if let Some(s) = e.downcast_ref::<String>() {
                                s.clone()
                            } else {
                                "Unknown panic".to_string()
                            };
                            polars_err!(ComputeError: "Pipeline execution failed at node '{}': {}", node_id, msg)
                        })?;

                    buffers.insert(node_id.clone(), result_buffer);
                }
            }

            // Encode each output
            for (alias, spec) in &self.outputs {
                if let Some(buffer) = buffers.get(&spec.node) {
                    // Create pipeline spec for encoding
                    let encode_spec = PipelineSpec {
                        source: SourceSpec {
                            format: "blob".to_string(),
                            dtype: None,
                            width: None,
                            height: None,
                            fill_value: 255,
                            background: 0,
                            shape_pipeline: None,
                        },
                        shape_hints: None,
                        ops: vec![],
                        sink: spec.sink.clone(),
                    };
                    let encoded = crate::execute::encode_sink(buffer, &encode_spec)?;
                    results.get_mut(alias).unwrap().push(Some(encoded));
                } else {
                    results.get_mut(alias).unwrap().push(None);
                }
            }
        }

        // Build output based on single vs multi output
        if self.is_single_output() {
            // Single output: return Binary column directly
            let data = results.remove("_output").unwrap();
            let output_ca =
                BinaryChunked::from_iter_options(inputs[0].name().clone(), data.into_iter());
            Ok(output_ca.into_series())
        } else {
            // Multi output: return Struct column
            let mut fields: Vec<Series> = Vec::with_capacity(output_aliases.len());
            for alias in &output_aliases {
                let data = results.remove(*alias).unwrap();
                let ca =
                    BinaryChunked::from_iter_options(PlSmallStr::from_str(alias), data.into_iter());
                fields.push(ca.into_series());
            }

            let output_name = inputs[0].name().clone();
            StructChunked::from_series(output_name, len, fields.iter())
                .map(|sc| sc.into_series())
        }
    }
}

// Keep legacy types for backward compatibility during transition
// These will be removed in cleanup phase

/// Output specification for the graph (single output mode) - DEPRECATED
#[derive(Debug, Deserialize)]
pub struct GraphOutput {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
}

/// Output specification for multi-output mode - DEPRECATED
#[derive(Debug, Deserialize)]
pub struct MultiOutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
}

/// Complete pipeline graph specification (single output mode) - DEPRECATED
#[derive(Debug, Deserialize)]
pub struct PipelineGraph {
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// Output specification.
    pub output: GraphOutput,
    /// Mapping from node IDs to input column indices.
    pub column_bindings: HashMap<String, usize>,
}

/// Complete pipeline graph specification (multi-output mode) - DEPRECATED
#[derive(Debug, Deserialize)]
pub struct MultiPipelineGraph {
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// Multiple output specifications (alias -> spec).
    pub outputs: HashMap<String, MultiOutputSpec>,
    /// Mapping from node IDs to input column indices.
    pub column_bindings: HashMap<String, usize>,
    /// Flag indicating multi-output mode (for serde).
    #[serde(default)]
    pub is_multi_output: bool,
}

impl PipelineGraph {
    /// Parse a graph from JSON.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError: "Failed to parse pipeline graph: {}", e))
    }

    /// Get nodes in topological order (dependencies first).
    pub fn topological_order(&self) -> PolarsResult<Vec<String>> {
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut order: Vec<String> = Vec::new();

        fn dfs(
            node_id: &str,
            graph: &PipelineGraph,
            visited: &mut std::collections::HashSet<String>,
            order: &mut Vec<String>,
        ) -> PolarsResult<()> {
            if visited.contains(node_id) {
                return Ok(());
            }

            visited.insert(node_id.to_string());

            if let Some(node) = graph.nodes.get(node_id) {
                for upstream_id in &node.upstream {
                    dfs(upstream_id, graph, visited, order)?;
                }
            }

            order.push(node_id.to_string());
            Ok(())
        }

        dfs(&self.output.node, self, &mut visited, &mut order)?;
        Ok(order)
    }

    /// Execute the graph on input series.
    pub fn execute(
        &self,
        inputs: &[Series],
        _expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        // Get topological order
        let order = self.topological_order()?;

        // Get length from first input
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError: "No input columns provided"));
        };

        let mut results: Vec<Option<Vec<u8>>> = Vec::with_capacity(len);

        for row_idx in 0..len {
            let mut buffers: HashMap<String, ViewBuffer> = HashMap::new();

            for node_id in &order {
                let node = self.nodes.get(node_id).ok_or_else(
                    || polars_err!(ComputeError: "Node '{}' not found in graph", node_id),
                )?;

                let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                if col_idx >= inputs.len() {
                    return Err(
                        polars_err!(ComputeError: "Column index {} out of bounds for node '{}'", col_idx, node_id),
                    );
                }

                let input_series = &inputs[col_idx];
                let input_ca = input_series.binary().map_err(
                    |_| polars_err!(ComputeError: "Expected Binary column for node '{}'", node_id),
                )?;

                let input_bytes = input_ca.get(row_idx);

                if let Some(bytes) = input_bytes {
                    let temp_spec = PipelineSpec {
                        source: node.source.clone(),
                        shape_hints: None,
                        ops: node.ops.clone(),
                        sink: self.output.sink.clone(),
                    };

                    let buffer = decode_source(bytes, &temp_spec)?;

                    let mut view_dtos = Vec::with_capacity(node.ops.len());
                    for op_spec in &node.ops {
                        let view_dto = resolve_op(op_spec, row_idx, _expr_columns)?;
                        view_dtos.push(view_dto);
                    }

                    let result_buffer =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let mut expr = ViewExpr::new_source(buffer);
                            for view_dto in view_dtos {
                                expr = expr.apply_op(view_dto);
                            }
                            expr.plan().execute()
                        }))
                        .map_err(|e| {
                            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                                (*s).to_string()
                            } else if let Some(s) = e.downcast_ref::<String>() {
                                s.clone()
                            } else {
                                "Unknown panic".to_string()
                            };
                            polars_err!(ComputeError: "Pipeline execution failed at node '{}': {}", node_id, msg)
                        })?;

                    buffers.insert(node_id.clone(), result_buffer);
                }
            }

            if let Some(output_buffer) = buffers.get(&self.output.node) {
                let encode_spec = PipelineSpec {
                    source: SourceSpec {
                        format: "blob".to_string(),
                        dtype: None,
                        width: None,
                        height: None,
                        fill_value: 255,
                        background: 0,
                        shape_pipeline: None,
                    },
                    shape_hints: None,
                    ops: vec![],
                    sink: self.output.sink.clone(),
                };
                let encoded = crate::execute::encode_sink(output_buffer, &encode_spec)?;
                results.push(Some(encoded));
            } else {
                results.push(None);
            }
        }

        let output_ca =
            BinaryChunked::from_iter_options(inputs[0].name().clone(), results.into_iter());
        Ok(output_ca.into_series())
    }
}

impl MultiPipelineGraph {
    /// Parse a multi-output graph from JSON.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        serde_json::from_str(json).map_err(
            |e| polars_err!(ComputeError: "Failed to parse multi-output pipeline graph: {}", e),
        )
    }

    /// Get all output node IDs.
    pub fn output_node_ids(&self) -> HashSet<String> {
        self.outputs.values().map(|s| s.node.clone()).collect()
    }

    /// Get nodes in topological order (dependencies first).
    pub fn topological_order(&self) -> PolarsResult<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();

        fn dfs(
            node_id: &str,
            nodes: &HashMap<String, GraphNode>,
            visited: &mut HashSet<String>,
            order: &mut Vec<String>,
        ) -> PolarsResult<()> {
            if visited.contains(node_id) {
                return Ok(());
            }

            visited.insert(node_id.to_string());

            if let Some(node) = nodes.get(node_id) {
                for upstream_id in &node.upstream {
                    dfs(upstream_id, nodes, visited, order)?;
                }
            }

            order.push(node_id.to_string());
            Ok(())
        }

        for spec in self.outputs.values() {
            dfs(&spec.node, &self.nodes, &mut visited, &mut order)?;
        }

        Ok(order)
    }

    /// Execute the graph on input series, returning a Struct column.
    pub fn execute(
        &self,
        inputs: &[Series],
        _expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        let order = self.topological_order()?;
        let _output_node_ids = self.output_node_ids();

        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError: "No input columns provided"));
        };

        let mut output_aliases: Vec<&String> = self.outputs.keys().collect();
        output_aliases.sort();

        let mut results: HashMap<String, Vec<Option<Vec<u8>>>> = HashMap::new();
        for alias in &output_aliases {
            results.insert((*alias).clone(), Vec::with_capacity(len));
        }

        for row_idx in 0..len {
            let mut buffers: HashMap<String, ViewBuffer> = HashMap::new();

            for node_id in &order {
                let node = self.nodes.get(node_id).ok_or_else(
                    || polars_err!(ComputeError: "Node '{}' not found in graph", node_id),
                )?;

                let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                if col_idx >= inputs.len() {
                    return Err(
                        polars_err!(ComputeError: "Column index {} out of bounds for node '{}'", col_idx, node_id),
                    );
                }

                let input_series = &inputs[col_idx];
                let input_ca = input_series.binary().map_err(
                    |_| polars_err!(ComputeError: "Expected Binary column for node '{}'", node_id),
                )?;

                let input_bytes = input_ca.get(row_idx);

                if let Some(bytes) = input_bytes {
                    let first_output = self.outputs.values().next().unwrap();
                    let temp_spec = PipelineSpec {
                        source: node.source.clone(),
                        shape_hints: None,
                        ops: node.ops.clone(),
                        sink: first_output.sink.clone(),
                    };

                    let buffer = decode_source(bytes, &temp_spec)?;

                    let mut view_dtos = Vec::with_capacity(node.ops.len());
                    for op_spec in &node.ops {
                        let view_dto = resolve_op(op_spec, row_idx, _expr_columns)?;
                        view_dtos.push(view_dto);
                    }

                    let result_buffer =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let mut expr = ViewExpr::new_source(buffer);
                            for view_dto in view_dtos {
                                expr = expr.apply_op(view_dto);
                            }
                            expr.plan().execute()
                        }))
                        .map_err(|e| {
                            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                                (*s).to_string()
                            } else if let Some(s) = e.downcast_ref::<String>() {
                                s.clone()
                            } else {
                                "Unknown panic".to_string()
                            };
                            polars_err!(ComputeError: "Pipeline execution failed at node '{}': {}", node_id, msg)
                        })?;

                    buffers.insert(node_id.clone(), result_buffer);
                }
            }

            for (alias, spec) in &self.outputs {
                if let Some(buffer) = buffers.get(&spec.node) {
                    let encode_spec = PipelineSpec {
                        source: SourceSpec {
                            format: "blob".to_string(),
                            dtype: None,
                            width: None,
                            height: None,
                            fill_value: 255,
                            background: 0,
                            shape_pipeline: None,
                        },
                        shape_hints: None,
                        ops: vec![],
                        sink: spec.sink.clone(),
                    };
                    let encoded = crate::execute::encode_sink(buffer, &encode_spec)?;
                    results.get_mut(alias).unwrap().push(Some(encoded));
                } else {
                    results.get_mut(alias).unwrap().push(None);
                }
            }
        }

        let mut fields: Vec<Series> = Vec::with_capacity(output_aliases.len());
        for alias in &output_aliases {
            let data = results.remove(*alias).unwrap();
            let ca =
                BinaryChunked::from_iter_options(PlSmallStr::from_str(alias), data.into_iter());
            fields.push(ca.into_series());
        }

        let output_name = inputs[0].name().clone();
        StructChunked::from_series(output_name, len, fields.iter())
            .map(|sc| sc.into_series())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_unified_single_output() {
        let json = r#"{
            "nodes": {
                "_node_0": {
                    "source": {"format": "image_bytes"},
                    "ops": []
                }
            },
            "outputs": {
                "_output": {"node": "_node_0", "sink": {"format": "numpy"}}
            },
            "column_bindings": {"_node_0": 0}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.is_single_output());
        assert!(graph.outputs.contains_key("_output"));
    }

    #[test]
    fn test_parse_unified_multi_output() {
        let json = r#"{
            "nodes": {
                "_node_0": {
                    "source": {"format": "image_bytes"},
                    "ops": [],
                    "alias": "original"
                },
                "_node_1": {
                    "source": {"format": "blob"},
                    "ops": [],
                    "upstream": ["_node_0"],
                    "alias": "processed"
                }
            },
            "outputs": {
                "original": {"node": "_node_0", "sink": {"format": "png"}},
                "processed": {"node": "_node_1", "sink": {"format": "numpy"}}
            },
            "column_bindings": {"_node_0": 0}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        assert_eq!(graph.nodes.len(), 2);
        assert!(!graph.is_single_output());
        assert!(graph.outputs.contains_key("original"));
        assert!(graph.outputs.contains_key("processed"));
    }

    #[test]
    fn test_unified_topological_order() {
        let json = r#"{
            "nodes": {
                "a": {"source": {"format": "image_bytes"}, "ops": [], "alias": "out_a"},
                "b": {"source": {"format": "blob"}, "ops": [], "upstream": ["a"], "alias": "out_b"}
            },
            "outputs": {
                "out_a": {"node": "a", "sink": {"format": "numpy"}},
                "out_b": {"node": "b", "sink": {"format": "png"}}
            },
            "column_bindings": {"a": 0}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        let order = graph.topological_order().unwrap();

        assert!(order.contains(&"a".to_string()));
        assert!(order.contains(&"b".to_string()));

        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        assert!(b_pos > a_pos);
    }

    #[test]
    fn test_output_node_ids() {
        let json = r#"{
            "nodes": {
                "a": {"source": {"format": "image_bytes"}, "ops": []},
                "b": {"source": {"format": "image_bytes"}, "ops": []}
            },
            "outputs": {
                "out1": {"node": "a", "sink": {"format": "numpy"}},
                "out2": {"node": "b", "sink": {"format": "png"}}
            },
            "column_bindings": {"a": 0, "b": 1}
        }"#;

        let graph = UnifiedGraph::from_json(json).unwrap();
        let output_ids = graph.output_node_ids();

        assert_eq!(output_ids.len(), 2);
        assert!(output_ids.contains("a"));
        assert!(output_ids.contains("b"));
    }

    // Legacy tests for backward compatibility
    #[test]
    fn test_parse_simple_graph() {
        let json = r#"{
            "nodes": {
                "img": {
                    "source": {"format": "image_bytes"},
                    "ops": []
                }
            },
            "output": {
                "node": "img",
                "sink": {"format": "numpy"}
            },
            "column_bindings": {"img": 0}
        }"#;

        let graph = PipelineGraph::from_json(json).unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.output.node, "img");
    }
}
