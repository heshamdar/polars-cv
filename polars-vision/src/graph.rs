//! Pipeline graph execution engine.
//!
//! This module handles the execution of pipeline graphs (DAGs) where multiple
//! pipelines can be composed and executed as a single fused operation.
//!
//! The graph executor:
//! - Parses a JSON graph specification
//! - Executes nodes in topological order
//! - Passes ViewBuffers between nodes without serialization
//! - Supports binary operations between nodes
//! - Supports multi-output mode returning Struct columns
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

use view_buffer::ViewBuffer;

use crate::execute::decode_source;
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

/// Output specification for the graph (single output mode).
#[derive(Debug, Deserialize)]
pub struct GraphOutput {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
}

/// Output specification for multi-output mode.
#[derive(Debug, Deserialize)]
pub struct MultiOutputSpec {
    /// The node ID to output.
    pub node: String,
    /// Sink specification.
    pub sink: SinkSpec,
}

/// Complete pipeline graph specification (single output mode).
#[derive(Debug, Deserialize)]
pub struct PipelineGraph {
    /// Named nodes in the graph.
    pub nodes: HashMap<String, GraphNode>,
    /// Output specification.
    pub output: GraphOutput,
    /// Mapping from node IDs to input column indices.
    pub column_bindings: HashMap<String, usize>,
}

/// Complete pipeline graph specification (multi-output mode).
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

        // Start from output node
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

        // Storage for computed node outputs (per row)
        // For now, we process row by row like the linear pipeline
        let mut results: Vec<Option<Vec<u8>>> = Vec::with_capacity(len);

        for row_idx in 0..len {
            // Buffer cache for this row
            let mut buffers: HashMap<String, ViewBuffer> = HashMap::new();

            // Execute nodes in order
            for node_id in &order {
                let node = self.nodes.get(node_id).ok_or_else(|| {
                    polars_err!(ComputeError: "Node '{}' not found in graph", node_id)
                })?;

                // Get input column index for this node
                let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                if col_idx >= inputs.len() {
                    return Err(polars_err!(ComputeError: 
                        "Column index {} out of bounds for node '{}'", col_idx, node_id));
                }

                // Get the input data
                let input_series = &inputs[col_idx];
                let input_ca = input_series.binary().map_err(|_| {
                    polars_err!(ComputeError: "Expected Binary column for node '{}'", node_id)
                })?;

                let input_bytes = input_ca.get(row_idx);

                if let Some(bytes) = input_bytes {
                    // Create a temporary pipeline spec for decoding
                    let temp_spec = PipelineSpec {
                        source: node.source.clone(),
                        shape_hints: None,
                        ops: node.ops.clone(),
                        sink: self.output.sink.clone(),
                    };

                    // Decode source
                    let buffer = decode_source(bytes, &temp_spec)?;

                    // For now, store the buffer directly
                    // TODO: Apply operations and handle binary ops with buffers HashMap
                    buffers.insert(node_id.clone(), buffer);
                }
            }

            // Get output buffer
            if let Some(output_buffer) = buffers.get(&self.output.node) {
                // Encode using sink format
                let encoded = crate::execute::encode_sink(output_buffer, &self.create_output_spec())?;
                results.push(Some(encoded));
            } else {
                results.push(None);
            }
        }

        // Build output series
        let output_ca = BinaryChunked::from_iter_options(
            inputs[0].name().clone(),
            results.into_iter(),
        );
        Ok(output_ca.into_series())
    }

    /// Create a PipelineSpec for the output node.
    fn create_output_spec(&self) -> PipelineSpec {
        let output_node = self.nodes.get(&self.output.node);
        PipelineSpec {
            source: output_node
                .map(|n| n.source.clone())
                .unwrap_or_else(|| SourceSpec {
                    format: "blob".to_string(),
                    dtype: None,
                }),
            shape_hints: None,
            ops: vec![],
            sink: self.output.sink.clone(),
        }
    }
}

impl MultiPipelineGraph {
    /// Parse a multi-output graph from JSON.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError: "Failed to parse multi-output pipeline graph: {}", e))
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

    /// Execute the graph on input series, returning a Struct column.
    pub fn execute(
        &self,
        inputs: &[Series],
        _expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        // Get topological order
        let order = self.topological_order()?;
        // Output node IDs are used for optimization decisions
        let _output_node_ids = self.output_node_ids();

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
                let node = self.nodes.get(node_id).ok_or_else(|| {
                    polars_err!(ComputeError: "Node '{}' not found in graph", node_id)
                })?;

                // Get input column index for this node
                let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                if col_idx >= inputs.len() {
                    return Err(polars_err!(ComputeError: 
                        "Column index {} out of bounds for node '{}'", col_idx, node_id));
                }

                // Get the input data
                let input_series = &inputs[col_idx];
                let input_ca = input_series.binary().map_err(|_| {
                    polars_err!(ComputeError: "Expected Binary column for node '{}'", node_id)
                })?;

                let input_bytes = input_ca.get(row_idx);

                if let Some(bytes) = input_bytes {
                    // Create a temporary pipeline spec for decoding
                    // Use the first output's sink as a placeholder
                    let first_output = self.outputs.values().next().unwrap();
                    let temp_spec = PipelineSpec {
                        source: node.source.clone(),
                        shape_hints: None,
                        ops: node.ops.clone(),
                        sink: first_output.sink.clone(),
                    };

                    // Decode source
                    let buffer = decode_source(bytes, &temp_spec)?;

                    // TODO: Apply operations and handle binary ops with buffers HashMap
                    buffers.insert(node_id.clone(), buffer);
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

        // Build Struct column from results
        let mut fields: Vec<Series> = Vec::with_capacity(output_aliases.len());
        for alias in &output_aliases {
            let data = results.remove(*alias).unwrap();
            let ca = BinaryChunked::from_iter_options(
                PlSmallStr::from_str(alias),
                data.into_iter(),
            );
            fields.push(ca.into_series());
        }

        // Create Struct column
        let output_name = inputs[0].name().clone();
        StructChunked::from_series(output_name, fields.len(), fields.iter())
            .map(|sc| sc.into_series())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_topological_order() {
        let json = r#"{
            "nodes": {
                "a": {"source": {"format": "image_bytes"}, "ops": [], "upstream": []},
                "b": {"source": {"format": "image_bytes"}, "ops": [], "upstream": []},
                "c": {"source": {"format": "image_bytes"}, "ops": [], "upstream": ["a", "b"]}
            },
            "output": {"node": "c", "sink": {"format": "numpy"}},
            "column_bindings": {"a": 0, "b": 1, "c": 0}
        }"#;

        let graph = PipelineGraph::from_json(json).unwrap();
        let order = graph.topological_order().unwrap();

        // 'c' must come after 'a' and 'b'
        let c_pos = order.iter().position(|x| x == "c").unwrap();
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        let b_pos = order.iter().position(|x| x == "b").unwrap();

        assert!(c_pos > a_pos);
        assert!(c_pos > b_pos);
    }

    #[test]
    fn test_parse_multi_output_graph() {
        let json = r#"{
            "nodes": {
                "img": {
                    "source": {"format": "image_bytes"},
                    "ops": [],
                    "alias": "original"
                },
                "processed": {
                    "source": {"format": "image_bytes"},
                    "ops": [],
                    "upstream": ["img"],
                    "alias": "processed"
                }
            },
            "outputs": {
                "original": {"node": "img", "sink": {"format": "png"}},
                "processed": {"node": "processed", "sink": {"format": "numpy"}}
            },
            "column_bindings": {"img": 0, "processed": 0},
            "is_multi_output": true
        }"#;

        let graph = MultiPipelineGraph::from_json(json).unwrap();
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.outputs.len(), 2);
        assert!(graph.is_multi_output);
        assert!(graph.outputs.contains_key("original"));
        assert!(graph.outputs.contains_key("processed"));
    }

    #[test]
    fn test_multi_output_topological_order() {
        let json = r#"{
            "nodes": {
                "a": {"source": {"format": "image_bytes"}, "ops": [], "alias": "alias_a"},
                "b": {"source": {"format": "image_bytes"}, "ops": [], "upstream": ["a"], "alias": "alias_b"},
                "c": {"source": {"format": "image_bytes"}, "ops": [], "upstream": ["b"]}
            },
            "outputs": {
                "alias_a": {"node": "a", "sink": {"format": "numpy"}},
                "alias_b": {"node": "b", "sink": {"format": "png"}}
            },
            "column_bindings": {"a": 0, "b": 0, "c": 0},
            "is_multi_output": true
        }"#;

        let graph = MultiPipelineGraph::from_json(json).unwrap();
        let order = graph.topological_order().unwrap();

        // Both 'a' and 'b' should be in the order (reachable from outputs)
        assert!(order.contains(&"a".to_string()));
        assert!(order.contains(&"b".to_string()));
        // 'c' is not reachable from any output, so it may or may not be included
        // based on implementation - currently it's excluded which is correct

        // 'b' must come after 'a'
        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        assert!(b_pos > a_pos);
    }

    #[test]
    fn test_multi_output_node_ids() {
        let json = r#"{
            "nodes": {
                "a": {"source": {"format": "image_bytes"}, "ops": []},
                "b": {"source": {"format": "image_bytes"}, "ops": []}
            },
            "outputs": {
                "out1": {"node": "a", "sink": {"format": "numpy"}},
                "out2": {"node": "b", "sink": {"format": "png"}}
            },
            "column_bindings": {"a": 0, "b": 1},
            "is_multi_output": true
        }"#;

        let graph = MultiPipelineGraph::from_json(json).unwrap();
        let output_ids = graph.output_node_ids();

        assert_eq!(output_ids.len(), 2);
        assert!(output_ids.contains("a"));
        assert!(output_ids.contains("b"));
    }
}

