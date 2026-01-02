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

use view_buffer::{BinaryOp, ViewBuffer, ViewDto, ViewExpr};

use crate::execute::{decode_contour_source, decode_contour_source_with_dims, decode_source, resolve_op};
use crate::pipeline::{PipelineSpec, SinkSpec, SourceSpec};

/// Apply a mask to a buffer.
///
/// The mask should be a single-channel buffer where:
/// - 255 values keep the original pixel (fully visible)
/// - 0 values zero out the pixel (fully hidden)
/// - Intermediate values provide weighted blending
///
/// If `invert` is true, the behavior is reversed:
/// - 0 values keep the original pixel
/// - 255 values zero out the pixel
///
/// Uses normalized blending: pixel * (mask / 255)
fn apply_mask(buffer: &ViewBuffer, mask: &ViewBuffer, invert: bool) -> ViewBuffer {
    // Get shapes
    let buf_shape = buffer.shape();
    let mask_shape = mask.shape();

    // Handle broadcasting: mask might be 2D (H, W) while buffer is 3D (H, W, C)
    // We need to broadcast the mask to match the buffer's channels
    let effective_mask = if mask_shape.len() == 2 && buf_shape.len() == 3 {
        // Need to expand mask from (H, W) to (H, W, C)
        let h = mask_shape[0];
        let w = mask_shape[1];
        let c = buf_shape[2];

        let mask_contig = mask.to_contiguous();
        let mask_data = mask_contig.as_slice::<u8>();

        // Create expanded mask with inversion applied if needed
        let mut expanded: Vec<u8> = Vec::with_capacity(h * w * c);
        for y in 0..h {
            for x in 0..w {
                let raw_val = mask_data[y * w + x];
                let mask_val = if invert { 255 - raw_val } else { raw_val };
                // Replicate across channels
                for _ in 0..c {
                    expanded.push(mask_val);
                }
            }
        }

        ViewBuffer::from_vec_with_shape(expanded, vec![h, w, c])
    } else {
        // Same dimensionality - just use as-is, possibly inverting
        if invert {
            let mask_contig = mask.to_contiguous();
            let mask_data = mask_contig.as_slice::<u8>();
            let inverted: Vec<u8> = mask_data.iter().map(|&v| 255 - v).collect();
            ViewBuffer::from_vec_with_shape(inverted, mask_shape.to_vec())
        } else {
            mask.clone()
        }
    };

    // Apply the mask using normalized blend: pixel * (mask / 255)
    // BinaryOp::Blend computes: (a/255) * (b/255) * 255 = a * b / 255
    // This gives us the desired: pixel * (mask / 255)
    BinaryOp::Blend.execute(buffer, &effective_mask)
}

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
    /// Note: Used for deserialization; alias becomes the key in outputs map.
    #[serde(default)]
    #[allow(dead_code)]
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
    /// Cached topological order (computed once during parsing).
    /// Not serialized - computed on load.
    #[serde(skip)]
    cached_order: Vec<String>,
}

impl UnifiedGraph {
    /// Parse a graph from JSON.
    ///
    /// This also computes and caches the topological order for efficient
    /// repeated execution.
    pub fn from_json(json: &str) -> PolarsResult<Self> {
        let mut graph: Self = serde_json::from_str(json)
            .map_err(|e| polars_err!(ComputeError: "Failed to parse pipeline graph: {}", e))?;
        
        // Pre-compute and cache the topological order
        graph.cached_order = graph.compute_topological_order()?;
        
        Ok(graph)
    }

    /// Check if this is a single-output graph (returns Binary instead of Struct).
    pub fn is_single_output(&self) -> bool {
        self.outputs.len() == 1 && self.outputs.contains_key("_output")
    }

    /// Get all output node IDs.
    #[allow(dead_code)]
    pub fn output_node_ids(&self) -> HashSet<String> {
        self.outputs.values().map(|s| s.node.clone()).collect()
    }

    /// Get cached topological order.
    /// The order is computed once during parsing and reused for all executions.
    fn topological_order(&self) -> &[String] {
        &self.cached_order
    }

    /// Compute nodes in topological order (dependencies first).
    /// Includes all nodes reachable from any output.
    fn compute_topological_order(&self) -> PolarsResult<Vec<String>> {
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
    ///
    /// # Optimizations
    ///
    /// 1. **Per-node precompilation**: Nodes where all op params are literals
    ///    have their ViewDtos resolved once before the row loop and reused.
    /// 2. **Batch-level panic catching**: A single catch_unwind wraps the
    ///    entire batch for reduced overhead vs per-row catching.
    /// 3. **Cached topological order**: Computed once during from_json().
    pub fn execute(
        &self,
        inputs: &[Series],
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        // Get cached topological order
        let order = self.topological_order();

        // Get length from first input
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError: "No input columns provided"));
        };

        // Get output aliases in deterministic order
        let mut output_aliases: Vec<&String> = self.outputs.keys().collect();
        output_aliases.sort();

        // ============================================================
        // OPTIMIZATION: Per-node precompilation
        // ============================================================
        // For nodes where all op params are literals, precompile ViewDtos
        // once and reuse for all rows. This avoids repeated parameter
        // resolution in the hot loop.
        let precompiled: HashMap<String, Vec<ViewDto>> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.ops.iter().all(|op| op.is_all_literal()))
            .filter_map(|(node_id, node)| {
                // Resolve ops with row_idx=0 and empty expr_columns (all literal anyway)
                let ops: Result<Vec<ViewDto>, _> = node
                    .ops
                    .iter()
                    .map(|op| resolve_op(op, 0, &HashMap::new()))
                    .collect();
                ops.ok().map(|v| (node_id.clone(), v))
            })
            .collect();

        // Prepare result vectors for each output
        let mut results: HashMap<String, Vec<Option<Vec<u8>>>> = HashMap::new();
        for alias in &output_aliases {
            results.insert((*alias).clone(), Vec::with_capacity(len));
        }

        // ============================================================
        // OPTIMIZATION: Batch-level panic catching
        // ============================================================
        // Wrap the entire row loop in a single catch_unwind to reduce
        // the overhead of setting up unwinding machinery per-row.
        let batch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for row_idx in 0..len {
                // Buffer cache for this row
                let mut buffers: HashMap<String, ViewBuffer> = HashMap::new();

                // Execute nodes in order
                for node_id in order {
                    let node = match self.nodes.get(node_id) {
                        Some(n) => n,
                        None => continue, // Skip missing nodes (shouldn't happen)
                    };

                    // Determine input source for this node
                    // A node is a "root" if it has a column binding (reads from DataFrame column)
                    // Nodes can have both column bindings AND upstream (e.g., contour with shape inference)
                    let has_column_binding = self.column_bindings.contains_key(node_id);
                    let buffer = if has_column_binding {
                        // Root node: get input from column binding
                        let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);

                        if col_idx >= inputs.len() {
                            // Return error as None to indicate failure
                            return Err(format!(
                                "Column index {col_idx} out of bounds for node '{node_id}'"
                            ));
                        }

                        let input_series = &inputs[col_idx];

                        // Check if this is a contour source (Struct input) vs binary source
                        let source_format = node.source.format.as_str();
                        if source_format == "contour" {
                            // Contour source: parse struct and rasterize
                            match input_series.get(row_idx) {
                                Ok(value) if !value.is_null() => {
                                    // Check if we have shape_pipeline for dimension inference
                                    if let Some(ref shape_pipeline) = node.source.shape_pipeline {
                                        // Extract node_id from shape_pipeline JSON
                                        let shape_node_id = shape_pipeline
                                            .get("node_id")
                                            .and_then(|v| v.as_str())
                                            .ok_or_else(|| {
                                                "shape_pipeline missing 'node_id'".to_string()
                                            })?;

                                        // Look up the referenced buffer
                                        let shape_buffer = buffers.get(shape_node_id).ok_or_else(|| {
                                            format!(
                                                "Shape reference '{shape_node_id}' not found. Ensure the shape source is defined before this contour pipeline."
                                            )
                                        })?;

                                        // Get dimensions from buffer shape (HWC layout: [height, width, channels])
                                        let shape = shape_buffer.shape();
                                        if shape.len() < 2 {
                                            return Err(format!(
                                                "Shape buffer has invalid dimensions: expected at least 2D, got {}D",
                                                shape.len()
                                            ));
                                        }
                                        let height = shape[0] as u32;
                                        let width = shape[1] as u32;

                                        // Get fill and background values
                                        let fill_value = node.source.fill_value;
                                        let background = node.source.background;

                                        match decode_contour_source_with_dims(
                                            &value, width, height, fill_value, background,
                                        ) {
                                            Ok(buf) => Some(buf),
                                            Err(e) => return Err(format!("Contour decode error: {e}")),
                                        }
                                    } else {
                                        // Use explicit width/height parameters
                                        let first_output = self.outputs.values().next().unwrap();
                                        let temp_spec = PipelineSpec {
                                            source: node.source.clone(),
                                            shape_hints: None,
                                            ops: vec![],
                                            sink: first_output.sink.clone(),
                                        };
                                        match decode_contour_source(
                                            &value,
                                            row_idx,
                                            &temp_spec,
                                            expr_columns,
                                        ) {
                                            Ok(buf) => Some(buf),
                                            Err(e) => return Err(format!("Contour decode error: {e}")),
                                        }
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            // Binary source: decode from bytes
                            let input_ca = match input_series.binary() {
                                Ok(ca) => ca,
                                Err(_) => {
                                    return Err(format!(
                                        "Expected Binary column for node '{node_id}'"
                                    ))
                                }
                            };

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
                                    // Copy the bytes to avoid any lifetime issues
                                    let bytes_owned = bytes.to_vec();
                                    match decode_source(&bytes_owned, &temp_spec) {
                                        Ok(buf) => Some(buf),
                                        Err(e) => return Err(format!("Decode error: {e}")),
                                    }
                                }
                                None => None,
                            }
                        }
                    } else {
                        // Non-root node: get input from upstream node's buffer
                        let upstream_id = &node.upstream[0];
                        buffers.get(upstream_id).cloned()
                    };

                    if let Some(input_buffer) = buffer {
                        // Get ViewDtos - use precompiled if available, otherwise resolve per-row
                        let view_dtos: Vec<ViewDto> = if let Some(cached) = precompiled.get(node_id)
                        {
                            // Fast path: clone precompiled ops
                            cached.clone()
                        } else {
                            // Slow path: resolve per-row for expression parameters
                            let mut dtos = Vec::with_capacity(node.ops.len());
                            for op_spec in &node.ops {
                                match resolve_op(op_spec, row_idx, expr_columns) {
                                    Ok(dto) => dtos.push(dto),
                                    Err(e) => return Err(format!("Op resolution error: {e}")),
                                }
                            }
                            dtos
                        };

                        // Build expression and execute, handling binary ops specially
                        let mut current_buffer = input_buffer;

                        for view_dto in view_dtos {
                            current_buffer = match view_dto {
                                ViewDto::Binary { op, other_node_id } => {
                                    // Binary operation: fetch the other buffer and apply
                                    match buffers.get(&other_node_id) {
                                        Some(other_buffer) => op.execute(&current_buffer, other_buffer),
                                        None => {
                                            return Err(format!(
                                                "Binary op references unknown node '{other_node_id}'"
                                            ))
                                        }
                                    }
                                }
                                ViewDto::ApplyMask {
                                    mask_node_id,
                                    invert,
                                } => {
                                    // Mask operation: fetch the mask buffer and apply
                                    match buffers.get(&mask_node_id) {
                                        Some(mask_buffer) => {
                                            apply_mask(&current_buffer, mask_buffer, invert)
                                        }
                                        None => {
                                            return Err(format!(
                                                "ApplyMask references unknown node '{mask_node_id}'"
                                            ))
                                        }
                                    }
                                }
                                _ => {
                                    // Regular operation: use ViewExpr
                                    let expr = ViewExpr::new_source(current_buffer);
                                    expr.apply_op(view_dto).plan().execute()
                                }
                            };
                        }

                        buffers.insert(node_id.clone(), current_buffer);
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
                        match crate::execute::encode_sink(buffer, &encode_spec) {
                            Ok(encoded) => {
                                results.get_mut(alias).unwrap().push(Some(encoded));
                            }
                            Err(e) => return Err(format!("Encode error: {e}")),
                        }
                    } else {
                        results.get_mut(alias).unwrap().push(None);
                    }
                }
            }

            Ok(results)
        }));

        // Handle batch result
        let results = match batch_result {
            Ok(Ok(r)) => r,
            Ok(Err(msg)) => {
                return Err(polars_err!(ComputeError: "Pipeline execution failed: {}", msg));
            }
            Err(panic_payload) => {
                // Extract panic message
                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic during batch execution".to_string()
                };
                return Err(polars_err!(ComputeError: "Pipeline batch failed: {}", panic_msg));
            }
        };

        // Build output based on single vs multi output
        if self.is_single_output() {
            // Single output: return Binary column directly
            let data = results.get("_output").unwrap().clone();
            let output_ca =
                BinaryChunked::from_iter_options(inputs[0].name().clone(), data.into_iter());
            Ok(output_ca.into_series())
        } else {
            // Multi output: return Struct column
            let mut fields: Vec<Series> = Vec::with_capacity(output_aliases.len());
            for alias in &output_aliases {
                let data = results.get(*alias).unwrap().clone();
                let ca =
                    BinaryChunked::from_iter_options(PlSmallStr::from_str(alias), data.into_iter());
                fields.push(ca.into_series());
            }

            let output_name = inputs[0].name().clone();
            StructChunked::from_series(output_name, len, fields.iter()).map(|sc| sc.into_series())
        }
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
        // The order is now cached during from_json, access via private method
        let order = graph.topological_order();

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

}
