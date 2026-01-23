//! Core types for the unified pipeline graph.
//!
//! This module contains the primary data structures for representing and executing
//! vision pipeline graphs: `UnifiedGraph`, `GraphNode`, `OutputSpec`, etc.

use polars::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use view_buffer::geometry::Contour;
use view_buffer::ops::NodeOutput;
use view_buffer::{ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr};

use crate::execute::{
    decode_contour_source, decode_contour_source_with_dims, decode_source, resolve_op,
};
use crate::pipeline::{PipelineSpec, SinkSpec, SourceSpec};

use super::decode::{
    apply_mask, build_series_from_spec, decode_binary_zero_copy, decode_list_or_array_source,
    get_binary_row_buffer, null_row_result_for_spec, pad_buffer,
};
use super::encode::{default_domain, default_dtype, encode_node_output, execute_geometry_op};

/// Output specification for a single output in the graph.
#[derive(Debug, Deserialize)]
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
            .map_err(|e| polars_err!(ComputeError : "Failed to parse pipeline graph: {}", e))?;
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
    pub(crate) fn topological_order(&self) -> &[String] {
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
        for spec in self.outputs.values() {
            dfs(&spec.node, &self.nodes, &mut visited, &mut order)?;
        }
        Ok(order)
    }
    /// Execute the graph on input series.
    ///
    /// Returns:
    /// - Binary column if single output ("_output" only)
    /// - Struct column with named fields (Binary/Float64/Struct) if multiple outputs
    ///
    /// # Optimizations
    ///
    /// 1. **Per-node precompilation**: Nodes where all op params are literals
    ///    have their ViewDtos resolved once before the row loop and reused.
    /// 2. **Batch-level panic catching**: A single catch_unwind wraps the
    ///    entire batch for reduced overhead vs per-row catching.
    /// 3. **Cached topological order**: Computed once during from_json().
    ///
    /// # Typed Node Support
    ///
    /// The executor now handles typed nodes via `NodeOutput`, supporting:
    /// - Buffer (images/arrays) → Binary encoding
    /// - Contours (geometry) → Struct encoding with "native" format
    /// - Scalar (single values) → Float64 with "native" format
    /// - Vector (multiple values) → List/Struct with "native" format
    pub fn execute(
        &self,
        inputs: &[Series],
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<Series> {
        let order = self.topological_order();
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError : "No input columns provided"));
        };
        let mut output_aliases: Vec<&String> = self.outputs.keys().collect();
        output_aliases.sort();
        let precompiled: HashMap<String, Vec<ViewDto>> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.ops.iter().all(|op| op.is_all_literal()))
            .filter_map(|(node_id, node)| {
                let ops: Result<Vec<ViewDto>, _> = node
                    .ops
                    .iter()
                    .map(|op| resolve_op(op, 0, &HashMap::new()))
                    .collect();
                ops.ok().map(|v| (node_id.clone(), v))
            })
            .collect();
        let mut results: HashMap<String, Vec<RowResult>> = HashMap::new();
        for alias in &output_aliases {
            results.insert((*alias).clone(), Vec::with_capacity(len));
        }
        let batch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for row_idx in 0..len {
                let mut node_outputs: HashMap<String, NodeOutput> = HashMap::new();
                for node_id in order {
                    let node = match self.nodes.get(node_id) {
                        Some(n) => n,
                        None => continue,
                    };
                    let has_column_binding = self.column_bindings.contains_key(node_id);
                    let node_input: Option<NodeOutput> = if has_column_binding {
                        let col_idx = self.column_bindings.get(node_id).copied().unwrap_or(0);
                        if col_idx >= inputs.len() {
                            return Err(format!(
                                "Column index {col_idx} out of bounds for node '{node_id}'"
                            ));
                        }
                        let input_series = &inputs[col_idx];
                        let source_format = node.source.format.as_str();
                        if source_format == "contour" {
                            match input_series.get(row_idx) {
                                Ok(value) if !value.is_null() => {
                                    if let Some(ref shape_pipeline) = node.source.shape_pipeline {
                                        let shape_node_id = shape_pipeline
                                            .get("node_id")
                                            .and_then(|v| v.as_str())
                                            .ok_or_else(|| {
                                                "shape_pipeline missing 'node_id'".to_string()
                                            })?;
                                        let shape_output = node_outputs
                                                .get(shape_node_id)
                                                .ok_or_else(|| {
                                                    format!(
                                                        "Shape reference '{shape_node_id}' not found. Ensure the shape source is defined before this contour pipeline."
                                                    )
                                                })?;
                                        let shape_buffer = shape_output
                                                .as_buffer()
                                                .ok_or_else(|| {
                                                    format!(
                                                        "Shape reference '{shape_node_id}' must be a Buffer, not {:?}",
                                                        shape_output.domain()
                                                    )
                                                })?;
                                        let shape = shape_buffer.shape();
                                        if shape.len() < 2 {
                                            return Err(
                                                    format!(
                                                        "Shape buffer has invalid dimensions: expected at least 2D, got {}D",
                                                        shape.len()
                                                    ),
                                                );
                                        }
                                        let height = shape[0] as u32;
                                        let width = shape[1] as u32;
                                        let fill_value = node.source.fill_value;
                                        let background = node.source.background;
                                        match decode_contour_source_with_dims(
                                            &value, width, height, fill_value, background,
                                        ) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => {
                                                return Err(format!("Contour decode error: {e}"))
                                            }
                                        }
                                    } else {
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
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => {
                                                return Err(format!("Contour decode error: {e}"))
                                            }
                                        }
                                    }
                                }
                                _ => None,
                            }
                        } else if source_format == "file_path" {
                            if input_series.dtype() == &DataType::Null {
                                None
                            } else {
                                let input_ca = match input_series.str() {
                                    Ok(ca) => ca,
                                    Err(_) => {
                                        return Err(
                                                format!(
                                                    "Expected String column for file_path source '{node_id}', got {:?}",
                                                    input_series.dtype()
                                                ),
                                            );
                                    }
                                };
                                match input_ca.get(row_idx) {
                                    Some(path) => {
                                        let bytes = if path.starts_with("s3://")
                                            || path.starts_with("gs://")
                                            || path.starts_with("az://")
                                            || path.starts_with("abfs://")
                                            || path.starts_with("abfss://")
                                            || path.starts_with("http://")
                                            || path.starts_with("https://")
                                        {
                                            match crate::cloud::read_file(path, None) {
                                                Ok(b) => b,
                                                Err(e) => {
                                                    return Err(format!(
                                                        "Failed to read remote file '{path}': {e}"
                                                    ));
                                                }
                                            }
                                        } else {
                                            std::fs::read(path).map_err(|e| {
                                                format!("Failed to read local file '{path}': {e}")
                                            })?
                                        };
                                        let first_output = self.outputs.values().next().unwrap();
                                        let mut source_spec = node.source.clone();
                                        source_spec.format = "image_bytes".to_string();
                                        let temp_spec = PipelineSpec {
                                            source: source_spec,
                                            shape_hints: None,
                                            ops: vec![],
                                            sink: first_output.sink.clone(),
                                        };
                                        match decode_source(&bytes, &temp_spec) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => {
                                                return Err(format!(
                                                    "Decode error for file '{path}': {e}"
                                                ));
                                            }
                                        }
                                    }
                                    None => None,
                                }
                            }
                        } else if node.source.format == "list" || node.source.format == "array" {
                            if input_series.dtype() == &DataType::Null {
                                None
                            } else {
                                let dtype_opt = node.source.dtype.as_deref();
                                let require_contiguous = node.source.require_contiguous;
                                match decode_list_or_array_source(
                                    input_series,
                                    row_idx,
                                    dtype_opt,
                                    require_contiguous,
                                ) {
                                    Ok(Some(buf)) => Some(NodeOutput::from_buffer(buf)),
                                    Ok(None) => None,
                                    Err(e) => {
                                        return Err(format!("List/Array decode error: {e}"));
                                    }
                                }
                            }
                        } else if input_series.dtype() == &DataType::Null {
                            None
                        } else {
                            let input_ca = match input_series.binary() {
                                Ok(ca) => ca,
                                Err(_) => {
                                    return Err(format!(
                                        "Expected Binary column for node '{node_id}', got {:?}",
                                        input_series.dtype()
                                    ));
                                }
                            };
                            let source_format = node.source.format.as_str();
                            if source_format == "blob" || source_format == "raw" {
                                if let Some((buffer, offset, len)) =
                                    get_binary_row_buffer(input_ca, row_idx)
                                {
                                    match decode_binary_zero_copy(
                                        buffer,
                                        offset,
                                        len,
                                        source_format,
                                        node.source.dtype.as_deref(),
                                    ) {
                                        Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                        Err(e) => {
                                            return Err(format!("Zero-copy decode error: {e}"))
                                        }
                                    }
                                } else {
                                    None
                                }
                            } else {
                                match input_ca.get(row_idx) {
                                    Some(bytes) => {
                                        let first_output = self.outputs.values().next().unwrap();
                                        let temp_spec = PipelineSpec {
                                            source: node.source.clone(),
                                            shape_hints: None,
                                            ops: vec![],
                                            sink: first_output.sink.clone(),
                                        };
                                        match decode_source(bytes, &temp_spec) {
                                            Ok(buf) => Some(NodeOutput::from_buffer(buf)),
                                            Err(e) => return Err(format!("Decode error: {e}")),
                                        }
                                    }
                                    None => None,
                                }
                            }
                        }
                    } else {
                        let upstream_id = &node.upstream[0];
                        node_outputs.get(upstream_id).cloned()
                    };
                    if let Some(input) = node_input {
                        let view_dtos: Vec<ViewDto> = if let Some(cached) = precompiled.get(node_id)
                        {
                            cached.clone()
                        } else {
                            let mut dtos = Vec::with_capacity(node.ops.len());
                            for op_spec in &node.ops {
                                match resolve_op(op_spec, row_idx, expr_columns) {
                                    Ok(dto) => dtos.push(dto),
                                    Err(e) => return Err(format!("Op resolution error: {e}")),
                                }
                            }
                            dtos
                        };
                        let mut current_output = input;
                        fn flush_buffer_ops(
                            output: NodeOutput,
                            pending_ops: &mut Vec<ViewDto>,
                        ) -> Result<NodeOutput, String> {
                            if pending_ops.is_empty() {
                                return Ok(output);
                            }
                            let buf = output.as_buffer().ok_or_else(|| {
                                format!(
                                    "Expected Buffer for pending ops, got {:?}",
                                    output.domain()
                                )
                            })?;
                            let mut expr = ViewExpr::new_source((**buf).clone());
                            for op in pending_ops.drain(..) {
                                expr = expr.apply_op(op);
                            }
                            let result = expr.plan().execute();
                            Ok(NodeOutput::from_buffer(result))
                        }
                        let mut pending_buffer_ops: Vec<ViewDto> = Vec::new();
                        for view_dto in view_dtos {
                            match &view_dto {
                                ViewDto::Geometry(geo_op) => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    current_output = execute_geometry_op(current_output, geo_op)?;
                                }
                                ViewDto::Binary { op, other_node_id } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Binary op requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let other_output = node_outputs
                                            .get(other_node_id)
                                            .ok_or_else(|| {
                                                format!(
                                                    "Binary op references unknown node '{other_node_id}'"
                                                )
                                            })?;
                                    let other_buf = other_output.as_buffer().ok_or_else(|| {
                                        format!(
                                            "Binary op other operand must be Buffer, got {:?}",
                                            other_output.domain()
                                        )
                                    })?;
                                    let result = op.execute(current_buf, other_buf);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::ApplyMask {
                                    mask_node_id,
                                    invert,
                                } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ApplyMask requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let mask_output = node_outputs
                                            .get(mask_node_id)
                                            .ok_or_else(|| {
                                                format!(
                                                    "ApplyMask references unknown node '{mask_node_id}'"
                                                )
                                            })?;
                                    let mask_buf = mask_output.as_buffer().ok_or_else(|| {
                                        format!(
                                            "ApplyMask mask must be Buffer, got {:?}",
                                            mask_output.domain()
                                        )
                                    })?;
                                    let result = apply_mask(current_buf, mask_buf, *invert);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::Reduction(reduction_op) => {
                                    use view_buffer::DType;
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Reduction requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let result = reduction_op.execute(current_buf);
                                    if result.shape() == [1] {
                                        // Extract scalar value based on actual dtype
                                        let scalar_val = match result.dtype() {
                                            DType::U8 => result.as_slice::<u8>()[0] as f64,
                                            DType::I8 => result.as_slice::<i8>()[0] as f64,
                                            DType::U16 => result.as_slice::<u16>()[0] as f64,
                                            DType::I16 => result.as_slice::<i16>()[0] as f64,
                                            DType::U32 => result.as_slice::<u32>()[0] as f64,
                                            DType::I32 => result.as_slice::<i32>()[0] as f64,
                                            DType::U64 => result.as_slice::<u64>()[0] as f64,
                                            DType::I64 => result.as_slice::<i64>()[0] as f64,
                                            DType::F32 => result.as_slice::<f32>()[0] as f64,
                                            DType::F64 => result.as_slice::<f64>()[0],
                                        };
                                        current_output = NodeOutput::Scalar(scalar_val);
                                    } else {
                                        current_output = NodeOutput::from_buffer(result);
                                    }
                                }
                                ViewDto::Histogram(histogram_op) => {
                                    use view_buffer::ops::histogram::HistogramOutput;
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Histogram requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let result = histogram_op.execute(current_buf);
                                    match histogram_op.output {
                                        HistogramOutput::Quantized => {
                                            current_output = NodeOutput::from_buffer(result);
                                        }
                                        _ => {
                                            current_output = NodeOutput::from_buffer(result);
                                        }
                                    }
                                }
                                ViewDto::ResizeScale {
                                    scale_x,
                                    scale_y,
                                    filter,
                                } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeScale requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let new_height = (input_height * scale_y).round() as u32;
                                    let new_width = (input_width * scale_x).round() as u32;
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeToHeight { height, filter } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeToHeight requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let aspect = input_width / input_height;
                                    let new_width = (*height as f32 * aspect).round() as u32;
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: *height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeToWidth { width, filter } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeToWidth requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let aspect = input_height / input_width;
                                    let new_height = (*width as f32 * aspect).round() as u32;
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: *width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeMax { max_size, filter } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeMax requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let scale = *max_size as f32 / input_height.max(input_width);
                                    let new_height = (input_height * scale).round() as u32;
                                    let new_width = (input_width * scale).round() as u32;
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::ResizeMin { min_size, filter } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ResizeMin requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    let input_height = shape[0] as f32;
                                    let input_width = shape[1] as f32;
                                    let scale = *min_size as f32 / input_height.min(input_width);
                                    let new_height = (input_height * scale).round() as u32;
                                    let new_width = (input_width * scale).round() as u32;
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: new_width,
                                            height: new_height,
                                            filter: filter.clone(),
                                        },
                                    }));
                                }
                                ViewDto::Pad {
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    value,
                                    mode,
                                } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Pad requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let result = pad_buffer(
                                        current_buf,
                                        *top,
                                        *bottom,
                                        *left,
                                        *right,
                                        *value,
                                        *mode,
                                    );
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::PadToSize {
                                    height,
                                    width,
                                    position,
                                    value,
                                } => {
                                    use view_buffer::ops::dto::{PadMode, PadPosition};
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "PadToSize requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    let current_h = shape[0] as u32;
                                    let current_w = shape[1] as u32;
                                    let pad_h = height.saturating_sub(current_h);
                                    let pad_w = width.saturating_sub(current_w);
                                    let (top, bottom, left, right) = match position {
                                        PadPosition::Center => {
                                            let t = pad_h / 2;
                                            let b = pad_h - t;
                                            let l = pad_w / 2;
                                            let r = pad_w - l;
                                            (t, b, l, r)
                                        }
                                        PadPosition::TopLeft => (0, pad_h, 0, pad_w),
                                        PadPosition::BottomRight => (pad_h, 0, pad_w, 0),
                                    };
                                    let result = pad_buffer(
                                        current_buf,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        *value,
                                        PadMode::Constant,
                                    );
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::Letterbox {
                                    height,
                                    width,
                                    value,
                                } => {
                                    use view_buffer::ops::dto::PadMode;
                                    use view_buffer::FilterType;
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "Letterbox requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    let input_h = shape[0] as f32;
                                    let input_w = shape[1] as f32;
                                    let scale_h = *height as f32 / input_h;
                                    let scale_w = *width as f32 / input_w;
                                    let scale = scale_h.min(scale_w);
                                    let resized_h = (input_h * scale).round() as u32;
                                    let resized_w = (input_w * scale).round() as u32;
                                    pending_buffer_ops.push(ViewDto::Image(ImageOp {
                                        kind: ImageOpKind::Resize {
                                            width: resized_w,
                                            height: resized_h,
                                            filter: FilterType::Lanczos3,
                                        },
                                    }));
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let resized = current_output.as_buffer().ok_or_else(|| {
                                        "Letterbox: resized buffer expected".to_string()
                                    })?;
                                    let pad_h = height.saturating_sub(resized_h);
                                    let pad_w = width.saturating_sub(resized_w);
                                    let top = pad_h / 2;
                                    let bottom = pad_h - top;
                                    let left = pad_w / 2;
                                    let right = pad_w - left;
                                    let result = pad_buffer(
                                        resized,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        *value,
                                        PadMode::Constant,
                                    );
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                ViewDto::ExtractShape => {
                                    // Extract shape from buffer and return as vector
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf =
                                        current_output.as_buffer().ok_or_else(|| {
                                            format!(
                                                "ExtractShape requires Buffer, got {:?}",
                                                current_output.domain()
                                            )
                                        })?;
                                    let shape = current_buf.shape();
                                    // Return shape as f64 vector [height, width, channels]
                                    let shape_vec: Vec<f64> =
                                        shape.iter().map(|&d| d as f64).collect();
                                    current_output = NodeOutput::from_vector(shape_vec);
                                }
                                ViewDto::Materialize => {
                                    // Force materialization of pending ops
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                }
                                _ => {
                                    pending_buffer_ops.push(view_dto.clone());
                                }
                            }
                        }
                        current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                        node_outputs.insert(node_id.clone(), current_output);
                    }
                }
                for (alias, spec) in &self.outputs {
                    if let Some(output) = node_outputs.get(&spec.node) {
                        match encode_node_output(output, &spec.sink) {
                            Ok(encoded) => {
                                let row_result = match encoded {
                                    OutputValue::Binary(bytes) => RowResult::Binary(Some(bytes)),
                                    OutputValue::Scalar(val) => RowResult::Scalar(Some(val)),
                                    OutputValue::Vector(vals) => {
                                        RowResult::Vector(Some((*vals).clone()))
                                    }
                                    OutputValue::Contours(contours) => {
                                        RowResult::Contours(Some((*contours).clone()))
                                    }
                                    OutputValue::TypedList { data, shape } => {
                                        RowResult::TypedList(Some((data, shape)))
                                    }
                                    OutputValue::TypedArray { data, shape } => {
                                        RowResult::TypedArray(Some((data, shape)))
                                    }
                                    OutputValue::NumpyStruct(buf) => {
                                        RowResult::NumpyStruct(Some(buf))
                                    }
                                };
                                results.get_mut(alias).unwrap().push(row_result);
                            }
                            Err(e) => {
                                return Err(format!("Encode error for '{alias}': {e}"));
                            }
                        }
                    } else {
                        let null_result = null_row_result_for_spec(spec);
                        results.get_mut(alias).unwrap().push(null_result);
                    }
                }
            }
            Ok(results)
        }));
        let results = match batch_result {
            Ok(Ok(r)) => r,
            Ok(Err(msg)) => {
                return Err(polars_err!(ComputeError : "Pipeline execution failed: {}", msg));
            }
            Err(panic_payload) => {
                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic during batch execution".to_string()
                };
                return Err(polars_err!(ComputeError : "Pipeline batch failed: {}", panic_msg));
            }
        };
        if self.is_single_output() {
            let spec = self.outputs.get("_output").unwrap();
            let data = results.get("_output").unwrap();
            build_series_from_spec(inputs[0].name().clone(), spec, data)
        } else {
            let mut fields: Vec<Series> = Vec::with_capacity(output_aliases.len());
            for alias in &output_aliases {
                let spec = self.outputs.get(*alias).unwrap();
                let data = results.get(*alias).unwrap();
                let field_series = build_series_from_spec(PlSmallStr::from_str(alias), spec, data)?;
                fields.push(field_series);
            }
            let output_name = inputs[0].name().clone();
            StructChunked::from_series(output_name, len, fields.iter()).map(|sc| sc.into_series())
        }
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
    /// Extract typed data from a ViewBuffer, preserving its dtype.
    pub(crate) fn from_buffer(buf: &ViewBuffer) -> Self {
        let contig = buf.to_contiguous();
        match contig.dtype() {
            view_buffer::DType::U8 => TypedBufferData::U8(contig.as_slice::<u8>().to_vec()),
            view_buffer::DType::I8 => TypedBufferData::I8(contig.as_slice::<i8>().to_vec()),
            view_buffer::DType::U16 => TypedBufferData::U16(contig.as_slice::<u16>().to_vec()),
            view_buffer::DType::I16 => TypedBufferData::I16(contig.as_slice::<i16>().to_vec()),
            view_buffer::DType::U32 => TypedBufferData::U32(contig.as_slice::<u32>().to_vec()),
            view_buffer::DType::I32 => TypedBufferData::I32(contig.as_slice::<i32>().to_vec()),
            view_buffer::DType::U64 => TypedBufferData::U64(contig.as_slice::<u64>().to_vec()),
            view_buffer::DType::I64 => TypedBufferData::I64(contig.as_slice::<i64>().to_vec()),
            view_buffer::DType::F32 => TypedBufferData::F32(contig.as_slice::<f32>().to_vec()),
            view_buffer::DType::F64 => TypedBufferData::F64(contig.as_slice::<f64>().to_vec()),
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
    /// Get the dtype string for this typed data.
    pub(crate) fn dtype_str(&self) -> &'static str {
        match self {
            TypedBufferData::U8(_) => "u8",
            TypedBufferData::I8(_) => "i8",
            TypedBufferData::U16(_) => "u16",
            TypedBufferData::I16(_) => "i16",
            TypedBufferData::U32(_) => "u32",
            TypedBufferData::I32(_) => "i32",
            TypedBufferData::U64(_) => "u64",
            TypedBufferData::I64(_) => "i64",
            TypedBufferData::F32(_) => "f32",
            TypedBufferData::F64(_) => "f64",
        }
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
