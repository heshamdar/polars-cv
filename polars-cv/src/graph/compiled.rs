//! Compiled, cacheable form of a pipeline graph.
//!
//! [`CompiledGraph`] is a pure function of the plugin kwargs
//! (`graph_json` + `expr_column_names`): JSON parsing, topological ordering,
//! expression-parameter slot binding, and static (all-literal) op resolution
//! all happen once at compile time. Because the plugin is registered as
//! elementwise, the streaming engine invokes it once **per morsel** — the
//! process-wide cache ([`get_or_compile`]) makes repeat invocations pay only
//! a hash lookup instead of re-parsing and re-compiling the graph.
//!
//! # Cache-safety invariants
//!
//! A `CompiledGraph` must contain **nothing derived from the data**:
//!
//! - no input dtypes (the `"auto"` dtype/ndim resolution from the input
//!   column type happens per call in [`resolved_output_specs`]),
//! - no shapes, row counts, null masks, or any first-row-derived facts.
//!
//! Per-row decode, per-row shapes, and per-row nulls remain exactly per-row,
//! so one cached graph is valid for inputs of any size, dtype mix, or null
//! pattern. A cache hit additionally requires **full string equality** of the
//! kwargs — the hash alone is never trusted.

use polars::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use view_buffer::geometry::{measures, predicates, Contour};
use view_buffer::ops::NodeOutput;
use view_buffer::{ImageOp, ImageOpKind, ViewBuffer, ViewDto, ViewExpr};

use crate::contour::parse_contour_list;
use crate::execute::{
    decode_contour_source, decode_contour_source_with_dims, decode_source, resolve_op,
};
use crate::params::{ParamCtx, ParamValue};
use crate::pipeline::{OpSpec, PipelineSpec};

use super::decode::{
    apply_mask, build_series_from_spec, decode_binary_zero_copy, decode_list_or_array_source,
    get_binary_row_buffer, null_row_result_for_spec, pad_buffer,
};
use super::encode::{encode_node_output, execute_geometry_op};
use super::types::{OutputSpec, OutputValue, RowErrorPolicy, RowResult, UnifiedGraph};

/// The exact kwargs a graph was compiled from. Stored on the compiled graph
/// so cache hits can be validated by full equality, never by hash alone.
struct GraphKwargsKey {
    graph_json: String,
    expr_column_names: Vec<String>,
}

/// A per-op resolver, fixed at graph-compile time.
pub(crate) enum OpResolver {
    /// All params literal: the `ViewDto` is resolved once and borrowed per row.
    Static(ViewDto),
    /// Has at least one dynamic (slot-bound) param: re-resolved per row with
    /// direct typed slot reads (no string-keyed lookups, no `AnyValue`).
    Dynamic(OpSpec),
}

/// A pipeline graph compiled for repeated execution.
///
/// See the module docs for what may and may not live in here.
pub struct CompiledGraph {
    /// Parsed graph with every expression param bound to an input slot.
    graph: UnifiedGraph,
    /// Per-node op resolvers, aligned with each node's `ops`.
    resolvers: HashMap<String, Vec<OpResolver>>,
    /// Expression column name → absolute input slot (for ops that carry the
    /// column *name* through a view-buffer DTO, e.g. `label_reduce`).
    name_to_slot: HashMap<String, usize>,
    /// Nodes whose source declared `on_error: "null"`: their decode errors
    /// become a null node output instead of failing the row.
    source_null_nodes: std::collections::HashSet<String>,
    /// The exact kwargs this graph was compiled from, kept for exact-match
    /// cache validation.
    key: GraphKwargsKey,
}

/// Per-call execution state: everything derived from the actual input series.
struct ExecState<'a> {
    inputs: &'a [Series],
    /// Typed parameter accessors, built once per call.
    ctx: ParamCtx<'a>,
    /// Output specs with `"auto"` dtype/ndim resolved from the input column
    /// type, sorted by alias.
    resolved_outputs: Vec<(String, OutputSpec)>,
}

impl CompiledGraph {
    /// Compile a graph from the plugin kwargs.
    pub fn compile(graph_json: &str, expr_column_names: &[String]) -> PolarsResult<Self> {
        let mut graph = UnifiedGraph::from_json(graph_json)?;

        // Expression columns are appended to the plugin inputs after the
        // source columns; bind each referenced name to its absolute index.
        let num_source_columns = graph.column_bindings.len().max(1);
        let name_to_slot: HashMap<String, usize> = expr_column_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), num_source_columns + i))
            .collect();

        bind_graph_params(&mut graph, &name_to_slot)?;

        // Parse the per-source error setting once at the edge (the executor
        // checks a set membership instead of comparing strings per row).
        let mut source_null_nodes = std::collections::HashSet::new();
        for (node_id, node) in &graph.nodes {
            match node.source.on_error.as_str() {
                "raise" => {}
                "null" => {
                    source_null_nodes.insert(node_id.clone());
                }
                other => {
                    return Err(polars_err!(ComputeError:
                        "Unknown source on_error value '{}' for node '{}' (expected 'raise' or 'null')",
                        other, node_id
                    ))
                }
            }
        }

        // `_error` is reserved for the error-message field of the
        // null_with_message policy; an output alias would collide with it.
        if graph.on_error == RowErrorPolicy::NullWithMessage && graph.outputs.contains_key("_error")
        {
            return Err(polars_err!(ComputeError:
                "Output alias '_error' is reserved when on_error='null_with_message'"
            ));
        }

        // Resolve all-literal ops once; anything slot-bound re-resolves per row.
        let empty_ctx = ParamCtx::empty();
        let mut resolvers: HashMap<String, Vec<OpResolver>> =
            HashMap::with_capacity(graph.nodes.len());
        for (node_id, node) in &graph.nodes {
            let node_resolvers: Vec<OpResolver> = node
                .ops
                .iter()
                .map(|spec| {
                    if spec.is_all_literal() {
                        resolve_op(spec, 0, &empty_ctx).map(OpResolver::Static)
                    } else {
                        Ok(OpResolver::Dynamic(spec.clone()))
                    }
                })
                .collect::<PolarsResult<_>>()?;
            resolvers.insert(node_id.clone(), node_resolvers);
        }

        Ok(CompiledGraph {
            graph,
            resolvers,
            name_to_slot,
            source_null_nodes,
            key: GraphKwargsKey {
                graph_json: graph_json.to_string(),
                expr_column_names: expr_column_names.to_vec(),
            },
        })
    }

    /// The parsed (compile-time) graph.
    pub fn graph(&self) -> &UnifiedGraph {
        &self.graph
    }

    /// Execute the graph on input series.
    ///
    /// Returns:
    /// - Binary/typed column if single output ("_output" only)
    /// - Struct column with named fields if multiple outputs
    pub fn execute(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError : "No input columns provided"));
        };
        let state = ExecState {
            inputs,
            ctx: ParamCtx::from_inputs(inputs),
            resolved_outputs: resolved_output_specs(&self.graph, inputs.first().map(|s| s.dtype())),
        };

        let mut results: HashMap<String, Vec<RowResult>> = HashMap::new();
        for (alias, _) in &state.resolved_outputs {
            results.insert(alias.clone(), Vec::with_capacity(len));
        }
        let batch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.execute_rows(&state, len, &mut results)
        }));
        let error_messages = match batch_result {
            Ok(Ok(messages)) => messages,
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

        let with_message = self.graph.on_error == RowErrorPolicy::NullWithMessage;
        if self.graph.is_single_output() && !with_message {
            let (_, spec) = &state.resolved_outputs[0];
            let data = results.get("_output").unwrap();
            build_series_from_spec(inputs[0].name().clone(), spec, data)
        } else {
            let mut fields: Vec<Series> = Vec::with_capacity(state.resolved_outputs.len() + 1);
            for (alias, spec) in &state.resolved_outputs {
                let data = results.get(alias).unwrap();
                let field_series = build_series_from_spec(PlSmallStr::from_str(alias), spec, data)?;
                fields.push(field_series);
            }
            if with_message {
                fields.push(Series::new(
                    PlSmallStr::from_static("_error"),
                    error_messages,
                ));
            }
            let output_name = inputs[0].name().clone();
            StructChunked::from_series(output_name, len, fields.iter()).map(|sc| sc.into_series())
        }
    }

    /// The per-row loop: decode sources, run ops, encode outputs.
    ///
    /// Applies the graph's [`RowErrorPolicy`] to per-row errors and returns
    /// one error-message slot per row when the policy is `NullWithMessage`
    /// (an empty Vec otherwise). Errors are `String` so the surrounding
    /// `catch_unwind`/`PolarsError` wrapping stays in one place.
    fn execute_rows(
        &self,
        state: &ExecState<'_>,
        len: usize,
        results: &mut HashMap<String, Vec<RowResult>>,
    ) -> Result<Vec<Option<String>>, String> {
        let policy = self.graph.on_error;
        let with_message = policy == RowErrorPolicy::NullWithMessage;
        let mut error_messages: Vec<Option<String>> = if with_message {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };
        // Allocated once and reused across rows/nodes to avoid per-row churn.
        let mut node_outputs: HashMap<String, NodeOutput> = HashMap::new();
        let mut dto_scratch: Vec<Cow<'_, ViewDto>> = Vec::new();
        for row_idx in 0..len {
            node_outputs.clear();
            let row_result =
                self.execute_one_row(state, row_idx, &mut node_outputs, &mut dto_scratch, results);
            match row_result {
                Ok(()) => {
                    if with_message {
                        error_messages.push(None);
                    }
                }
                Err(msg) => match policy {
                    RowErrorPolicy::Raise => return Err(msg),
                    RowErrorPolicy::Null | RowErrorPolicy::NullWithMessage => {
                        // All-or-nothing per row: drop anything this row may
                        // have pushed before failing, then null every output.
                        for (alias, spec) in &state.resolved_outputs {
                            let rows = results.get_mut(alias).unwrap();
                            rows.truncate(row_idx);
                            rows.push(null_row_result_for_spec(spec));
                        }
                        if with_message {
                            error_messages.push(Some(msg));
                        }
                    }
                },
            }
        }
        Ok(error_messages)
    }

    /// Execute every node and encode every output for one row.
    fn execute_one_row<'g>(
        &'g self,
        state: &ExecState<'_>,
        row_idx: usize,
        node_outputs: &mut HashMap<String, NodeOutput>,
        dto_scratch: &mut Vec<Cow<'g, ViewDto>>,
        results: &mut HashMap<String, Vec<RowResult>>,
    ) -> Result<(), String> {
        let order = self.graph.topological_order();
        let inputs = state.inputs;
        let ctx = &state.ctx;
        {
            for node_id in order {
                let node = match self.graph.nodes.get(node_id) {
                    Some(n) => n,
                    None => continue,
                };
                let has_column_binding = self.graph.column_bindings.contains_key(node_id);
                let node_input: Option<NodeOutput> = if has_column_binding {
                    let on_error_null = self.source_null_nodes.contains(node_id);
                    let decode_result: Result<Option<NodeOutput>, String> = (|| {
                        let col_idx = self
                            .graph
                            .column_bindings
                            .get(node_id)
                            .copied()
                            .unwrap_or(0);
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
                                            return Err(format!(
                                                "Shape buffer has invalid dimensions: expected at least 2D, got {}D",
                                                shape.len()
                                            ));
                                        }
                                        let height = shape[0] as u32;
                                        let width = shape[1] as u32;
                                        let fill_value = node.source.fill_value;
                                        let background = node.source.background;
                                        match decode_contour_source_with_dims(
                                            &value, width, height, fill_value, background,
                                        ) {
                                            Ok(buf) => Ok(Some(NodeOutput::from_buffer(buf))),
                                            Err(e) => Err(format!("Contour decode error: {e}")),
                                        }
                                    } else {
                                        let temp_spec = self.temp_pipeline_spec(node, state);
                                        match decode_contour_source(
                                            &value, row_idx, &temp_spec, ctx,
                                        ) {
                                            Ok(buf) => Ok(Some(NodeOutput::from_buffer(buf))),
                                            Err(e) => Err(format!("Contour decode error: {e}")),
                                        }
                                    }
                                }
                                _ => Ok(None),
                            }
                        // TODO: Add path allowlisting/sandboxing for file_path source.
                        // Currently reads arbitrary local files without sanitization.
                        // This is safe when data comes from trusted input only.
                        } else if source_format == "file_path" {
                            if input_series.dtype() == &DataType::Null {
                                Ok(None)
                            } else {
                                let input_ca = match input_series.str() {
                                    Ok(ca) => ca,
                                    Err(_) => {
                                        return Err(format!(
                                            "Expected String column for file_path source '{node_id}', got {:?}",
                                            input_series.dtype()
                                        ));
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
                                        let mut temp_spec = self.temp_pipeline_spec(node, state);
                                        temp_spec.source.format = "image_bytes".to_string();
                                        match decode_source(&bytes, &temp_spec) {
                                            Ok(buf) => Ok(Some(NodeOutput::from_buffer(buf))),
                                            Err(e) => {
                                                Err(format!("Decode error for file '{path}': {e}"))
                                            }
                                        }
                                    }
                                    None => Ok(None),
                                }
                            }
                        } else if node.source.format == "list" || node.source.format == "array" {
                            if input_series.dtype() == &DataType::Null {
                                Ok(None)
                            } else {
                                let dtype_opt = node.source.dtype.as_deref();
                                let require_contiguous = node.source.require_contiguous;
                                match decode_list_or_array_source(
                                    input_series,
                                    row_idx,
                                    dtype_opt,
                                    require_contiguous,
                                ) {
                                    Ok(Some(buf)) => Ok(Some(NodeOutput::from_buffer(buf))),
                                    Ok(None) => Ok(None),
                                    Err(e) => Err(format!("List/Array decode error: {e}")),
                                }
                            }
                        } else if input_series.dtype() == &DataType::Null {
                            Ok(None)
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
                                        Ok(buf) => Ok(Some(NodeOutput::from_buffer(buf))),
                                        Err(e) => Err(format!("Zero-copy decode error: {e}")),
                                    }
                                } else {
                                    Ok(None)
                                }
                            } else {
                                match input_ca.get(row_idx) {
                                    Some(bytes) => {
                                        let temp_spec = self.temp_pipeline_spec(node, state);
                                        match decode_source(bytes, &temp_spec) {
                                            Ok(buf) => Ok(Some(NodeOutput::from_buffer(buf))),
                                            Err(e) => Err(format!("Decode error: {e}")),
                                        }
                                    }
                                    None => Ok(None),
                                }
                            }
                        }
                    })(
                    );
                    match decode_result {
                        Ok(output) => output,
                        Err(_e) if on_error_null => None,
                        Err(e) => return Err(e),
                    }
                } else {
                    let upstream_id = &node.upstream[0];
                    node_outputs.get(upstream_id).cloned()
                };
                if let Some(input) = node_input {
                    // Static ops are borrowed from the compiled graph; dynamic
                    // ops are re-resolved for this row through typed slot
                    // reads. The scratch Vec is reused so steady-state rows
                    // allocate nothing here.
                    dto_scratch.clear();
                    if let Some(resolvers) = self.resolvers.get(node_id) {
                        for resolver in resolvers {
                            match resolver {
                                OpResolver::Static(dto) => dto_scratch.push(Cow::Borrowed(dto)),
                                OpResolver::Dynamic(spec) => match resolve_op(spec, row_idx, ctx) {
                                    Ok(dto) => dto_scratch.push(Cow::Owned(dto)),
                                    Err(e) => return Err(format!("Op resolution error: {e}")),
                                },
                            }
                        }
                    }
                    let mut current_output = input;
                    fn flush_buffer_ops(
                        output: NodeOutput,
                        pending_ops: &mut Vec<ViewDto>,
                    ) -> Result<NodeOutput, String> {
                        if pending_ops.is_empty() {
                            return Ok(output);
                        }
                        let buf = output.as_buffer().ok_or_else(|| {
                            format!("Expected Buffer for pending ops, got {:?}", output.domain())
                        })?;
                        let mut expr = ViewExpr::new_source((**buf).clone());
                        for op in pending_ops.drain(..) {
                            expr = expr.apply_op(op);
                        }
                        let result = expr.plan().execute();
                        Ok(NodeOutput::from_buffer(result))
                    }
                    let mut pending_buffer_ops: Vec<ViewDto> = Vec::new();
                    for view_dto in dto_scratch.iter() {
                        match view_dto.as_ref() {
                            ViewDto::Geometry(geo_op) => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                current_output = execute_geometry_op(current_output, geo_op)?;
                            }
                            ViewDto::Binary { op, other_node_id } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "Binary op requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let other_output =
                                    node_outputs.get(other_node_id).ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ApplyMask requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let mask_output =
                                    node_outputs.get(mask_node_id).ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "Histogram requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let result = histogram_op.execute(current_buf);
                                current_output = NodeOutput::from_buffer(result);
                            }
                            ViewDto::ResizeScale {
                                scale_x,
                                scale_y,
                                filter,
                            } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
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
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ExtractShape requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let shape = current_buf.shape();
                                // Return shape as f64 vector [height, width, channels]
                                let shape_vec: Vec<f64> = shape.iter().map(|&d| d as f64).collect();
                                current_output = NodeOutput::from_vector(shape_vec);
                            }
                            ViewDto::LabelReduce {
                                contours_expr,
                                reduction,
                                region_mode,
                            } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "LabelReduce requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let slot =
                                    self.name_to_slot.get(contours_expr).ok_or_else(|| {
                                        format!(
                                            "LabelReduce contour column '{contours_expr}' not found in expression inputs"
                                        )
                                    })?;
                                let contour_col = ctx.col(*slot).map_err(|e| e.to_string())?;
                                let contour_value = contour_col.get_any(row_idx).map_err(|e| {
                                    format!(
                                        "LabelReduce failed to read contours at row {row_idx}: {e}"
                                    )
                                })?;
                                if contour_value.is_null() {
                                    current_output = NodeOutput::from_vector(Vec::new());
                                    continue;
                                }
                                let contours = parse_contour_list(&contour_value).map_err(|e| {
                                    format!("LabelReduce contour parsing failed: {e}")
                                })?;
                                let parsed_reduction = parse_label_reduction(reduction)?;
                                let parsed_region_mode = parse_label_region_mode(region_mode)?;
                                let grid = buffer_to_grid(current_buf)?;
                                let scores: Vec<f64> = contours
                                    .iter()
                                    .map(|contour| {
                                        contour_score_on_grid(
                                            contour,
                                            &grid,
                                            parsed_reduction,
                                            parsed_region_mode,
                                        )
                                    })
                                    .collect();
                                current_output = NodeOutput::from_vector(scores);
                            }
                            ViewDto::ChannelSwap { order } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ChannelSwap requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let result = view_buffer::apply_channel_swap(current_buf, order);
                                current_output = NodeOutput::from_buffer(result);
                            }
                            ViewDto::ChannelMerge { other_node_ids } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ChannelMerge requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let mut all_bufs: Vec<&ViewBuffer> = vec![current_buf];
                                for other_id in other_node_ids {
                                    let other_output =
                                        node_outputs.get(other_id).ok_or_else(|| {
                                            format!("ChannelMerge: node '{other_id}' not found")
                                        })?;
                                    let other_buf = other_output.as_buffer().ok_or_else(|| {
                                        format!("ChannelMerge: node '{other_id}' is not a Buffer")
                                    })?;
                                    all_bufs.push(other_buf);
                                }
                                let result = view_buffer::apply_channel_merge(&all_bufs);
                                current_output = NodeOutput::from_buffer(result);
                            }
                            ViewDto::Color(color_op) => {
                                use view_buffer::ops::color::apply_color_convert;
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ColorConvert requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let result = apply_color_convert(current_buf, color_op);
                                current_output = NodeOutput::from_buffer(result);
                            }
                            ViewDto::Filter(convolve_op) => {
                                use view_buffer::ops::filter::apply_convolve2d;
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "Convolve2D requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let result = apply_convolve2d(current_buf, convolve_op);
                                current_output = NodeOutput::from_buffer(result);
                            }
                            ViewDto::Materialize => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                            }
                            _ => {
                                pending_buffer_ops.push(view_dto.as_ref().clone());
                            }
                        }
                    }
                    current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                    node_outputs.insert(node_id.clone(), current_output);
                }
            }
            for (alias, spec) in &state.resolved_outputs {
                if let Some(output) = node_outputs.get(&spec.node) {
                    // Validate that the buffer dtype matches the planned
                    // expected_dtype (inferred by the Python planner from
                    // this op's view-buffer contract).
                    // "auto" is resolved elsewhere and skipped here.
                    if spec.expected_dtype != "auto" {
                        if let Some(buf) = output.as_buffer() {
                            if let Ok(expected) =
                                super::decode::parse_dtype_str(&spec.expected_dtype)
                            {
                                let actual = buf.dtype();
                                if actual != expected {
                                    return Err(format!(
                                        "Output '{alias}': planned dtype {expected:?} but execution \
                                         produced {actual:?}. This indicates a mismatch between \
                                         the planner's view-buffer contract and the Rust implementation."
                                    ));
                                }
                            }
                        }
                    }
                    match encode_node_output(output, spec) {
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
                                OutputValue::NumpyStruct(buf) => RowResult::NumpyStruct(Some(buf)),
                                OutputValue::HistogramBuckets(buckets) => {
                                    RowResult::HistogramBuckets(Some(buckets))
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
        Ok(())
    }

    /// Build the throwaway `PipelineSpec` some decode paths expect.
    ///
    /// Only the source spec matters to those paths; the sink is filled from
    /// the first output for structural completeness.
    fn temp_pipeline_spec(
        &self,
        node: &super::types::GraphNode,
        state: &ExecState<'_>,
    ) -> PipelineSpec {
        PipelineSpec {
            source: node.source.clone(),
            shape_hints: None,
            ops: vec![],
            sink: state.resolved_outputs[0].1.sink.clone(),
        }
    }
}

/// Bind every expression parameter in the graph to its input slot.
///
/// `label_reduce`'s `contours` param is deliberately left as `Expr` — the
/// column *name* travels through the view-buffer DTO and is mapped to a slot
/// by the executor (see the `LabelReduce` arm in `execute_rows`).
fn bind_graph_params(
    graph: &mut UnifiedGraph,
    name_to_slot: &HashMap<String, usize>,
) -> PolarsResult<()> {
    for node in graph.nodes.values_mut() {
        if let Some(w) = node.source.width.as_mut() {
            bind_param(w, name_to_slot)?;
        }
        if let Some(h) = node.source.height.as_mut() {
            bind_param(h, name_to_slot)?;
        }
        for op in node.ops.iter_mut() {
            let keep_named = op.op == "label_reduce";
            for (pname, p) in op.params.iter_mut() {
                if keep_named && pname == "contours" {
                    continue;
                }
                bind_param(p, name_to_slot)?;
            }
        }
    }
    Ok(())
}

/// Rewrite one `Expr` param to its bound `Slot` form (literals untouched).
fn bind_param(p: &mut ParamValue, name_to_slot: &HashMap<String, usize>) -> PolarsResult<()> {
    if let ParamValue::Expr { col, .. } = p {
        let name = col
            .as_deref()
            .ok_or_else(|| polars_err!(ComputeError: "Expression parameter missing column name"))?;
        let idx = name_to_slot.get(name).ok_or_else(|| {
            polars_err!(ComputeError:
                "Column '{}' not found in expression inputs", name
            )
        })?;
        *p = ParamValue::Slot { idx: *idx };
    }
    Ok(())
}

/// Resolve `"auto"` dtype and missing ndim on the graph's output specs from
/// the first input column's type, returning per-call clones sorted by alias.
///
/// This is the single implementation shared by the planning-time
/// (`unified_output_dtype`) and execution-time entry points, so the inferred
/// schema cannot diverge between the two. It must stay per-call (never cached):
/// the resolution depends on the input column's dtype.
///
/// Only the leaf type of List/Array sources is meaningful for dtype: for
/// Binary/String (image/file) sources the column type does not reflect the
/// decoded buffer dtype, so `"auto"` is left unresolved.
pub(crate) fn resolved_output_specs(
    graph: &UnifiedGraph,
    first_input_dtype: Option<&DataType>,
) -> Vec<(String, OutputSpec)> {
    let mut specs: Vec<(String, OutputSpec)> = graph
        .outputs
        .iter()
        .map(|(alias, spec)| (alias.clone(), spec.clone()))
        .collect();
    specs.sort_by(|a, b| a.0.cmp(&b.0));

    let Some(dt) = first_input_dtype else {
        return specs;
    };
    let (leaf_dtype, ndim) = peel_nesting(dt);
    let inferred_dtype_str = polars_dtype_to_str(&leaf_dtype);

    for (_, spec) in specs.iter_mut() {
        if spec.expected_dtype == "auto" {
            match &leaf_dtype {
                // Image/file sources: cannot infer buffer dtype from column type.
                DataType::Binary | DataType::String | DataType::Null => {}
                // List/array sources: leaf type is meaningful.
                _ => spec.expected_dtype = inferred_dtype_str.to_string(),
            }
        }
        if spec.expected_ndim.is_none() && ndim > 0 {
            spec.expected_ndim = Some(ndim);
        }
    }
    specs
}

/// Recursively peel List/Array nesting to find the leaf dtype and depth.
fn peel_nesting(dt: &DataType) -> (DataType, usize) {
    match dt {
        DataType::List(inner) => {
            let (leaf, depth) = peel_nesting(inner);
            (leaf, depth + 1)
        }
        DataType::Array(inner, _) => {
            let (leaf, depth) = peel_nesting(inner);
            (leaf, depth + 1)
        }
        other => (other.clone(), 0),
    }
}

/// Convert a Polars DataType to the dtype string used in output specs.
fn polars_dtype_to_str(dt: &DataType) -> &'static str {
    match dt {
        DataType::UInt8 => "u8",
        DataType::Int8 => "i8",
        DataType::UInt16 => "u16",
        DataType::Int16 => "i16",
        DataType::UInt32 => "u32",
        DataType::Int32 => "i32",
        DataType::UInt64 => "u64",
        DataType::Int64 => "i64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        _ => "u8", // fallback for non-numeric types
    }
}

// ============================================================================
// Process-wide compiled-graph cache
// ============================================================================

/// Maximum number of compiled graphs kept resident. Each entry is small
/// (the parsed spec, no data), so this mostly bounds pathological churn from
/// many distinct pipelines in one process.
const GRAPH_CACHE_CAP: usize = 32;

type CacheEntries = Vec<(u64, Arc<CompiledGraph>)>;

fn graph_cache() -> &'static Mutex<CacheEntries> {
    static CACHE: OnceLock<Mutex<CacheEntries>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn cache_key_hash(graph_json: &str, expr_column_names: &[String]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    graph_json.hash(&mut hasher);
    expr_column_names.hash(&mut hasher);
    hasher.finish()
}

/// Fetch the compiled form of a graph, compiling and caching on miss.
///
/// A hit requires both the hash **and** full equality of the kwargs against
/// the stored copy — the hash alone is never trusted. Most-recently-used
/// entries are kept at the front; the lock is never held during compilation.
pub(crate) fn get_or_compile(
    graph_json: &str,
    expr_column_names: &[String],
) -> PolarsResult<Arc<CompiledGraph>> {
    let hash = cache_key_hash(graph_json, expr_column_names);

    {
        let mut cache = graph_cache().lock().unwrap();
        if let Some(pos) = cache.iter().position(|(h, compiled)| {
            *h == hash
                && compiled.key.graph_json == graph_json
                && compiled.key.expr_column_names == expr_column_names
        }) {
            let entry = cache.remove(pos);
            let compiled = entry.1.clone();
            cache.insert(0, entry);
            return Ok(compiled);
        }
    }

    // Compile outside the lock; concurrent misses may compile the same graph
    // twice, which is harmless (the result is deterministic).
    let compiled = Arc::new(CompiledGraph::compile(graph_json, expr_column_names)?);

    let mut cache = graph_cache().lock().unwrap();
    let already_present = cache.iter().any(|(h, c)| {
        *h == hash && c.key.graph_json == graph_json && c.key.expr_column_names == expr_column_names
    });
    if !already_present {
        cache.insert(0, (hash, compiled.clone()));
        cache.truncate(GRAPH_CACHE_CAP);
    }
    Ok(compiled)
}

// ============================================================================
// label_reduce helpers (used only by the executor)
// ============================================================================

#[derive(Clone, Copy)]
enum LabelReduction {
    Max,
    Mean,
    Sum,
}

#[derive(Clone, Copy)]
enum LabelRegionMode {
    Interior,
    Boundary,
    Bbox,
}

fn parse_label_reduction(value: &str) -> Result<LabelReduction, String> {
    match value {
        "max" => Ok(LabelReduction::Max),
        "mean" => Ok(LabelReduction::Mean),
        "sum" => Ok(LabelReduction::Sum),
        other => Err(format!(
            "Unsupported reduction '{other}'. Expected one of: max, mean, sum"
        )),
    }
}

fn parse_label_region_mode(value: &str) -> Result<LabelRegionMode, String> {
    match value {
        "interior" => Ok(LabelRegionMode::Interior),
        "boundary" => Ok(LabelRegionMode::Boundary),
        "bbox" => Ok(LabelRegionMode::Bbox),
        other => Err(format!(
            "Unsupported region_mode '{other}'. Expected one of: interior, boundary, bbox"
        )),
    }
}

fn buffer_to_grid(buffer: &ViewBuffer) -> Result<Vec<Vec<f64>>, String> {
    let shape = buffer.shape();
    if shape.len() < 2 {
        return Err(format!(
            "label_reduce requires at least 2D buffer input, got shape {:?}",
            shape
        ));
    }
    let height = shape[0];
    let width = shape[1];
    let channels = if shape.len() > 2 { shape[2] } else { 1 };
    if channels != 1 {
        return Err(format!(
            "label_reduce currently requires a single-channel buffer, got {channels} channels"
        ));
    }
    let contig = buffer.to_contiguous();

    /// Read the (y, x) grid values out of one concrete element type.
    macro_rules! fill_grid {
        ($t:ty) => {{
            let data = contig.as_slice::<$t>();
            let mut grid = vec![vec![0.0; width]; height];
            for y in 0..height {
                for x in 0..width {
                    grid[y][x] = data[y * width + x] as f64;
                }
            }
            grid
        }};
    }
    let grid = match contig.dtype() {
        view_buffer::DType::U8 => fill_grid!(u8),
        view_buffer::DType::I8 => fill_grid!(i8),
        view_buffer::DType::U16 => fill_grid!(u16),
        view_buffer::DType::I16 => fill_grid!(i16),
        view_buffer::DType::U32 => fill_grid!(u32),
        view_buffer::DType::I32 => fill_grid!(i32),
        view_buffer::DType::U64 => fill_grid!(u64),
        view_buffer::DType::I64 => fill_grid!(i64),
        view_buffer::DType::F32 => fill_grid!(f32),
        view_buffer::DType::F64 => fill_grid!(f64),
    };
    Ok(grid)
}

fn contour_score_on_grid(
    contour: &Contour,
    grid: &[Vec<f64>],
    reduction: LabelReduction,
    region_mode: LabelRegionMode,
) -> f64 {
    let height = grid.len();
    if height == 0 {
        return 0.0;
    }
    let width = grid[0].len();
    if width == 0 {
        return 0.0;
    }
    let Some(bbox) = contour.bounding_box() else {
        return 0.0;
    };

    let x0 = bbox.x.floor().max(0.0) as usize;
    let y0 = bbox.y.floor().max(0.0) as usize;
    let x1 = (bbox.x + bbox.width).ceil().min(width as f64).max(0.0) as usize;
    let y1 = (bbox.y + bbox.height).ceil().min(height as f64).max(0.0) as usize;
    if x0 >= x1 || y0 >= y1 {
        return 0.0;
    }

    let mut acc = 0.0;
    let mut max_val = f64::NEG_INFINITY;
    let mut count = 0usize;
    for (y, row) in grid.iter().enumerate().skip(y0).take(y1 - y0) {
        for (x, value) in row.iter().enumerate().skip(x0).take(x1 - x0) {
            let include = match region_mode {
                LabelRegionMode::Bbox => true,
                LabelRegionMode::Interior => {
                    predicates::contains_point(contour, x as f64 + 0.5, y as f64 + 0.5)
                }
                LabelRegionMode::Boundary => {
                    predicates::point_in_contour(
                        &view_buffer::geometry::Point::new(x as f64 + 0.5, y as f64 + 0.5),
                        contour,
                    ) >= 0
                }
            };
            if include {
                let val = *value;
                acc += val;
                max_val = max_val.max(val);
                count += 1;
            }
        }
    }
    if count == 0 {
        // Centroid fallback: sub-pixel contours have an empty rasterized
        // interior.  Sample the grid at the contour centroid instead so
        // that single-pixel detections receive their actual pixel value.
        let c = measures::centroid(contour);
        let cx = c.x.floor() as usize;
        let cy = c.y.floor() as usize;
        if cy < height && cx < width {
            return grid[cy][cx];
        }
        return 0.0;
    }
    match reduction {
        LabelReduction::Max => max_val,
        LabelReduction::Mean => acc / count as f64,
        LabelReduction::Sum => acc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_GRAPH: &str = r#"{
        "nodes": {
            "n0": {
                "source": {"format": "blob"},
                "ops": [{"op": "relu"}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        },
        "column_bindings": {"n0": 0}
    }"#;

    const DYNAMIC_GRAPH: &str = r#"{
        "nodes": {
            "n0": {
                "source": {"format": "blob"},
                "ops": [{"op": "scale", "factor": {"type": "expr", "col": "f"}}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        },
        "column_bindings": {"n0": 0}
    }"#;

    #[test]
    fn cache_hit_returns_same_compilation() {
        let names: Vec<String> = vec![];
        let a = get_or_compile(SIMPLE_GRAPH, &names).unwrap();
        let b = get_or_compile(SIMPLE_GRAPH, &names).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "identical kwargs must hit the cache, not recompile"
        );
    }

    #[test]
    fn different_expr_names_do_not_collide() {
        let a = get_or_compile(DYNAMIC_GRAPH, &["f".to_string()]).unwrap();
        let b = get_or_compile(DYNAMIC_GRAPH, &["f".to_string(), "g".to_string()]).unwrap();
        assert!(
            !Arc::ptr_eq(&a, &b),
            "same JSON with different expr columns must compile separately"
        );
    }

    #[test]
    fn static_ops_are_precompiled_and_dynamic_are_not() {
        let compiled = CompiledGraph::compile(SIMPLE_GRAPH, &[]).unwrap();
        assert!(matches!(compiled.resolvers["n0"][0], OpResolver::Static(_)));

        let compiled = CompiledGraph::compile(DYNAMIC_GRAPH, &["f".to_string()]).unwrap();
        assert!(matches!(
            compiled.resolvers["n0"][0],
            OpResolver::Dynamic(_)
        ));
        // The dynamic op's expr param must have been bound to a slot:
        // 1 source column + position 0 → absolute slot 1.
        match &compiled.resolvers["n0"][0] {
            OpResolver::Dynamic(spec) => match spec.params.get("factor").unwrap() {
                ParamValue::Slot { idx } => assert_eq!(*idx, 1),
                other => panic!("expected bound slot, got {other:?}"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn unknown_expr_column_fails_at_compile_time() {
        let err = CompiledGraph::compile(DYNAMIC_GRAPH, &[])
            .err()
            .expect("compiling with a missing expr column must fail");
        assert!(err.to_string().contains("not found in expression inputs"));
    }
}
