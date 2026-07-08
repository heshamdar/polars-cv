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
use view_buffer::geometry::label::score_contours_on_buffer;
use view_buffer::ops::NodeOutput;
use view_buffer::{ViewBuffer, ViewDto, ViewExpr};

use crate::contour::parse_contour_list;
use crate::execute::{
    decode_contour_source, decode_contour_source_with_dims, decode_source, resolve_op,
};
use crate::params::{ParamCtx, ParamValue};
use crate::pipeline::OpSpec;

use super::step::GraphStep;

use super::decode::{
    build_series_from_spec, decode_binary_zero_copy, decode_list_or_array_source,
    get_binary_row_buffer, null_row_result_for_spec,
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
    /// All params literal: the `GraphStep` is resolved once and borrowed per row.
    Static(GraphStep),
    /// Has at least one dynamic (slot-bound) param: re-resolved per row with
    /// direct typed slot reads (no string-keyed lookups, no `AnyValue`).
    Dynamic(OpSpec),
    /// `rasterize(shape=<node>)`: output dimensions come from another node's
    /// buffer at execution time (the referenced node is an upstream
    /// dependency, so it has already run); the remaining params resolve from
    /// the spec like any dynamic op.
    RasterizeShapeRef { spec: OpSpec, shape_node: String },
}

/// One op of a node's chain, resolved for the current row.
enum ResolvedStep<'a> {
    Step(Cow<'a, GraphStep>),
    RasterizeShapeRef {
        spec: &'a OpSpec,
        shape_node: &'a str,
    },
}

/// A pipeline graph compiled for repeated execution.
///
/// See the module docs for what may and may not live in here.
pub struct CompiledGraph {
    /// Parsed graph with every expression param bound to an input slot.
    graph: UnifiedGraph,
    /// Per-node op resolvers, aligned with each node's `ops`.
    resolvers: HashMap<String, Vec<OpResolver>>,
    /// Expression column name → absolute input slot (for steps that carry
    /// the column *name*, e.g. `label_reduce`).
    name_to_slot: HashMap<String, usize>,
    /// Nodes whose source declared `on_error: "null"`: their decode errors
    /// become a null node output instead of failing the row.
    source_null_nodes: std::collections::HashSet<String>,
    /// Per-node cloud credentials for `file_path` sources, parsed once at
    /// the edge from the source spec's string map.
    cloud_options: HashMap<String, crate::cloud::CloudOptions>,
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
    /// Bytes of this batch's remote `file_path` sources, fetched concurrently
    /// up front (node_id → path → result). Converts per-row network latency
    /// into per-batch latency; per-path errors surface at their row so the
    /// usual error policies apply.
    prefetched: HashMap<String, HashMap<String, Result<Vec<u8>, String>>>,
}

/// Maximum concurrent remote fetches during source prefetch.
const PREFETCH_CONCURRENCY: usize = 16;

impl CompiledGraph {
    /// Compile a graph from the plugin kwargs.
    pub fn compile(graph_json: &str, expr_column_names: &[String]) -> PolarsResult<Self> {
        let mut graph = UnifiedGraph::from_json(graph_json)?;
        validate_graph_structure(&graph)?;

        // Expression columns are appended to the plugin inputs after the
        // source columns; bind each referenced name to its absolute index.
        // Several root nodes may share one input column (one binding entry
        // each, same column index), so the offset is the number of *distinct*
        // columns — counting entries would shift every expression slot.
        let num_source_columns = graph
            .column_bindings
            .values()
            .copied()
            .max()
            .map(|max_idx| max_idx + 1)
            .unwrap_or(1);
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

        // Parse per-node cloud credentials once at the edge.
        let mut cloud_options: HashMap<String, crate::cloud::CloudOptions> = HashMap::new();
        for (node_id, node) in &graph.nodes {
            if let Some(map) = &node.source.cloud_options {
                cloud_options.insert(node_id.clone(), crate::cloud::CloudOptions::from_map(map));
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
            let mut node_resolvers: Vec<OpResolver> = Vec::with_capacity(node.ops.len());
            for spec in &node.ops {
                // rasterize(shape=<node>) carries a shape_ref instead of
                // width/height; it gets a dedicated resolver because its
                // dimensions come from another node's output, not a param.
                if spec.op == "rasterize" {
                    if let Some(shape_ref) = spec.params.get("shape_ref") {
                        let shape_node = shape_ref.resolve_string()?.to_string();
                        if !graph.nodes.contains_key(&shape_node) {
                            return Err(polars_err!(ComputeError:
                                "Node '{}': rasterize shape reference '{}' is not a node in the graph",
                                node_id, shape_node
                            ));
                        }
                        node_resolvers.push(OpResolver::RasterizeShapeRef {
                            spec: spec.clone(),
                            shape_node,
                        });
                        continue;
                    }
                }
                if spec.is_all_literal() {
                    node_resolvers.push(OpResolver::Static(resolve_op(spec, 0, &empty_ctx)?));
                } else {
                    node_resolvers.push(OpResolver::Dynamic(spec.clone()));
                }
            }
            resolvers.insert(node_id.clone(), node_resolvers);
        }

        Ok(CompiledGraph {
            graph,
            resolvers,
            name_to_slot,
            source_null_nodes,
            cloud_options,
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
        // Column bindings are compile-time data but their bounds depend on
        // this call's inputs: check once here instead of per row.
        for (node_id, col_idx) in &self.graph.column_bindings {
            if *col_idx >= inputs.len() {
                return Err(polars_err!(ComputeError:
                    "Column index {} out of bounds for node '{}' ({} input columns)",
                    col_idx, node_id, inputs.len()
                ));
            }
        }
        let state = ExecState {
            inputs,
            ctx: ParamCtx::from_inputs(inputs),
            resolved_outputs: resolved_output_specs(&self.graph, inputs.first().map(|s| s.dtype())),
            prefetched: self.prefetch_remote_sources(inputs),
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
            // Take ownership of the row results so the encoder can move each
            // row's bytes/buffers into Arrow instead of copying them.
            let data = results.remove("_output").unwrap();
            build_series_from_spec(inputs[0].name().clone(), spec, data)
        } else {
            let mut fields: Vec<Series> = Vec::with_capacity(state.resolved_outputs.len() + 1);
            for (alias, spec) in &state.resolved_outputs {
                let data = results.remove(alias).unwrap();
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
        let mut dto_scratch: Vec<ResolvedStep<'_>> = Vec::new();
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
        dto_scratch: &mut Vec<ResolvedStep<'g>>,
        results: &mut HashMap<String, Vec<RowResult>>,
    ) -> Result<(), String> {
        self.run_row_nodes(state, row_idx, node_outputs, dto_scratch)?;
        for (alias, spec) in &state.resolved_outputs {
            if let Some(output) = node_outputs.get(&spec.node) {
                validate_output_dtype(alias, spec, output)?;
                match encode_node_output(output, spec) {
                    Ok(encoded) => {
                        let row_result = match encoded {
                            OutputValue::Binary(bytes) => RowResult::Binary(Some(bytes)),
                            OutputValue::Scalar(val) => RowResult::Scalar(Some(val)),
                            OutputValue::Vector(vals) => RowResult::Vector(Some((*vals).clone())),
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
        Ok(())
    }

    /// Run every node of the graph for one row, leaving each node's output in
    /// `node_outputs` (no output encoding).
    fn run_row_nodes<'g>(
        &'g self,
        state: &ExecState<'_>,
        row_idx: usize,
        node_outputs: &mut HashMap<String, NodeOutput>,
        dto_scratch: &mut Vec<ResolvedStep<'g>>,
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
                        // Bounds are validated once per call in `execute()`.
                        let col_idx = self
                            .graph
                            .column_bindings
                            .get(node_id)
                            .copied()
                            .unwrap_or(0);
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
                                        match decode_contour_source(
                                            &value,
                                            row_idx,
                                            &node.source,
                                            ctx,
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
                                        // Remote sources were fetched concurrently before
                                        // the row loop; local files are read inline.
                                        let owned_bytes: Vec<u8>;
                                        let bytes: &[u8] = if crate::cloud::is_remote_path(path) {
                                            match state
                                                .prefetched
                                                .get(node_id)
                                                .and_then(|m| m.get(path))
                                            {
                                                Some(Ok(b)) => b.as_slice(),
                                                Some(Err(e)) => {
                                                    return Err(format!(
                                                        "Failed to read remote file '{path}': {e}"
                                                    ));
                                                }
                                                // Defensive: every remote path in the batch
                                                // is prefetched, but fall back to an inline
                                                // fetch rather than miss.
                                                None => {
                                                    owned_bytes = crate::cloud::read_file(
                                                        path,
                                                        self.cloud_options.get(node_id),
                                                    )
                                                    .map_err(|e| {
                                                        format!(
                                                            "Failed to read remote file '{path}': {e}"
                                                        )
                                                    })?;
                                                    owned_bytes.as_slice()
                                                }
                                            }
                                        } else {
                                            // cloud::read_file handles both bare
                                            // paths and file:// URLs (which
                                            // std::fs::read would reject with
                                            // ENOENT, scheme and all).
                                            owned_bytes = crate::cloud::read_file(path, None)
                                                .map_err(|e| {
                                                    format!(
                                                        "Failed to read local file '{path}': {e}"
                                                    )
                                                })?;
                                            owned_bytes.as_slice()
                                        };
                                        // file_path contents decode like image bytes.
                                        let mut source = node.source.clone();
                                        source.format = "image_bytes".to_string();
                                        match decode_source(bytes, &source) {
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
                                    Some(bytes) => match decode_source(bytes, &node.source) {
                                        Ok(buf) => Ok(Some(NodeOutput::from_buffer(buf))),
                                        Err(e) => Err(format!("Decode error: {e}")),
                                    },
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
                                OpResolver::Static(step) => {
                                    dto_scratch.push(ResolvedStep::Step(Cow::Borrowed(step)))
                                }
                                OpResolver::Dynamic(spec) => match resolve_op(spec, row_idx, ctx) {
                                    Ok(step) => {
                                        dto_scratch.push(ResolvedStep::Step(Cow::Owned(step)))
                                    }
                                    Err(e) => return Err(format!("Op resolution error: {e}")),
                                },
                                OpResolver::RasterizeShapeRef { spec, shape_node } => dto_scratch
                                    .push(ResolvedStep::RasterizeShapeRef { spec, shape_node }),
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
                    for step in dto_scratch.iter() {
                        let graph_step = match step {
                            ResolvedStep::RasterizeShapeRef { spec, shape_node } => {
                                // Dimensions come from the referenced node's
                                // buffer (already executed: it is upstream).
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let shape_output =
                                    node_outputs.get(*shape_node).ok_or_else(|| {
                                        format!(
                                            "Rasterize shape reference '{shape_node}' produced no output"
                                        )
                                    })?;
                                let shape_buf = shape_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "Rasterize shape reference '{shape_node}' must be a Buffer, got {:?}",
                                        shape_output.domain()
                                    )
                                })?;
                                let dims = shape_buf.shape();
                                if dims.len() < 2 {
                                    return Err(format!(
                                        "Rasterize shape reference '{shape_node}' must be at least 2D, got {}D",
                                        dims.len()
                                    ));
                                }
                                let height = dims[0] as u32;
                                let width = dims[1] as u32;
                                let (fill_value, background, anti_alias) =
                                    crate::execute::resolve_rasterize_style(
                                        &spec.params,
                                        row_idx,
                                        ctx,
                                    )
                                    .map_err(|e| e.to_string())?;
                                let geo_op = view_buffer::GeometryOp::Rasterize {
                                    width,
                                    height,
                                    fill_value,
                                    background,
                                    anti_alias,
                                };
                                current_output = execute_geometry_op(current_output, &geo_op)?;
                                continue;
                            }
                            ResolvedStep::Step(step) => step,
                        };
                        match graph_step.as_ref() {
                            GraphStep::Geometry(geo_op) => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                current_output = execute_geometry_op(current_output, geo_op)?;
                            }
                            GraphStep::Binary { op, other } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "Binary op requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let other_output = node_outputs.get(other).ok_or_else(|| {
                                    format!("Binary op references unknown node '{other}'")
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
                            GraphStep::ApplyMask { mask, invert } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ApplyMask requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let mask_output = node_outputs.get(mask).ok_or_else(|| {
                                    format!("ApplyMask references unknown node '{mask}'")
                                })?;
                                let mask_buf = mask_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ApplyMask mask must be Buffer, got {:?}",
                                        mask_output.domain()
                                    )
                                })?;
                                let result =
                                    view_buffer::apply_mask(current_buf, mask_buf, *invert);
                                current_output = NodeOutput::from_buffer(result);
                            }
                            GraphStep::Reduction(reduction_op) => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "Reduction requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let result = reduction_op.execute(current_buf);
                                // The op's declared output domain (the same
                                // authority the planner reads) decides scalar
                                // vs buffer — not the result's shape, which is
                                // also [1] for an axis reduction of a 1-D
                                // buffer (a buffer-domain output).
                                current_output = if reduction_op.output_domain()
                                    == view_buffer::ops::Domain::Scalar
                                {
                                    let v = result.scalar_f64().ok_or_else(|| {
                                        format!(
                                            "Scalar reduction produced non-scalar shape {:?}",
                                            result.shape()
                                        )
                                    })?;
                                    NodeOutput::Scalar(v)
                                } else {
                                    NodeOutput::from_buffer(result)
                                };
                            }
                            GraphStep::Histogram(histogram_op) => {
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
                            GraphStep::ExtractShape => {
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
                            GraphStep::LabelReduce {
                                contours_col,
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
                                    self.name_to_slot.get(contours_col).ok_or_else(|| {
                                        format!(
                                            "LabelReduce contour column '{contours_col}' not found in expression inputs"
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
                                let scores = score_contours_on_buffer(
                                    current_buf,
                                    &contours,
                                    *reduction,
                                    *region_mode,
                                )?;
                                current_output = NodeOutput::from_vector(scores);
                            }
                            GraphStep::ChannelMerge { others } => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = current_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "ChannelMerge requires Buffer, got {:?}",
                                        current_output.domain()
                                    )
                                })?;
                                let mut all_bufs: Vec<&ViewBuffer> = vec![current_buf];
                                for other_id in others {
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
                            // Fusable single-buffer engine ops accumulate and
                            // run as one ViewExpr chain at the next flush.
                            GraphStep::Buffer(dto) => {
                                pending_buffer_ops.push(dto.clone());
                            }
                        }
                    }
                    current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                    node_outputs.insert(node_id.clone(), current_output);
                }
            }
        }
        Ok(())
    }

    /// Concurrently fetch every remote `file_path` source in this batch.
    ///
    /// Per-call (per morsel) and derived only from this batch's path values —
    /// nothing here is cached on the compiled graph. Distinct paths are
    /// fetched once; wrong-dtype columns are skipped so the row loop reports
    /// them with its usual error message.
    fn prefetch_remote_sources(
        &self,
        inputs: &[Series],
    ) -> HashMap<String, HashMap<String, Result<Vec<u8>, String>>> {
        let mut prefetched = HashMap::new();
        for (node_id, node) in &self.graph.nodes {
            if node.source.format != "file_path" {
                continue;
            }
            let Some(col_idx) = self.graph.column_bindings.get(node_id) else {
                continue;
            };
            let Some(series) = inputs.get(*col_idx) else {
                continue;
            };
            let Ok(ca) = series.str() else {
                continue;
            };
            let remote: Vec<String> = ca
                .iter()
                .flatten()
                .filter(|p| crate::cloud::is_remote_path(p))
                .map(str::to_string)
                .collect();
            if remote.is_empty() {
                continue;
            }
            prefetched.insert(
                node_id.clone(),
                crate::cloud::read_files_concurrent(
                    &remote,
                    self.cloud_options.get(node_id),
                    PREFETCH_CONCURRENCY,
                ),
            );
        }
        prefetched
    }
}

/// Source formats the executor can decode. Kept in sync with the row loop's
/// source dispatch and `decode_source`.
const KNOWN_SOURCE_FORMATS: &[&str] = &[
    "array",
    "blob",
    "contour",
    "file_path",
    "image_bytes",
    "list",
    "raw",
];

/// Validate the structural invariants the executor relies on, at compile time.
///
/// Each of these used to fail late and badly: a typo'd output node silently
/// produced all-null rows, a non-root node without upstream panicked at
/// `upstream[0]`, and unknown source formats surfaced as per-row decode
/// errors. Compile-time rejection gives one clear error instead.
fn validate_graph_structure(graph: &UnifiedGraph) -> PolarsResult<()> {
    for (node_id, node) in &graph.nodes {
        if !KNOWN_SOURCE_FORMATS.contains(&node.source.format.as_str()) {
            polars_bail!(ComputeError:
                "Node '{}': unknown source format '{}' (expected one of {:?})",
                node_id, node.source.format, KNOWN_SOURCE_FORMATS
            );
        }
        if !graph.column_bindings.contains_key(node_id) && node.upstream.is_empty() {
            polars_bail!(ComputeError:
                "Node '{}' has neither an input column binding nor an upstream node",
                node_id
            );
        }
        for upstream_id in &node.upstream {
            if !graph.nodes.contains_key(upstream_id) {
                polars_bail!(ComputeError:
                    "Node '{}' references unknown upstream node '{}'",
                    node_id, upstream_id
                );
            }
        }
    }
    for (alias, spec) in &graph.outputs {
        if !graph.nodes.contains_key(&spec.node) {
            polars_bail!(ComputeError:
                "Output '{}' references unknown node '{}'",
                alias, spec.node
            );
        }
        match spec.expected_encoding.as_deref() {
            None | Some("histogram_buckets") => {}
            Some(other) => polars_bail!(ComputeError:
                "Output '{}': unknown expected_encoding '{}' (expected 'histogram_buckets')",
                alias, other
            ),
        }
    }
    Ok(())
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

/// Validate that a buffer output's dtype matches the planned expected_dtype
/// (inferred by the Python planner from the op's view-buffer contract).
/// "auto" is resolved elsewhere and skipped here.
fn validate_output_dtype(
    alias: &str,
    spec: &OutputSpec,
    output: &NodeOutput,
) -> Result<(), String> {
    if spec.expected_dtype != "auto" {
        if let Some(buf) = output.as_buffer() {
            if let Ok(expected) = super::decode::parse_dtype_str(&spec.expected_dtype) {
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
    Ok(())
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

    /// Pin the live decode path for `blob` and `raw` sources: both are decoded
    /// by the zero-copy branch in `run_row_nodes` (`decode_binary_zero_copy`),
    /// NOT by `execute::decode_source` — its blob/raw arms are deliberately
    /// absent. This end-to-end test must keep passing when those dead arms are
    /// deleted.
    #[test]
    fn blob_and_raw_sources_decode_via_zero_copy() {
        // blob: f32 buffer round-trips through source(blob) → relu → sink(blob).
        let buf = ViewBuffer::from_vec_with_shape(vec![1.0f32, -2.0, 3.0], vec![3]);
        let input = Series::new("b".into(), &[buf.to_blob()]);
        let compiled = CompiledGraph::compile(SIMPLE_GRAPH, &[]).unwrap();
        let out = compiled.execute(&[input]).unwrap();
        let out_bytes = out.binary().unwrap().get(0).unwrap();
        let decoded = ViewBuffer::from_blob(out_bytes).unwrap();
        assert_eq!(decoded.as_slice::<f32>(), &[1.0, 0.0, 3.0]);

        // raw: u8 bytes with an explicit dtype decode as a 1-D u8 buffer
        // (relu then promotes to f32 per the scalar-op dtype contract).
        const RAW_GRAPH: &str = r#"{
            "nodes": {
                "n0": {
                    "source": {"format": "raw", "dtype": "u8"},
                    "ops": [{"op": "relu"}]
                }
            },
            "outputs": {
                "_output": {"node": "n0", "sink": {"format": "blob"}}
            },
            "column_bindings": {"n0": 0}
        }"#;
        let input = Series::new("r".into(), &[vec![1u8, 2, 3]]);
        let compiled = CompiledGraph::compile(RAW_GRAPH, &[]).unwrap();
        let out = compiled.execute(&[input]).unwrap();
        let out_bytes = out.binary().unwrap().get(0).unwrap();
        let decoded = ViewBuffer::from_blob(out_bytes).unwrap();
        assert_eq!(decoded.as_slice::<f32>(), &[1.0, 2.0, 3.0]);
    }

    /// Completeness guard + execution coverage: every `GraphStep` variant
    /// executes end-to-end through `CompiledGraph::execute`. The exhaustive
    /// match in `assert_step_covered` makes adding a variant a compile error
    /// until it is acknowledged (and given a graph below).
    fn assert_step_covered(step: &GraphStep) {
        match step {
            GraphStep::Buffer(_)
            | GraphStep::Binary { .. }
            | GraphStep::ApplyMask { .. }
            | GraphStep::ChannelMerge { .. }
            | GraphStep::Geometry(_)
            | GraphStep::Reduction(_)
            | GraphStep::Histogram(_)
            | GraphStep::ExtractShape
            | GraphStep::LabelReduce { .. } => (),
        }
    }

    fn exec(graph: &str, names: &[String], inputs: &[Series]) -> Series {
        CompiledGraph::compile(graph, names)
            .expect("graph must compile")
            .execute(inputs)
            .expect("graph must execute")
    }

    fn mask_blob() -> Vec<u8> {
        // 4x4 single-channel u8 with a bright 2x2 block.
        let mut data = vec![0u8; 16];
        for y in 1..3 {
            for x in 1..3 {
                data[y * 4 + x] = 255;
            }
        }
        ViewBuffer::from_vec_with_shape(data, vec![4, 4, 1]).to_blob()
    }

    #[test]
    fn every_graph_step_variant_executes() {
        let f32_blob =
            ViewBuffer::from_vec_with_shape(vec![1.0f32, -2.0, 3.0, 4.0], vec![2, 2]).to_blob();
        let no_names: Vec<String> = vec![];

        // Buffer (relu executes via the fused ViewExpr run).
        let step = resolve_op(
            &OpSpec {
                op: "relu".into(),
                params: HashMap::new(),
            },
            0,
            &ParamCtx::empty(),
        )
        .unwrap();
        assert_step_covered(&step);
        let out = exec(
            SIMPLE_GRAPH,
            &no_names,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        assert_eq!(out.null_count(), 0);

        // Binary: n1 = n0 + n0.
        let out = exec(
            r#"{
                "nodes": {
                    "n0": {"source": {"format": "blob"}},
                    "n1": {"source": {"format": "blob"}, "upstream": ["n0"],
                           "ops": [{"op": "add", "other_node": {"type": "literal", "value": "n0"}}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let doubled = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(doubled.as_slice::<f32>(), &[2.0, -4.0, 6.0, 8.0]);

        // ApplyMask: mask a buffer with itself (u8 mask semantics).
        let out = exec(
            r#"{
                "nodes": {
                    "n0": {"source": {"format": "blob"}},
                    "n1": {"source": {"format": "blob"}, "upstream": ["n0"],
                           "ops": [{"op": "apply_mask", "other_node": {"type": "literal", "value": "n0"}}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        assert_eq!(out.null_count(), 0);

        // ChannelMerge: [2,2] + [2,2] -> [2,2,2].
        let single = ViewBuffer::from_vec_with_shape(vec![1u8, 2, 3, 4], vec![2, 2]).to_blob();
        let out = exec(
            r#"{
                "nodes": {
                    "n0": {"source": {"format": "blob"}},
                    "n1": {"source": {"format": "blob"}, "upstream": ["n0"],
                           "ops": [{"op": "channel_merge", "other_nodes": {"type": "literal", "value": ["n0"]}}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), &[single])],
        );
        let merged = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(merged.shape(), &[2, 2, 2]);

        // Geometry: extract_contours from a binary mask (buffer -> contour).
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "extract_contours"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}, "expected_domain": "contour"}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        assert_eq!(out.null_count(), 0);

        // Reduction: global sum -> scalar.
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "reduce_sum"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}, "expected_domain": "scalar"}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        assert_eq!(out.f64().unwrap().get(0), Some(6.0));

        // Histogram: counts buffer.
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "histogram",
                                           "bins": {"type": "literal", "value": 4},
                                           "closed": {"type": "literal", "value": "left"},
                                           "output": {"type": "literal", "value": "counts"}}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let counts = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(counts.as_slice::<u64>().iter().sum::<u64>(), 4);

        // ExtractShape: dimension vector.
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "extract_shape"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}, "expected_domain": "vector"}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        assert_eq!(out.null_count(), 0);

        // LabelReduce: score one contour region from an expression column.
        let contour = view_buffer::geometry::Contour::from_tuples(&[
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
        ]);
        let contour_av = crate::contour::contour_to_anyvalue(&contour);
        let inner = Series::from_any_values("".into(), &[contour_av], false).unwrap();
        let row = AnyValue::List(inner);
        let cont_col = Series::from_any_values("cont".into(), &[row], false).unwrap();
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "label_reduce",
                                           "contours": {"type": "expr", "col": "cont"},
                                           "reduction": {"type": "literal", "value": "max"}}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}, "expected_domain": "vector"}},
                "column_bindings": {"n0": 0}
            }"#,
            &["cont".to_string()],
            &[Series::new("b".into(), &[mask_blob()]), cont_col],
        );
        assert_eq!(out.null_count(), 0);
    }

    /// An axis reduction over a 1-D buffer yields shape [1] but its declared
    /// domain is Buffer — the executor must follow the op's output_domain()
    /// (the planning authority), not sniff the [1] shape into a scalar.
    #[test]
    fn axis_reduction_of_1d_buffer_stays_buffer() {
        let blob = ViewBuffer::from_vec_with_shape(vec![1.0f32, 5.0, 3.0], vec![3]).to_blob();
        let no_names: Vec<String> = vec![];
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "reduce_max",
                                           "axis": {"type": "literal", "value": 0}}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), &[blob])],
        );
        let buf = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(buf.shape(), &[1], "axis reduction must stay a buffer");
        assert_eq!(buf.as_slice::<f32>(), &[5.0]);
    }

    /// Percentile is a global scalar reduction end-to-end (regression for the
    /// old DTO match that mislabeled it Buffer).
    #[test]
    fn percentile_reduction_is_scalar() {
        let blob = ViewBuffer::from_vec_with_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).to_blob();
        let no_names: Vec<String> = vec![];
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "reduce_percentile",
                                           "q": {"type": "literal", "value": 50.0}}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}, "expected_domain": "scalar"}},
                "column_bindings": {"n0": 0}
            }"#,
            &no_names,
            &[Series::new("b".into(), &[blob])],
        );
        assert_eq!(out.dtype(), &DataType::Float64);
        assert!(out.f64().unwrap().get(0).is_some());
    }

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
    // --- Compile-time structural validation ---
    //
    // Each case used to fail late (per-row error), silently (all-null
    // output), or with a panic. They must now be clear compile errors.

    fn compile_err(graph_json: &str) -> String {
        CompiledGraph::compile(graph_json, &[])
            .err()
            .expect("malformed graph must fail to compile")
            .to_string()
    }

    #[test]
    fn output_referencing_unknown_node_is_a_compile_error() {
        // Previously: silent all-null output column.
        let err = compile_err(
            r#"{
            "nodes": {"n0": {"source": {"format": "blob"}}},
            "outputs": {"_output": {"node": "nope", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(err.contains("unknown node 'nope'"), "{err}");
    }

    #[test]
    fn node_without_binding_or_upstream_is_a_compile_error() {
        // Previously: panic at `upstream[0]` in the row loop.
        let err = compile_err(
            r#"{
            "nodes": {
                "n0": {"source": {"format": "blob"}},
                "orphan": {"source": {"format": "blob"}}
            },
            "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(
            err.contains("neither an input column binding nor an upstream"),
            "{err}"
        );
    }

    #[test]
    fn unknown_upstream_reference_is_a_compile_error() {
        let err = compile_err(
            r#"{
            "nodes": {
                "n0": {"source": {"format": "blob"}},
                "n1": {"source": {"format": "blob"}, "upstream": ["ghost"]}
            },
            "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(err.contains("unknown upstream node 'ghost'"), "{err}");
    }

    #[test]
    fn unknown_source_format_is_a_compile_error() {
        let err = compile_err(
            r#"{
            "nodes": {"n0": {"source": {"format": "carrier_pigeon"}}},
            "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(
            err.contains("unknown source format 'carrier_pigeon'"),
            "{err}"
        );
    }

    #[test]
    fn unknown_expected_encoding_is_a_compile_error() {
        // Previously: silently ignored (treated as plain encoding).
        let err = compile_err(
            r#"{
            "nodes": {"n0": {"source": {"format": "blob"}}},
            "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}, "expected_encoding": "morse"}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(err.contains("unknown expected_encoding 'morse'"), "{err}");
    }
}
