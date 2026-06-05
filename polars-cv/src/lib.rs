//! polars-cv: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod cloud;
mod contour;
mod execute;
mod graph;
mod image_metadata;
mod output;
mod params;
mod pipeline;
mod point;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Python module entry point for maturin.
/// The module name `_lib` must match pyproject.toml's `module-name = "polars_cv._lib"`.
#[pymodule]
#[pyo3(name = "_lib")]
fn polars_cv_lib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register tiling configuration functions
    m.add_function(wrap_pyfunction!(configure_tiling, m)?)?;
    m.add_function(wrap_pyfunction!(get_tiling_config, m)?)?;
    m.add_function(wrap_pyfunction!(op_dtype_rule, m)?)?;
    Ok(())
}

// ============================================================================
// Contract introspection (single-authority bridge for the Python schema layer)
// ============================================================================

/// Canonical short name for a view-buffer `DType`.
///
/// Matches the values of the Python `DType` enum so the two vocabularies line up.
fn dtype_short_name(dt: view_buffer::DType) -> &'static str {
    use view_buffer::DType;
    match dt {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F32 => "f32",
        DType::F64 => "f64",
    }
}

/// Canonical string for an output-dtype rule.
///
/// This is the shared vocabulary the Python contract-parity test compares
/// against. It mirrors the Python `DTypeEffect` values: `preserve`, `promote`,
/// `fixed:<dtype>`, `config:<dtype>`.
fn dtype_rule_name(rule: view_buffer::OutputDTypeRule) -> String {
    use view_buffer::OutputDTypeRule as R;
    match rule {
        R::PreserveInput => "preserve".to_string(),
        R::PromoteToFloat => "promote".to_string(),
        R::Fixed(d) => format!("fixed:{}", dtype_short_name(d)),
        R::Configurable(d) => format!("config:{}", dtype_short_name(d)),
        R::ForceF64 => "fixed:f64".to_string(),
        R::ForceI64 => "fixed:i64".to_string(),
        R::ForceU64 => "fixed:u64".to_string(),
        R::ForceU32 => "fixed:u32".to_string(),
    }
}

/// Return the canonical output-dtype rule for a single serialized op spec.
///
/// `op_json` is one `OpSpec` as produced by the Python builder
/// (`OpSpec.to_dict()`): `{"op": "<name>", "<param>": {..ParamValue..}, ...}`.
///
/// This exposes view-buffer's `ViewDto::output_dtype_rule()` — the single
/// authority for "what dtype does this op produce" — so the Python schema layer
/// can be checked against (and ultimately defer to) it instead of maintaining a
/// parallel dtype table that can drift.
#[pyfunction]
fn op_dtype_rule(op_json: &str) -> PyResult<String> {
    let op_spec: crate::pipeline::OpSpec = serde_json::from_str(op_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid op json: {e}")))?;
    let empty: std::collections::HashMap<String, &Series> = std::collections::HashMap::new();
    let dto = crate::execute::resolve_op(&op_spec, 0, &empty)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("resolve_op: {e}")))?;
    Ok(dtype_rule_name(dto.output_dtype_rule()))
}

// ============================================================================
// Tiling Configuration (Python-exposed)
// ============================================================================

/// Configure tiled execution for large image processing.
///
/// Tiled execution improves cache efficiency when processing large images
/// by dividing them into smaller tiles (default 256x256) that fit in CPU cache.
///
/// By default, tiling is enabled for images larger than 512 pixels in any dimension.
///
/// Args:
///     min_image_size: Minimum dimension (height or width) for tiling to activate.
///         - `None` or `0`: Disable tiling entirely
///         - Positive integer: Only tile images larger than this threshold
///         - Default when polars-cv loads: 512
///
/// Examples:
///     >>> import polars_cv
///     >>> # Disable tiling (process all images as single buffers)
///     >>> polars_cv.configure_tiling(None)
///     >>>
///     >>> # Only tile very large images (>2048 pixels)
///     >>> polars_cv.configure_tiling(2048)
///     >>>
///     >>> # Disable tiling (same as None)
///     >>> polars_cv.configure_tiling(0)
///
/// Note:
///     Tiling is transparent - results are identical whether tiling is on or off.
///     The only difference is memory access patterns and cache efficiency.
#[pyfunction]
#[pyo3(signature = (min_image_size=None))]
fn configure_tiling(min_image_size: Option<usize>) -> PyResult<()> {
    match min_image_size {
        None | Some(0) => {
            // Disable tiling
            view_buffer::set_tile_config(None);
        }
        Some(size) => {
            // Enable tiling with specified threshold
            view_buffer::configure_tiling(Some(size));
        }
    }
    Ok(())
}

/// Get the current tiling configuration.
///
/// Returns:
///     A dict with 'enabled', 'tile_size', and 'min_image_size' keys,
///     or None if tiling is disabled.
///
/// Examples:
///     >>> import polars_cv
///     >>> config = polars_cv.get_tiling_config()
///     >>> if config:
///     ...     print(f"Tiling enabled: tile_size={config['tile_size']}, min={config['min_image_size']}")
///     ... else:
///     ...     print("Tiling disabled")
#[pyfunction]
fn get_tiling_config(py: Python<'_>) -> PyResult<PyObject> {
    match view_buffer::get_tile_config() {
        Some(config) => {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("enabled", true)?;
            dict.set_item("tile_size", config.tile_size)?;
            dict.set_item("min_image_size", config.min_image_size)?;
            Ok(dict.into())
        }
        None => Ok(py.None()),
    }
}

use crate::graph::UnifiedGraph;

// ============================================================================
// Graph Execution
// ============================================================================

/// Kwargs for the graph-based pipeline function.
#[derive(Debug, Deserialize)]
pub struct GraphKwargs {
    /// JSON-serialized pipeline graph specification.
    pub graph_json: String,
    /// Names of expression columns (for resolving dynamic parameters).
    #[serde(default)]
    pub expr_column_names: Vec<String>,
}

/// Shared implementation for graph execution.
///
/// Handles both single-output and multi-output graphs uniformly.
fn execute_graph(inputs: &[Series], kwargs: &GraphKwargs) -> PolarsResult<Series> {
    // Parse the unified graph specification
    let mut graph = UnifiedGraph::from_json(&kwargs.graph_json)?;

    // Resolve "auto" dtype/ndim from the input column type. Shared with the
    // planning-time path (`unified_output_dtype`) so the two can never drift.
    resolve_auto_inputs(&mut graph, inputs.first().map(|s| s.dtype()));

    // Count the number of root node column bindings to determine where expression columns start
    let num_source_columns = graph.column_bindings.len().max(1);

    // Build expression columns map from inputs after the source columns
    let expr_columns: std::collections::HashMap<String, &Series> = kwargs
        .expr_column_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            inputs
                .get(num_source_columns + i)
                .map(|s| (name.clone(), s))
        })
        .collect();

    // Execute the graph
    graph.execute(inputs, &expr_columns)
}

/// Unified pipeline graph execution for single output.
///
/// This function handles single-output graph execution using the unified
/// graph format. Returns appropriately typed column based on domain/dtype.
///
/// Use this when you know the graph has only one output ("_output" key).
#[polars_expr(output_type_func_with_kwargs=unified_output_dtype)]
fn vb_graph(inputs: &[Series], kwargs: GraphKwargs) -> PolarsResult<Series> {
    execute_graph(inputs, &kwargs)
}

/// Compute the output dtype for unified graph (single or multi-output).
///
/// This function receives kwargs and parses the graph JSON to determine
/// the exact output type based on domain and dtype information:
/// - Single output: Returns appropriate typed column (Binary, Float64, List, etc.)
/// - Multi-output: Returns Struct with appropriately typed fields
fn unified_output_dtype(input_fields: &[Field], kwargs: GraphKwargs) -> PolarsResult<Field> {
    let name = if !input_fields.is_empty() {
        input_fields[0].name().clone()
    } else {
        PlSmallStr::from_static("output")
    };

    // Parse the graph JSON to extract output specifications
    let mut graph = UnifiedGraph::from_json(&kwargs.graph_json)?;

    // Resolve "auto" sentinels in output specs from the input column type.
    // Shared with the execution-time path (`execute_graph`) so the planned and
    // executed schema are computed by exactly one piece of logic.
    resolve_auto_inputs(&mut graph, input_fields.first().map(|f| f.dtype()));

    if graph.is_single_output() {
        // Single output mode - return typed field based on domain/sink/dtype
        let spec = graph
            .outputs
            .get("_output")
            .ok_or_else(|| polars_err!(ComputeError: "Single output graph missing _output key"))?;
        let dtype = crate::graph::dtype_for_output(spec)?;
        Ok(Field::new(name, dtype))
    } else {
        // Multi-output mode - build Struct with typed fields
        let mut output_names: Vec<&String> = graph.outputs.keys().collect();
        output_names.sort();

        let mut fields: Vec<Field> = Vec::with_capacity(output_names.len());
        for alias in output_names {
            let spec = graph.outputs.get(alias).unwrap();
            let dtype = crate::graph::dtype_for_output(spec)?;
            fields.push(Field::new(PlSmallStr::from(alias.as_str()), dtype));
        }

        Ok(Field::new(name, DataType::Struct(fields)))
    }
}

/// Resolve `"auto"` dtype and missing ndim on a graph's output specs from the
/// first input column's type.
///
/// This is the single implementation shared by both the planning-time
/// (`unified_output_dtype`) and execution-time (`execute_graph`) entry points,
/// so the inferred schema cannot diverge between the two.
///
/// Only the leaf type of List/Array sources is meaningful for dtype: for
/// Binary/String (image/file) sources the column type does not reflect the
/// decoded buffer dtype, so `"auto"` is left unresolved.
fn resolve_auto_inputs(graph: &mut UnifiedGraph, first_input_dtype: Option<&DataType>) {
    let Some(dt) = first_input_dtype else {
        return;
    };
    let (leaf_dtype, ndim) = peel_nesting(dt);
    let inferred_dtype_str = polars_dtype_to_str(&leaf_dtype);

    for spec in graph.outputs.values_mut() {
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
