//! polars-cv: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod cloud;
mod contour;
mod execute;
mod graph;
mod output;
mod params;
mod pipeline;

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
    Ok(())
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
///     >>> # Tile all images regardless of size
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

    // Resolve "auto" dtype/ndim from input series (mirrors unified_output_dtype logic)
    if !inputs.is_empty() {
        let (leaf_dtype, ndim) = peel_nesting(inputs[0].dtype());
        let inferred_dtype_str = polars_dtype_to_str(&leaf_dtype);

        for spec in graph.outputs.values_mut() {
            if spec.expected_dtype == "auto" {
                spec.expected_dtype = inferred_dtype_str.to_string();
            }
            if spec.expected_ndim.is_none() && ndim > 0 {
                spec.expected_ndim = Some(ndim);
            }
        }
    }

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

    // If we have input fields, extract the inner dtype and nesting depth
    // so we can resolve "auto" sentinels in output specs.
    if !input_fields.is_empty() {
        let (leaf_dtype, ndim) = peel_nesting(input_fields[0].dtype());
        let inferred_dtype_str = polars_dtype_to_str(&leaf_dtype);

        for spec in graph.outputs.values_mut() {
            if spec.expected_dtype == "auto" {
                spec.expected_dtype = inferred_dtype_str.to_string();
            }
            if spec.expected_ndim.is_none() && ndim > 0 {
                spec.expected_ndim = Some(ndim);
            }
        }
    }

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
