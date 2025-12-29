//! polars-vision: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod execute;
mod graph;
mod params;
mod pipeline;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Python module entry point for maturin.
/// The module name `_lib` must match pyproject.toml's `module-name = "polars_vision._lib"`.
#[pymodule]
#[pyo3(name = "_lib")]
fn polars_vision_lib(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Plugin functions are registered via polars_expr, not here
    Ok(())
}

use crate::execute::execute_pipeline;
use crate::graph::{MultiPipelineGraph, PipelineGraph};
use crate::pipeline::PipelineSpec;

/// Kwargs passed from Python to the plugin function.
#[derive(Debug, Deserialize)]
pub struct PipelineKwargs {
    /// JSON-serialized pipeline specification.
    pub pipeline_json: String,
    /// Names of expression columns (for resolving dynamic parameters).
    #[serde(default)]
    pub expr_column_names: Vec<String>,
}

/// Apply a vision pipeline to a binary column.
///
/// This is the main entry point for the plugin. It receives:
/// - inputs[0]: The main data column (Binary)
/// - inputs[1..]: Expression columns referenced in the pipeline
///
/// The pipeline is deserialized from JSON and executed for each row.
///
/// Output type is always Binary for now - the sink format determines
/// how the bytes should be interpreted.
#[polars_expr(output_type=Binary)]
fn vb_pipeline(inputs: &[Series], kwargs: PipelineKwargs) -> PolarsResult<Series> {
    // Parse the pipeline specification
    let pipeline: PipelineSpec = serde_json::from_str(&kwargs.pipeline_json)
        .map_err(|e| polars_err!(ComputeError: "Failed to parse pipeline: {}", e))?;

    // Get the main data column
    let data_series = &inputs[0];

    // Build a map of expression column names to their series
    let expr_columns: std::collections::HashMap<String, &Series> = kwargs
        .expr_column_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| inputs.get(i + 1).map(|s| (name.clone(), s)))
        .collect();

    // Execute the pipeline
    execute_pipeline(data_series, &pipeline, &expr_columns)
}

/// Kwargs for the graph-based pipeline function.
#[derive(Debug, Deserialize)]
pub struct GraphKwargs {
    /// JSON-serialized pipeline graph specification.
    pub graph_json: String,
}

/// Apply a pipeline graph (DAG) to multiple binary columns.
///
/// This enables composable pipelines where multiple pipelines can be
/// fused into a single execution without intermediate serialization.
///
/// The graph is executed in topological order, with ViewBuffers passed
/// directly between nodes.
#[polars_expr(output_type=Binary)]
fn vb_pipeline_graph(inputs: &[Series], kwargs: GraphKwargs) -> PolarsResult<Series> {
    // Parse the graph specification
    let graph = PipelineGraph::from_json(&kwargs.graph_json)?;

    // Build expression columns map (empty for now, may be used for dynamic params)
    let expr_columns: std::collections::HashMap<String, &Series> =
        std::collections::HashMap::new();

    // Execute the graph
    graph.execute(inputs, &expr_columns)
}

/// Compute the output dtype for multi-output graph.
///
/// This function is called by the polars_expr macro to determine the output type.
/// It parses the graph JSON to extract output aliases and creates a Struct dtype.
fn multi_output_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    // For multi-output, we return a Struct with Binary fields
    // The actual field names are determined at runtime based on the graph
    // For now, we use a placeholder struct type - the actual execution will
    // return the correct struct with named fields

    // Default to returning a struct with the input field name
    let name = if !input_fields.is_empty() {
        input_fields[0].name().clone()
    } else {
        PlSmallStr::from_static("output")
    };

    // Create a placeholder struct type - the actual struct fields are dynamic
    // and determined by the graph JSON at execution time
    Ok(Field::new(name, DataType::Unknown(UnknownKind::Any)))
}

/// Apply a multi-output pipeline graph (DAG) to multiple binary columns.
///
/// Similar to `vb_pipeline_graph`, but returns a Struct column where each
/// field is a named output from the graph. This enables extracting multiple
/// intermediate results from a single pipeline execution.
///
/// The output is a Struct column where:
/// - Each field name corresponds to an alias defined in the graph
/// - Each field value is Binary data encoded in the specified format
#[polars_expr(output_type_func=multi_output_dtype)]
fn vb_pipeline_graph_multi(inputs: &[Series], kwargs: GraphKwargs) -> PolarsResult<Series> {
    // Parse the multi-output graph specification
    let graph = MultiPipelineGraph::from_json(&kwargs.graph_json)?;

    // Build expression columns map (empty for now, may be used for dynamic params)
    let expr_columns: std::collections::HashMap<String, &Series> =
        std::collections::HashMap::new();

    // Execute the graph and return Struct column
    graph.execute(inputs, &expr_columns)
}
