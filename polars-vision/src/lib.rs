//! polars-vision: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod execute;
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
