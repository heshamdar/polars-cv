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

// Import geometry operations from view-buffer
use view_buffer::geometry::{
    contour::{Contour, Point, Winding},
    measures, pairwise, predicates, transforms,
};

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

// ============================================================================
// Contour Plugin Functions
// ============================================================================

/// Kwargs for contour operations with optional parameters.
#[derive(Debug, Deserialize)]
pub struct ContourKwargs {
    /// Whether to compute signed area (for area operation).
    #[serde(default)]
    pub signed: bool,
    /// Reference width for coordinate operations.
    #[serde(default)]
    pub ref_width: Option<f64>,
    /// Reference height for coordinate operations.
    #[serde(default)]
    pub ref_height: Option<f64>,
    /// X offset for translation.
    #[serde(default)]
    pub dx: Option<f64>,
    /// Y offset for translation.
    #[serde(default)]
    pub dy: Option<f64>,
    /// X scale factor.
    #[serde(default)]
    pub sx: Option<f64>,
    /// Y scale factor.
    #[serde(default)]
    pub sy: Option<f64>,
    /// Tolerance for simplification.
    #[serde(default)]
    pub tolerance: Option<f64>,
    /// Winding direction for ensure_winding.
    #[serde(default)]
    pub direction: Option<String>,
    /// X coordinate for point tests.
    #[serde(default)]
    pub x: Option<f64>,
    /// Y coordinate for point tests.
    #[serde(default)]
    pub y: Option<f64>,
}

/// Helper function to parse a contour from a Polars Struct value.
/// 
/// Contours are stored as a struct column matching the schema:
/// {exterior: List[{x: f64, y: f64}], holes: List[List[{x: f64, y: f64}]]}
fn parse_contour(value: &AnyValue) -> PolarsResult<Contour> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();
            
            // Find the exterior field
            for (i, field) in fields.iter().enumerate() {
                if field.name().as_str() == "exterior" || field.name().as_str() == "points" {
                    if let Some(AnyValue::List(series)) = values.get(i) {
                        let points = extract_points_from_series(series)?;
                        return Ok(Contour::new(points));
                    }
                }
            }
            
            // If no named field found, try to use the first list field
            for av in values.iter() {
                if let AnyValue::List(series) = av {
                    let points = extract_points_from_series(series)?;
                    return Ok(Contour::new(points));
                }
            }
            
            Err(polars_err!(ComputeError: "Contour struct missing exterior/points field"))
        }
        AnyValue::List(series) => {
            // Direct list of points (simpler format)
            let points = extract_points_from_series(series)?;
            Ok(Contour::new(points))
        }
        _ => Err(polars_err!(ComputeError: "Expected Struct or List for contour, got {:?}", value)),
    }
}

/// Extract points from a Series of point structs.
fn extract_points_from_series(series: &Series) -> PolarsResult<Vec<Point>> {
    let len = series.len();
    let mut points = Vec::with_capacity(len);
    
    // Try to get the struct columns directly
    if let Ok(struct_ca) = series.struct_() {
        // Get x and y columns from the struct
        let x_col = struct_ca.field_by_name("x")
            .or_else(|_| struct_ca.field_by_name("X"))
            .map_err(|_| polars_err!(ComputeError: "Point struct missing 'x' field"))?;
        let y_col = struct_ca.field_by_name("y")
            .or_else(|_| struct_ca.field_by_name("Y"))
            .map_err(|_| polars_err!(ComputeError: "Point struct missing 'y' field"))?;
        
        let x_ca = x_col.f64().map_err(|_| polars_err!(ComputeError: "x field must be f64"))?;
        let y_ca = y_col.f64().map_err(|_| polars_err!(ComputeError: "y field must be f64"))?;
        
        for i in 0..len {
            let x = x_ca.get(i).unwrap_or(0.0);
            let y = y_ca.get(i).unwrap_or(0.0);
            points.push(Point::new(x, y));
        }
    } else {
        // Fallback: iterate through values
        for i in 0..len {
            let value = series.get(i)?;
            match value {
                AnyValue::StructOwned(boxed) => {
                    let (values, _) = boxed.as_ref();
                    let x = values.first().and_then(|v| v.try_extract::<f64>().ok()).unwrap_or(0.0);
                    let y = values.get(1).and_then(|v| v.try_extract::<f64>().ok()).unwrap_or(0.0);
                    points.push(Point::new(x, y));
                }
                _ => {
                    return Err(polars_err!(ComputeError: "Expected Struct for point"));
                }
            }
        }
    }
    
    Ok(points)
}

/// Compute contour area.
#[polars_expr(output_type=Float64)]
fn contour_area(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut results = Vec::with_capacity(len);
    
    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let area_val = measures::area(&contour, kwargs.signed);
            results.push(Some(area_val));
        }
    }
    
    Ok(Float64Chunked::from_iter_options(series.name().clone(), results.into_iter()).into_series())
}

/// Compute contour perimeter.
#[polars_expr(output_type=Float64)]
fn contour_perimeter(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut results = Vec::with_capacity(len);
    
    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let perimeter = measures::perimeter(&contour);
            results.push(Some(perimeter));
        }
    }
    
    Ok(Float64Chunked::from_iter_options(series.name().clone(), results.into_iter()).into_series())
}

/// Compute winding direction.
#[polars_expr(output_type=String)]
fn contour_winding(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<&str>> = Vec::with_capacity(len);
    
    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let winding = measures::contour_winding(&contour);
            results.push(Some(match winding {
                Winding::CounterClockwise => "ccw",
                Winding::Clockwise => "cw",
            }));
        }
    }
    
    Ok(StringChunked::from_iter_options(series.name().clone(), results.into_iter()).into_series())
}

/// Check if contour is convex.
#[polars_expr(output_type=Boolean)]
fn contour_is_convex(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut results = Vec::with_capacity(len);
    
    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let is_convex = predicates::contour_is_convex(&contour);
            results.push(Some(is_convex));
        }
    }
    
    Ok(BooleanChunked::from_iter_options(series.name().clone(), results.into_iter()).into_series())
}

/// Compute IoU between two contours.
#[polars_expr(output_type=Float64)]
fn contour_iou(inputs: &[Series]) -> PolarsResult<Series> {
    let series_a = &inputs[0];
    let series_b = &inputs[1];
    let len = series_a.len();
    let mut results = Vec::with_capacity(len);
    
    for i in 0..len {
        let value_a = series_a.get(i)?;
        let value_b = series_b.get(i)?;
        
        if value_a.is_null() || value_b.is_null() {
            results.push(None);
        } else {
            let contour_a = parse_contour(&value_a)?;
            let contour_b = parse_contour(&value_b)?;
            let iou_val = pairwise::iou(&contour_a, &contour_b);
            results.push(Some(iou_val));
        }
    }
    
    Ok(Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter()).into_series())
}

/// Compute Dice coefficient between two contours.
#[polars_expr(output_type=Float64)]
fn contour_dice(inputs: &[Series]) -> PolarsResult<Series> {
    let series_a = &inputs[0];
    let series_b = &inputs[1];
    let len = series_a.len();
    let mut results = Vec::with_capacity(len);
    
    for i in 0..len {
        let value_a = series_a.get(i)?;
        let value_b = series_b.get(i)?;
        
        if value_a.is_null() || value_b.is_null() {
            results.push(None);
        } else {
            let contour_a = parse_contour(&value_a)?;
            let contour_b = parse_contour(&value_b)?;
            let dice_val = pairwise::dice(&contour_a, &contour_b);
            results.push(Some(dice_val));
        }
    }
    
    Ok(Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter()).into_series())
}

/// Compute Hausdorff distance between two contours.
#[polars_expr(output_type=Float64)]
fn contour_hausdorff(inputs: &[Series]) -> PolarsResult<Series> {
    let series_a = &inputs[0];
    let series_b = &inputs[1];
    let len = series_a.len();
    let mut results = Vec::with_capacity(len);
    
    for i in 0..len {
        let value_a = series_a.get(i)?;
        let value_b = series_b.get(i)?;
        
        if value_a.is_null() || value_b.is_null() {
            results.push(None);
        } else {
            let contour_a = parse_contour(&value_a)?;
            let contour_b = parse_contour(&value_b)?;
            let hausdorff = pairwise::hausdorff_distance(&contour_a, &contour_b);
            results.push(Some(hausdorff));
        }
    }
    
    Ok(Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter()).into_series())
}

/// Translate contour by offset.
fn contour_translate_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if let Some(field) = input_fields.first() {
        Ok(field.clone())
    } else {
        Ok(Field::new(PlSmallStr::from_static("output"), DataType::Unknown(UnknownKind::Any)))
    }
}

#[polars_expr(output_type_func=contour_translate_output_type)]
fn contour_translate(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let dx = kwargs.dx.unwrap_or(0.0);
    let dy = kwargs.dy.unwrap_or(0.0);
    
    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<AnyValue> = Vec::with_capacity(len);
    
    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(AnyValue::Null);
        } else {
            let contour = parse_contour(&value)?;
            let translated = transforms::translate(&contour, dx, dy);
            // For now, return the original value - proper serialization would need schema work
            let _ = translated;
            results.push(value.clone().into_static());
        }
    }
    
    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}

/// Scale contour.
fn contour_scale_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if let Some(field) = input_fields.first() {
        Ok(field.clone())
    } else {
        Ok(Field::new(PlSmallStr::from_static("output"), DataType::Unknown(UnknownKind::Any)))
    }
}

#[polars_expr(output_type_func=contour_scale_output_type)]
fn contour_scale(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let sx = kwargs.sx.unwrap_or(1.0);
    let sy = kwargs.sy.unwrap_or(1.0);
    
    let series = &inputs[0];
    let len = series.len();
    
    for i in 0..len {
        let value = series.get(i)?;
        if !value.is_null() {
            let contour = parse_contour(&value)?;
            let _scaled = transforms::scale(&contour, sx, sy, view_buffer::geometry::ops::ScaleOrigin::Centroid);
        }
    }
    
    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}

/// Simplify contour.
fn contour_simplify_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if let Some(field) = input_fields.first() {
        Ok(field.clone())
    } else {
        Ok(Field::new(PlSmallStr::from_static("output"), DataType::Unknown(UnknownKind::Any)))
    }
}

#[polars_expr(output_type_func=contour_simplify_output_type)]
fn contour_simplify(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let tolerance = kwargs.tolerance.unwrap_or(1.0);
    
    let series = &inputs[0];
    let len = series.len();
    
    for i in 0..len {
        let value = series.get(i)?;
        if !value.is_null() {
            let contour = parse_contour(&value)?;
            let _simplified = transforms::simplify(&contour, tolerance);
        }
    }
    
    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}

/// Flip contour (reverse winding).
fn contour_flip_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if let Some(field) = input_fields.first() {
        Ok(field.clone())
    } else {
        Ok(Field::new(PlSmallStr::from_static("output"), DataType::Unknown(UnknownKind::Any)))
    }
}

#[polars_expr(output_type_func=contour_flip_output_type)]
fn contour_flip(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    
    for i in 0..len {
        let value = series.get(i)?;
        if !value.is_null() {
            let contour = parse_contour(&value)?;
            let _flipped = transforms::flip(&contour);
        }
    }
    
    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}

/// Compute convex hull.
fn contour_convex_hull_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if let Some(field) = input_fields.first() {
        Ok(field.clone())
    } else {
        Ok(Field::new(PlSmallStr::from_static("output"), DataType::Unknown(UnknownKind::Any)))
    }
}

#[polars_expr(output_type_func=contour_convex_hull_output_type)]
fn contour_convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    
    for i in 0..len {
        let value = series.get(i)?;
        if !value.is_null() {
            let contour = parse_contour(&value)?;
            let _hull = transforms::convex_hull(&contour);
        }
    }
    
    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}
