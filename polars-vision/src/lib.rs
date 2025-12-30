//! polars-vision: A Polars plugin for vision/array operations.
//!
//! This crate provides expression functions for applying image and array
//! processing pipelines to Polars DataFrame columns, powered by view-buffer.

mod execute;
mod graph;
mod params;
mod pipeline;

use polars::prelude::*;
use polars_arrow::array::{ListArray, PrimitiveArray, StructArray as ArrowStructArray};
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
use crate::graph::{MultiPipelineGraph, PipelineGraph, UnifiedGraph};
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
    let expr_columns: std::collections::HashMap<String, &Series> = std::collections::HashMap::new();

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
    let expr_columns: std::collections::HashMap<String, &Series> = std::collections::HashMap::new();

    // Execute the graph and return Struct column
    graph.execute(inputs, &expr_columns)
}

/// Unified pipeline graph execution for single output.
///
/// This function handles single-output graph execution using the unified
/// graph format. It always returns a Binary column.
///
/// Use this when you know the graph has only one output ("_output" key).
#[polars_expr(output_type=Binary)]
fn vb_graph(inputs: &[Series], kwargs: GraphKwargs) -> PolarsResult<Series> {
    // Parse the unified graph specification
    let graph = UnifiedGraph::from_json(&kwargs.graph_json)?;

    // Build expression columns map (empty for now, may be used for dynamic params)
    let expr_columns: std::collections::HashMap<String, &Series> = std::collections::HashMap::new();

    // Execute the graph - should return Binary for single output
    graph.execute(inputs, &expr_columns)
}

/// Compute the output dtype for multi-output unified graph.
fn unified_multi_output_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    // For multi-output, we return a Struct with Binary fields
    // The actual field names are determined at runtime based on the graph
    let name = if !input_fields.is_empty() {
        input_fields[0].name().clone()
    } else {
        PlSmallStr::from_static("output")
    };

    // Use Unknown(Any) for dynamic struct type
    Ok(Field::new(name, DataType::Unknown(UnknownKind::Any)))
}

/// Unified pipeline graph execution for multiple outputs.
///
/// This function handles multi-output graph execution using the unified
/// graph format. It returns a Struct column with named Binary fields.
///
/// Use this when the graph has multiple outputs.
#[polars_expr(output_type_func=unified_multi_output_dtype)]
fn vb_graph_multi(inputs: &[Series], kwargs: GraphKwargs) -> PolarsResult<Series> {
    // Parse the unified graph specification
    let graph = UnifiedGraph::from_json(&kwargs.graph_json)?;

    // Build expression columns map (empty for now, may be used for dynamic params)
    let expr_columns: std::collections::HashMap<String, &Series> = std::collections::HashMap::new();

    // Execute the graph - should return Struct for multi-output
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
        // Handle AnyValue::Struct (non-owned variant with row index and array reference)
        AnyValue::Struct(row_idx, struct_array, fields) => {
            // Find the exterior field in the struct array
            for (i, field) in fields.iter().enumerate() {
                if field.name().as_str() == "exterior" || field.name().as_str() == "points" {
                    // Get the column from the struct array
                    let column = struct_array.values()[i].clone();
                    // Get the value at the row index
                    let list_arr = column.as_any().downcast_ref::<ListArray<i64>>();
                    if let Some(list_arr) = list_arr {
                        // Extract the points from the list at this row
                        let offsets = list_arr.offsets();
                        let start = offsets[*row_idx] as usize;
                        let end = offsets[*row_idx + 1] as usize;
                        let values_arr = list_arr.values();

                        // The values should be a struct array of points
                        if let Some(struct_arr) =
                            values_arr.as_any().downcast_ref::<ArrowStructArray>()
                        {
                            let points = extract_points_from_struct_array(struct_arr, start, end)?;
                            return Ok(Contour::new(points));
                        }
                    }
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

/// Extract points from a StructArray slice (for use with AnyValue::Struct variant).
fn extract_points_from_struct_array(
    struct_arr: &ArrowStructArray,
    start: usize,
    end: usize,
) -> PolarsResult<Vec<Point>> {
    let mut points = Vec::with_capacity(end - start);

    // Get x and y arrays from the struct
    let values = struct_arr.values();
    if values.len() < 2 {
        return Err(polars_err!(ComputeError: "Point struct must have x and y fields"));
    }

    // Try to get x and y as Float64 arrays
    let x_arr = values[0].as_any().downcast_ref::<PrimitiveArray<f64>>();
    let y_arr = values[1].as_any().downcast_ref::<PrimitiveArray<f64>>();

    match (x_arr, y_arr) {
        (Some(x), Some(y)) => {
            for i in start..end {
                let x_val = x.get(i).unwrap_or(0.0);
                let y_val = y.get(i).unwrap_or(0.0);
                points.push(Point::new(x_val, y_val));
            }
            Ok(points)
        }
        _ => Err(polars_err!(ComputeError: "Point x/y fields must be Float64")),
    }
}

/// Extract points from a Series of point structs.
fn extract_points_from_series(series: &Series) -> PolarsResult<Vec<Point>> {
    let len = series.len();
    let mut points = Vec::with_capacity(len);

    // Try to get the struct columns directly
    if let Ok(struct_ca) = series.struct_() {
        // Get x and y columns from the struct
        let x_col = struct_ca
            .field_by_name("x")
            .or_else(|_| struct_ca.field_by_name("X"))
            .map_err(|_| polars_err!(ComputeError: "Point struct missing 'x' field"))?;
        let y_col = struct_ca
            .field_by_name("y")
            .or_else(|_| struct_ca.field_by_name("Y"))
            .map_err(|_| polars_err!(ComputeError: "Point struct missing 'y' field"))?;

        let x_ca = x_col
            .f64()
            .map_err(|_| polars_err!(ComputeError: "x field must be f64"))?;
        let y_ca = y_col
            .f64()
            .map_err(|_| polars_err!(ComputeError: "y field must be f64"))?;

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
                    let x = values
                        .first()
                        .and_then(|v| v.try_extract::<f64>().ok())
                        .unwrap_or(0.0);
                    let y = values
                        .get(1)
                        .and_then(|v| v.try_extract::<f64>().ok())
                        .unwrap_or(0.0);
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

    Ok(
        Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter())
            .into_series(),
    )
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

    Ok(
        Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter())
            .into_series(),
    )
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

    Ok(
        Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter())
            .into_series(),
    )
}

/// Translate contour by offset.
fn contour_translate_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if let Some(field) = input_fields.first() {
        Ok(field.clone())
    } else {
        Ok(Field::new(
            PlSmallStr::from_static("output"),
            DataType::Unknown(UnknownKind::Any),
        ))
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
        Ok(Field::new(
            PlSmallStr::from_static("output"),
            DataType::Unknown(UnknownKind::Any),
        ))
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
            let _scaled = transforms::scale(
                &contour,
                sx,
                sy,
                view_buffer::geometry::ops::ScaleOrigin::Centroid,
            );
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
        Ok(Field::new(
            PlSmallStr::from_static("output"),
            DataType::Unknown(UnknownKind::Any),
        ))
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
        Ok(Field::new(
            PlSmallStr::from_static("output"),
            DataType::Unknown(UnknownKind::Any),
        ))
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
        Ok(Field::new(
            PlSmallStr::from_static("output"),
            DataType::Unknown(UnknownKind::Any),
        ))
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

/// Compute contour centroid - returns a Struct with x and y fields.
fn contour_centroid_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new(PlSmallStr::from_static("x"), DataType::Float64),
        Field::new(PlSmallStr::from_static("y"), DataType::Float64),
    ];
    Ok(Field::new(
        PlSmallStr::from_static("centroid"),
        DataType::Struct(fields),
    ))
}

#[polars_expr(output_type_func=contour_centroid_output_type)]
fn contour_centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut x_results: Vec<Option<f64>> = Vec::with_capacity(len);
    let mut y_results: Vec<Option<f64>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let center = measures::centroid(&contour);
            x_results.push(Some(center.x));
            y_results.push(Some(center.y));
        }
    }

    // Build struct column
    let x_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("x"), x_results.into_iter())
            .into_series();
    let y_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("y"), y_results.into_iter())
            .into_series();

    StructChunked::from_series(
        PlSmallStr::from_static("centroid"),
        len,
        [x_col, y_col].iter(),
    )
    .map(|ca| ca.into_series())
}

/// Compute contour bounding box - returns a Struct with x, y, width, height fields.
fn contour_bbox_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new(PlSmallStr::from_static("x"), DataType::Float64),
        Field::new(PlSmallStr::from_static("y"), DataType::Float64),
        Field::new(PlSmallStr::from_static("width"), DataType::Float64),
        Field::new(PlSmallStr::from_static("height"), DataType::Float64),
    ];
    Ok(Field::new(
        PlSmallStr::from_static("bbox"),
        DataType::Struct(fields),
    ))
}

#[polars_expr(output_type_func=contour_bbox_output_type)]
fn contour_bbox(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut x_results: Vec<Option<f64>> = Vec::with_capacity(len);
    let mut y_results: Vec<Option<f64>> = Vec::with_capacity(len);
    let mut w_results: Vec<Option<f64>> = Vec::with_capacity(len);
    let mut h_results: Vec<Option<f64>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            x_results.push(None);
            y_results.push(None);
            w_results.push(None);
            h_results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            if let Some(bbox) = measures::bounding_box(&contour) {
                x_results.push(Some(bbox.x));
                y_results.push(Some(bbox.y));
                w_results.push(Some(bbox.width));
                h_results.push(Some(bbox.height));
            } else {
                x_results.push(None);
                y_results.push(None);
                w_results.push(None);
                h_results.push(None);
            }
        }
    }

    // Build struct column
    let x_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("x"), x_results.into_iter())
            .into_series();
    let y_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("y"), y_results.into_iter())
            .into_series();
    let w_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("width"), w_results.into_iter())
            .into_series();
    let h_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("height"), h_results.into_iter())
            .into_series();

    StructChunked::from_series(
        PlSmallStr::from_static("bbox"),
        len,
        [x_col, y_col, w_col, h_col].iter(),
    )
    .map(|ca| ca.into_series())
}

/// Normalize contour coordinates to [0, 1] range.
#[polars_expr(output_type_func=contour_translate_output_type)]
fn contour_normalize(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let ref_width = kwargs.ref_width.unwrap_or(1.0);
    let ref_height = kwargs.ref_height.unwrap_or(1.0);

    let series = &inputs[0];
    let len = series.len();

    for i in 0..len {
        let value = series.get(i)?;
        if !value.is_null() {
            let contour = parse_contour(&value)?;
            let _normalized = transforms::normalize(&contour, ref_width, ref_height);
        }
    }

    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}

/// Convert normalized coordinates to absolute pixel coordinates.
#[polars_expr(output_type_func=contour_translate_output_type)]
fn contour_to_absolute(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let ref_width = kwargs.ref_width.unwrap_or(1.0);
    let ref_height = kwargs.ref_height.unwrap_or(1.0);

    let series = &inputs[0];
    let len = series.len();

    for i in 0..len {
        let value = series.get(i)?;
        if !value.is_null() {
            let contour = parse_contour(&value)?;
            let _absolute = transforms::to_absolute(&contour, ref_width, ref_height);
        }
    }

    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}

/// Ensure contour has specified winding direction.
#[polars_expr(output_type_func=contour_translate_output_type)]
fn contour_ensure_winding(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let direction = match kwargs.direction.as_deref() {
        Some("cw") | Some("clockwise") => Winding::Clockwise,
        Some("ccw") | Some("counterclockwise") => Winding::CounterClockwise,
        _ => Winding::CounterClockwise, // Default to CCW
    };

    let series = &inputs[0];
    let len = series.len();

    for i in 0..len {
        let value = series.get(i)?;
        if !value.is_null() {
            let contour = parse_contour(&value)?;
            let _ensured = transforms::ensure_winding(&contour, direction);
        }
    }

    // For now, return the original series - proper transform output requires schema work
    Ok(series.clone())
}

/// Check if contour contains a specific point.
#[polars_expr(output_type=Boolean)]
fn contour_contains_point(inputs: &[Series]) -> PolarsResult<Series> {
    let contour_series = &inputs[0];
    let point_series = &inputs[1];
    let len = contour_series.len();
    let mut results: Vec<Option<bool>> = Vec::with_capacity(len);

    for i in 0..len {
        let contour_value = contour_series.get(i)?;
        let point_value = point_series.get(i)?;

        if contour_value.is_null() || point_value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&contour_value)?;
            // Parse point from struct
            let (x, y) = match &point_value {
                AnyValue::StructOwned(boxed) => {
                    let (values, _) = boxed.as_ref();
                    let x = values
                        .first()
                        .and_then(|v| v.try_extract::<f64>().ok())
                        .unwrap_or(0.0);
                    let y = values
                        .get(1)
                        .and_then(|v| v.try_extract::<f64>().ok())
                        .unwrap_or(0.0);
                    (x, y)
                }
                AnyValue::Struct(row_idx, struct_arr, _) => {
                    let values = struct_arr.values();
                    if values.len() >= 2 {
                        let x_arr = values[0].as_any().downcast_ref::<PrimitiveArray<f64>>();
                        let y_arr = values[1].as_any().downcast_ref::<PrimitiveArray<f64>>();
                        match (x_arr, y_arr) {
                            (Some(x), Some(y)) => (
                                x.get(*row_idx).unwrap_or(0.0),
                                y.get(*row_idx).unwrap_or(0.0),
                            ),
                            _ => (0.0, 0.0),
                        }
                    } else {
                        (0.0, 0.0)
                    }
                }
                _ => {
                    return Err(polars_err!(ComputeError: "Expected Struct for point"));
                }
            };
            let contains = predicates::contains_point(&contour, x, y);
            results.push(Some(contains));
        }
    }

    Ok(
        BooleanChunked::from_iter_options(contour_series.name().clone(), results.into_iter())
            .into_series(),
    )
}
