//! Contour plugin functions for polars-cv.
//!
//! This module provides Polars expression functions for contour geometry operations,
//! including measures (area, perimeter), predicates (is_convex, contains_point),
//! transforms (translate, scale, simplify), and pairwise comparisons (IoU, Dice).

use polars::prelude::*;
use polars_arrow::array::{ListArray, PrimitiveArray, StructArray as ArrowStructArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

// Import geometry operations from view-buffer
use view_buffer::geometry::{
    contour::{Contour, Point, Winding},
    measures, pairwise, predicates, transforms,
};

// ============================================================================
// Contour Serialization Helpers
// ============================================================================

/// Convert a Contour to a Polars AnyValue matching CONTOUR_SCHEMA.
///
/// The schema is:
/// - exterior: List[{x: Float64, y: Float64}]
/// - holes: List[List[{x: Float64, y: Float64}]]
/// - is_closed: Boolean
pub fn contour_to_anyvalue(contour: &Contour) -> AnyValue<'static> {
    // Build exterior points as list of structs
    let exterior_points: Vec<AnyValue> = contour
        .exterior
        .iter()
        .map(|p| {
            AnyValue::StructOwned(Box::new((
                vec![AnyValue::Float64(p.x), AnyValue::Float64(p.y)],
                vec![
                    Field::new(PlSmallStr::from_static("x"), DataType::Float64),
                    Field::new(PlSmallStr::from_static("y"), DataType::Float64),
                ],
            )))
        })
        .collect();

    // Build holes as list of list of structs
    let holes_list: Vec<AnyValue> = contour
        .holes
        .iter()
        .map(|hole| {
            let hole_points: Vec<AnyValue> = hole
                .iter()
                .map(|p| {
                    AnyValue::StructOwned(Box::new((
                        vec![AnyValue::Float64(p.x), AnyValue::Float64(p.y)],
                        vec![
                            Field::new(PlSmallStr::from_static("x"), DataType::Float64),
                            Field::new(PlSmallStr::from_static("y"), DataType::Float64),
                        ],
                    )))
                })
                .collect();
            // Create a Series from hole points for the inner list
            let point_schema = DataType::Struct(vec![
                Field::new(PlSmallStr::from_static("x"), DataType::Float64),
                Field::new(PlSmallStr::from_static("y"), DataType::Float64),
            ]);
            let hole_series = Series::from_any_values_and_dtype(
                PlSmallStr::from_static("hole"),
                &hole_points,
                &point_schema,
                false,
            )
            .unwrap_or_else(|_| Series::new_empty(PlSmallStr::from_static("hole"), &point_schema));
            AnyValue::List(hole_series)
        })
        .collect();

    // Build the exterior series
    let point_schema = DataType::Struct(vec![
        Field::new(PlSmallStr::from_static("x"), DataType::Float64),
        Field::new(PlSmallStr::from_static("y"), DataType::Float64),
    ]);
    let exterior_series = Series::from_any_values_and_dtype(
        PlSmallStr::from_static("exterior"),
        &exterior_points,
        &point_schema,
        false,
    )
    .unwrap_or_else(|_| Series::new_empty(PlSmallStr::from_static("exterior"), &point_schema));

    // Build the holes series (list of lists)
    let hole_list_schema = DataType::List(Box::new(point_schema.clone()));
    let holes_series = Series::from_any_values_and_dtype(
        PlSmallStr::from_static("holes"),
        &holes_list,
        &hole_list_schema,
        false,
    )
    .unwrap_or_else(|_| Series::new_empty(PlSmallStr::from_static("holes"), &hole_list_schema));

    // Create the outer contour struct
    AnyValue::StructOwned(Box::new((
        vec![
            AnyValue::List(exterior_series),
            AnyValue::List(holes_series),
            AnyValue::Boolean(true), // is_closed
        ],
        vec![
            Field::new(
                PlSmallStr::from_static("exterior"),
                DataType::List(Box::new(point_schema.clone())),
            ),
            Field::new(
                PlSmallStr::from_static("holes"),
                DataType::List(Box::new(hole_list_schema)),
            ),
            Field::new(PlSmallStr::from_static("is_closed"), DataType::Boolean),
        ],
    )))
}

/// Build a contour Series from a vector of Contours.
///
/// This is used by contour transform operations that need to return
/// a properly typed Series.
pub fn build_contour_series(
    name: PlSmallStr,
    contours: Vec<Option<Contour>>,
    input_dtype: &DataType,
) -> PolarsResult<Series> {
    let any_values: Vec<AnyValue> = contours
        .into_iter()
        .map(|opt_c| match opt_c {
            Some(c) => contour_to_anyvalue(&c),
            None => AnyValue::Null,
        })
        .collect();

    Series::from_any_values_and_dtype(name, &any_values, input_dtype, true)
}

// ============================================================================
// Contour Parsing Helpers
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
    /// Origin for scale operations: "origin", "centroid", or "bbox_center".
    #[serde(default)]
    pub origin: Option<String>,
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

// ============================================================================
// Contour Plugin Functions - Measures
// ============================================================================

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

// ============================================================================
// Contour Plugin Functions - Predicates
// ============================================================================

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

// ============================================================================
// Contour Plugin Functions - Pairwise Comparisons
// ============================================================================

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

// ============================================================================
// Contour Plugin Functions - Transforms
// ============================================================================

/// Output type function for contour transform operations (preserves input type).
fn contour_transform_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    if let Some(field) = input_fields.first() {
        Ok(field.clone())
    } else {
        Ok(Field::new(
            PlSmallStr::from_static("output"),
            DataType::Unknown(UnknownKind::Any),
        ))
    }
}

/// Translate contour by offset.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_translate(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let dx = kwargs.dx.unwrap_or(0.0);
    let dy = kwargs.dy.unwrap_or(0.0);

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let translated = transforms::translate(&contour, dx, dy);
            results.push(Some(translated));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Scale contour.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_scale(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let sx = kwargs.sx.unwrap_or(1.0);
    let sy = kwargs.sy.unwrap_or(1.0);

    // Parse origin parameter
    let scale_origin = match kwargs.origin.as_deref() {
        Some("origin") => view_buffer::geometry::ops::ScaleOrigin::Origin,
        Some("bbox_center") => view_buffer::geometry::ops::ScaleOrigin::BBoxCenter,
        Some("centroid") | None => view_buffer::geometry::ops::ScaleOrigin::Centroid,
        _ => view_buffer::geometry::ops::ScaleOrigin::Centroid, // Default fallback
    };

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let scaled = transforms::scale(&contour, sx, sy, scale_origin);
            results.push(Some(scaled));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Simplify contour.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_simplify(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let tolerance = kwargs.tolerance.unwrap_or(1.0);

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let simplified = transforms::simplify(&contour, tolerance);
            results.push(Some(simplified));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Flip contour (reverse winding).
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_flip(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let flipped = transforms::flip(&contour);
            results.push(Some(flipped));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Compute convex hull.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let hull = transforms::convex_hull(&contour);
            results.push(Some(hull));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Normalize contour coordinates to [0, 1] range.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_normalize(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let ref_width = kwargs.ref_width.unwrap_or(1.0);
    let ref_height = kwargs.ref_height.unwrap_or(1.0);

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let normalized = transforms::normalize(&contour, ref_width, ref_height);
            results.push(Some(normalized));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Convert normalized coordinates to absolute pixel coordinates.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_to_absolute(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let ref_width = kwargs.ref_width.unwrap_or(1.0);
    let ref_height = kwargs.ref_height.unwrap_or(1.0);

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let absolute = transforms::to_absolute(&contour, ref_width, ref_height);
            results.push(Some(absolute));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Ensure contour has specified winding direction.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_ensure_winding(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let direction = match kwargs.direction.as_deref() {
        Some("cw") | Some("clockwise") => Winding::Clockwise,
        Some("ccw") | Some("counterclockwise") => Winding::CounterClockwise,
        _ => Winding::CounterClockwise, // Default to CCW
    };

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            results.push(None);
        } else {
            let contour = parse_contour(&value)?;
            let ensured = transforms::ensure_winding(&contour, direction);
            results.push(Some(ensured));
        }
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}
