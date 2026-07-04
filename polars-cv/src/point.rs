//! Point plugin functions for polars-cv.
//!
//! This module provides Polars expression functions for point geometry operations,
//! including coordinate transforms (normalize, translate, scale) and distance
//! calculations (Euclidean, Manhattan, point-to-contour).

use polars::prelude::*;
use polars_arrow::array::PrimitiveArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use view_buffer::geometry::contour::Point;

// ============================================================================
// Point Kwargs
// ============================================================================

/// Kwargs for point operations with optional parameters.
#[derive(Debug, Deserialize)]
pub struct PointKwargs {
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
    /// Rotation angle in radians.
    #[serde(default)]
    pub angle: Option<f64>,
    /// Interpolation parameter (0 to 1).
    #[serde(default)]
    pub t: Option<f64>,
}

// ============================================================================
// Point Parsing Helpers
// ============================================================================

/// Parse a point from a Polars Struct value.
///
/// Points are stored as a struct column matching POINT_SCHEMA:
/// {x: Float64, y: Float64}
fn parse_point(value: &AnyValue) -> PolarsResult<(f64, f64)> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();
            let mut x = None;
            let mut y = None;

            for (i, field) in fields.iter().enumerate() {
                match field.name().as_str() {
                    "x" => x = values.get(i).and_then(|v| v.try_extract::<f64>().ok()),
                    "y" => y = values.get(i).and_then(|v| v.try_extract::<f64>().ok()),
                    _ => {}
                }
            }

            let x = x.ok_or_else(|| polars_err!(ComputeError: "Point struct missing 'x' field"))?;
            let y = y.ok_or_else(|| polars_err!(ComputeError: "Point struct missing 'y' field"))?;
            Ok((x, y))
        }
        AnyValue::Struct(row_idx, struct_arr, fields) => {
            let values = struct_arr.values();
            let mut x = None;
            let mut y = None;

            for (i, field) in fields.iter().enumerate() {
                if let Some(arr) = values.get(i) {
                    if let Some(f64_arr) = arr.as_any().downcast_ref::<PrimitiveArray<f64>>() {
                        match field.name().as_str() {
                            "x" => x = f64_arr.get(*row_idx),
                            "y" => y = f64_arr.get(*row_idx),
                            _ => {}
                        }
                    }
                }
            }

            let x = x.ok_or_else(|| polars_err!(ComputeError: "Point struct missing 'x' field"))?;
            let y = y.ok_or_else(|| polars_err!(ComputeError: "Point struct missing 'y' field"))?;
            Ok((x, y))
        }
        _ => Err(polars_err!(ComputeError: "Expected Point struct, got {:?}", value)),
    }
}

/// Build a Point struct Series from x and y vectors.
fn build_point_series(
    name: PlSmallStr,
    x_values: Vec<Option<f64>>,
    y_values: Vec<Option<f64>>,
) -> PolarsResult<Series> {
    let len = x_values.len();
    let x_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("x"), x_values.into_iter())
            .into_series();
    let y_col =
        Float64Chunked::from_iter_options(PlSmallStr::from_static("y"), y_values.into_iter())
            .into_series();

    StructChunked::from_series(name, len, [x_col, y_col].iter()).map(|ca| ca.into_series())
}

// ============================================================================
// Output Type Functions
// ============================================================================

/// Output type for point transform operations (returns Point struct).
fn point_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new(PlSmallStr::from_static("x"), DataType::Float64),
        Field::new(PlSmallStr::from_static("y"), DataType::Float64),
    ];
    Ok(Field::new(
        PlSmallStr::from_static("point"),
        DataType::Struct(fields),
    ))
}

// ============================================================================
// Point Plugin Functions - Coordinate Transforms
// ============================================================================

/// Normalize point coordinates to [0, 1] range.
#[polars_expr(output_type_func=point_output_type)]
fn point_normalize(inputs: &[Series], kwargs: PointKwargs) -> PolarsResult<Series> {
    let ref_width = kwargs
        .ref_width
        .ok_or_else(|| polars_err!(ComputeError: "ref_width is required"))?;
    let ref_height = kwargs
        .ref_height
        .ok_or_else(|| polars_err!(ComputeError: "ref_height is required"))?;

    if ref_width == 0.0 || ref_height == 0.0 {
        return Err(polars_err!(ComputeError: "ref_width and ref_height must be non-zero"));
    }

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (x, y) = parse_point(&value)?;
            x_results.push(Some(x / ref_width));
            y_results.push(Some(y / ref_height));
        }
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

/// Convert normalized coordinates to absolute pixel coordinates.
#[polars_expr(output_type_func=point_output_type)]
fn point_to_absolute(inputs: &[Series], kwargs: PointKwargs) -> PolarsResult<Series> {
    let ref_width = kwargs
        .ref_width
        .ok_or_else(|| polars_err!(ComputeError: "ref_width is required"))?;
    let ref_height = kwargs
        .ref_height
        .ok_or_else(|| polars_err!(ComputeError: "ref_height is required"))?;

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (x, y) = parse_point(&value)?;
            x_results.push(Some(x * ref_width));
            y_results.push(Some(y * ref_height));
        }
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

/// Translate point by offset.
#[polars_expr(output_type_func=point_output_type)]
fn point_translate(inputs: &[Series], kwargs: PointKwargs) -> PolarsResult<Series> {
    let dx = kwargs.dx.unwrap_or(0.0);
    let dy = kwargs.dy.unwrap_or(0.0);

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (x, y) = parse_point(&value)?;
            x_results.push(Some(x + dx));
            y_results.push(Some(y + dy));
        }
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

/// Scale point coordinates.
#[polars_expr(output_type_func=point_output_type)]
fn point_scale(inputs: &[Series], kwargs: PointKwargs) -> PolarsResult<Series> {
    let sx = kwargs.sx.unwrap_or(1.0);
    let sy = kwargs.sy.unwrap_or(1.0);

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (x, y) = parse_point(&value)?;
            x_results.push(Some(x * sx));
            y_results.push(Some(y * sy));
        }
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

// ============================================================================
// Point Plugin Functions - Pairwise Distance Operations
// ============================================================================

/// Compute Euclidean distance between two points.
#[polars_expr(output_type=Float64)]
fn point_distance(inputs: &[Series]) -> PolarsResult<Series> {
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
            let (x1, y1) = parse_point(&value_a)?;
            let (x2, y2) = parse_point(&value_b)?;
            let dx = x2 - x1;
            let dy = y2 - y1;
            results.push(Some((dx * dx + dy * dy).sqrt()));
        }
    }

    Ok(
        Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter())
            .into_series(),
    )
}

/// Compute Manhattan (L1) distance between two points.
#[polars_expr(output_type=Float64)]
fn point_manhattan_distance(inputs: &[Series]) -> PolarsResult<Series> {
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
            let (x1, y1) = parse_point(&value_a)?;
            let (x2, y2) = parse_point(&value_b)?;
            results.push(Some((x2 - x1).abs() + (y2 - y1).abs()));
        }
    }

    Ok(
        Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter())
            .into_series(),
    )
}

// ============================================================================
// Point Plugin Functions - Point-to-Contour Operations
// ============================================================================

/// Compute minimum distance from point to contour boundary.
#[polars_expr(output_type=Float64)]
fn point_distance_to_contour(inputs: &[Series]) -> PolarsResult<Series> {
    let point_series = &inputs[0];
    let contour_series = &inputs[1];
    let len = point_series.len();
    let mut results = Vec::with_capacity(len);

    for i in 0..len {
        let point_value = point_series.get(i)?;
        let contour_value = contour_series.get(i)?;

        if point_value.is_null() || contour_value.is_null() {
            results.push(None);
        } else {
            let (px, py) = parse_point(&point_value)?;
            let contour = crate::contour::parse_contour(&contour_value)?;
            let point = Point::new(px, py);
            let dist = view_buffer::geometry::measures::distance_to_contour(&point, &contour);
            results.push(Some(dist));
        }
    }

    Ok(
        Float64Chunked::from_iter_options(point_series.name().clone(), results.into_iter())
            .into_series(),
    )
}

/// Compute signed distance from point to contour boundary.
/// Negative if inside, positive if outside.
#[polars_expr(output_type=Float64)]
fn point_signed_distance_to_contour(inputs: &[Series]) -> PolarsResult<Series> {
    let point_series = &inputs[0];
    let contour_series = &inputs[1];
    let len = point_series.len();
    let mut results = Vec::with_capacity(len);

    for i in 0..len {
        let point_value = point_series.get(i)?;
        let contour_value = contour_series.get(i)?;

        if point_value.is_null() || contour_value.is_null() {
            results.push(None);
        } else {
            let (px, py) = parse_point(&point_value)?;
            let contour = crate::contour::parse_contour(&contour_value)?;
            let point = Point::new(px, py);
            let dist = view_buffer::geometry::measures::distance_to_contour(&point, &contour);

            // Check if point is inside using point-in-polygon
            let is_inside = view_buffer::geometry::predicates::contains_point(&contour, px, py);

            let signed_dist = if is_inside { -dist } else { dist };
            results.push(Some(signed_dist));
        }
    }

    Ok(
        Float64Chunked::from_iter_options(point_series.name().clone(), results.into_iter())
            .into_series(),
    )
}

/// Find nearest point on contour boundary.
#[polars_expr(output_type_func=point_output_type)]
fn point_nearest_on_contour(inputs: &[Series]) -> PolarsResult<Series> {
    let point_series = &inputs[0];
    let contour_series = &inputs[1];
    let len = point_series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let point_value = point_series.get(i)?;
        let contour_value = contour_series.get(i)?;

        if point_value.is_null() || contour_value.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (px, py) = parse_point(&point_value)?;
            let contour = crate::contour::parse_contour(&contour_value)?;
            let point = Point::new(px, py);

            if let Some(nearest) =
                view_buffer::geometry::measures::nearest_point_on_contour(&point, &contour)
            {
                x_results.push(Some(nearest.x));
                y_results.push(Some(nearest.y));
            } else {
                x_results.push(None);
                y_results.push(None);
            }
        }
    }

    build_point_series(point_series.name().clone(), x_results, y_results)
}

// ============================================================================
// Point Plugin Functions - Geometric Operations
// ============================================================================

/// Compute angle from this point to another in radians.
#[polars_expr(output_type=Float64)]
fn point_angle_to(inputs: &[Series]) -> PolarsResult<Series> {
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
            let (x1, y1) = parse_point(&value_a)?;
            let (x2, y2) = parse_point(&value_b)?;
            let angle = (y2 - y1).atan2(x2 - x1);
            results.push(Some(angle));
        }
    }

    Ok(
        Float64Chunked::from_iter_options(series_a.name().clone(), results.into_iter())
            .into_series(),
    )
}

/// Rotate point around origin by angle (radians).
#[polars_expr(output_type_func=point_output_type)]
fn point_rotate(inputs: &[Series], kwargs: PointKwargs) -> PolarsResult<Series> {
    let angle = kwargs.angle.unwrap_or(0.0);
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let point_series = &inputs[0];
    let len = point_series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    // Check if an origin point is provided
    let has_origin = inputs.len() > 1;

    for i in 0..len {
        let point_value = point_series.get(i)?;

        if point_value.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (x, y) = parse_point(&point_value)?;

            // Get origin (default to 0,0)
            let (ox, oy) = if has_origin {
                let origin_value = inputs[1].get(i)?;
                if origin_value.is_null() {
                    (0.0, 0.0)
                } else {
                    parse_point(&origin_value)?
                }
            } else {
                (0.0, 0.0)
            };

            // Rotate around origin
            let dx = x - ox;
            let dy = y - oy;
            let new_x = dx * cos_a - dy * sin_a + ox;
            let new_y = dx * sin_a + dy * cos_a + oy;

            x_results.push(Some(new_x));
            y_results.push(Some(new_y));
        }
    }

    build_point_series(point_series.name().clone(), x_results, y_results)
}

/// Compute midpoint between two points.
#[polars_expr(output_type_func=point_output_type)]
fn point_midpoint(inputs: &[Series]) -> PolarsResult<Series> {
    let series_a = &inputs[0];
    let series_b = &inputs[1];
    let len = series_a.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value_a = series_a.get(i)?;
        let value_b = series_b.get(i)?;

        if value_a.is_null() || value_b.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (x1, y1) = parse_point(&value_a)?;
            let (x2, y2) = parse_point(&value_b)?;
            x_results.push(Some((x1 + x2) / 2.0));
            y_results.push(Some((y1 + y2) / 2.0));
        }
    }

    build_point_series(series_a.name().clone(), x_results, y_results)
}

/// Linear interpolation between two points.
#[polars_expr(output_type_func=point_output_type)]
fn point_interpolate(inputs: &[Series], kwargs: PointKwargs) -> PolarsResult<Series> {
    let t = kwargs.t.unwrap_or(0.5);

    let series_a = &inputs[0];
    let series_b = &inputs[1];
    let len = series_a.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value_a = series_a.get(i)?;
        let value_b = series_b.get(i)?;

        if value_a.is_null() || value_b.is_null() {
            x_results.push(None);
            y_results.push(None);
        } else {
            let (x1, y1) = parse_point(&value_a)?;
            let (x2, y2) = parse_point(&value_b)?;
            x_results.push(Some(x1 + t * (x2 - x1)));
            y_results.push(Some(y1 + t * (y2 - y1)));
        }
    }

    build_point_series(series_a.name().clone(), x_results, y_results)
}

/// Parse bbox struct {x, y, width, height} from any supported AnyValue format.
fn parse_bbox(value: &AnyValue) -> PolarsResult<(f64, f64, f64, f64)> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();
            let mut x = None;
            let mut y = None;
            let mut w = None;
            let mut h = None;

            for (i, field) in fields.iter().enumerate() {
                match field.name().as_str() {
                    "x" => x = values.get(i).and_then(|v| v.try_extract::<f64>().ok()),
                    "y" => y = values.get(i).and_then(|v| v.try_extract::<f64>().ok()),
                    "width" => w = values.get(i).and_then(|v| v.try_extract::<f64>().ok()),
                    "height" => h = values.get(i).and_then(|v| v.try_extract::<f64>().ok()),
                    _ => {}
                }
            }

            Ok((
                x.unwrap_or(0.0),
                y.unwrap_or(0.0),
                w.unwrap_or(0.0),
                h.unwrap_or(0.0),
            ))
        }
        AnyValue::Struct(row_idx, struct_arr, fields) => {
            let values = struct_arr.values();
            let mut x = None;
            let mut y = None;
            let mut w = None;
            let mut h = None;

            for (i, field) in fields.iter().enumerate() {
                if let Some(arr) = values.get(i) {
                    if let Some(f64_arr) = arr.as_any().downcast_ref::<PrimitiveArray<f64>>() {
                        match field.name().as_str() {
                            "x" => x = f64_arr.get(*row_idx),
                            "y" => y = f64_arr.get(*row_idx),
                            "width" => w = f64_arr.get(*row_idx),
                            "height" => h = f64_arr.get(*row_idx),
                            _ => {}
                        }
                    }
                }
            }

            Ok((
                x.unwrap_or(0.0),
                y.unwrap_or(0.0),
                w.unwrap_or(0.0),
                h.unwrap_or(0.0),
            ))
        }
        _ => Err(polars_err!(ComputeError: "Expected BBox struct")),
    }
}

/// Check if point is within bounding box.
#[polars_expr(output_type=Boolean)]
fn point_within_bbox(inputs: &[Series]) -> PolarsResult<Series> {
    let point_series = &inputs[0];
    let bbox_series = &inputs[1];
    let len = point_series.len();
    let mut results = Vec::with_capacity(len);

    for i in 0..len {
        let point_value = point_series.get(i)?;
        let bbox_value = bbox_series.get(i)?;

        if point_value.is_null() || bbox_value.is_null() {
            results.push(None);
        } else {
            let (px, py) = parse_point(&point_value)?;
            let (bx, by, bw, bh) = parse_bbox(&bbox_value)?;

            let within = px >= bx && px <= bx + bw && py >= by && py <= by + bh;
            results.push(Some(within));
        }
    }

    Ok(
        BooleanChunked::from_iter_options(point_series.name().clone(), results.into_iter())
            .into_series(),
    )
}
