//! Point plugin functions for polars-cv.
//!
//! This module provides Polars expression functions for point geometry operations,
//! including coordinate transforms (normalize, translate, scale) and distance
//! calculations (Euclidean, Manhattan, point-to-contour).

use polars::prelude::*;
use polars_arrow::array::PrimitiveArray;
use pyo3_polars::derive::polars_expr;

use view_buffer::geometry::contour::Point;

use crate::geom_fns::PointFn;
use crate::geom_params::{parsed_as_another, GeomKwargs, GeomParams};
use view_buffer::mode::Wire;

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

/// Run one row of a point transform and record its result.
///
/// Every transform in this module has the same row shape, and two different
/// things make a row null: a null input point (`input_is_null`), and — under
/// `on_null="null"` — a null per-row parameter. Routing both through one
/// helper keeps the second from being re-implemented in each transform, and
/// keeps them producing identical output.
fn point_row(
    params: &GeomParams,
    input_is_null: bool,
    x_results: &mut Vec<Option<f64>>,
    y_results: &mut Vec<Option<f64>>,
    transform: impl FnOnce() -> PolarsResult<(f64, f64)>,
) -> PolarsResult<()> {
    let result = if input_is_null {
        None
    } else {
        params.row(transform)?
    };
    match result {
        Some((x, y)) => {
            x_results.push(Some(x));
            y_results.push(Some(y));
        }
        None => {
            x_results.push(None);
            y_results.push(None);
        }
    }
    Ok(())
}

// ============================================================================
// Output Type Functions
// ============================================================================

/// Output type for point transform operations (returns Point struct).
fn point_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("point"),
        crate::geom_schema::point_struct_dtype(),
    ))
}

// ============================================================================
// Point Plugin Functions - Coordinate Transforms
// ============================================================================

/// Normalize point coordinates to [0, 1] range.
#[polars_expr(output_type_func=point_output_type)]
fn point_normalize(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_normalize";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::Normalize { width, height } = &op else {
        return Err(parsed_as_another(NAME));
    };

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        point_row(
            &params,
            value.is_null(),
            &mut x_results,
            &mut y_results,
            || {
                let ref_width = params.value(width, i)?;
                let ref_height = params.value(height, i)?;
                // Per-row dimensions cannot be validated once per batch, so the
                // divide-by-zero guard moves into the loop and names the row.
                if ref_width == 0.0 || ref_height == 0.0 {
                    polars_bail!(ComputeError:
                        "ref_width and ref_height must be non-zero (row {})", i
                    );
                }
                let (x, y) = parse_point(&value)?;
                Ok((x / ref_width, y / ref_height))
            },
        )?;
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

/// Convert normalized coordinates to absolute pixel coordinates.
#[polars_expr(output_type_func=point_output_type)]
fn point_to_absolute(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_to_absolute";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::ToAbsolute { width, height } = &op else {
        return Err(parsed_as_another(NAME));
    };

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        point_row(
            &params,
            value.is_null(),
            &mut x_results,
            &mut y_results,
            || {
                let ref_width = params.value(width, i)?;
                let ref_height = params.value(height, i)?;
                let (x, y) = parse_point(&value)?;
                Ok((x * ref_width, y * ref_height))
            },
        )?;
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

/// Translate point by offset.
#[polars_expr(output_type_func=point_output_type)]
fn point_translate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_translate";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::Translate { dx, dy } = &op else {
        return Err(parsed_as_another(NAME));
    };

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        point_row(
            &params,
            value.is_null(),
            &mut x_results,
            &mut y_results,
            || {
                let dx = params.value(dx, i)?;
                let dy = params.value(dy, i)?;
                let (x, y) = parse_point(&value)?;
                Ok((x + dx, y + dy))
            },
        )?;
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

/// Scale point coordinates.
#[polars_expr(output_type_func=point_output_type)]
fn point_scale(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_scale";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::Scale { sx, sy } = &op else {
        return Err(parsed_as_another(NAME));
    };

    let series = &inputs[0];
    let len = series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        point_row(
            &params,
            value.is_null(),
            &mut x_results,
            &mut y_results,
            || {
                let sx = params.value(sx, i)?;
                let sy = params.value(sy, i)?;
                let (x, y) = parse_point(&value)?;
                Ok((x * sx, y * sy))
            },
        )?;
    }

    build_point_series(series.name().clone(), x_results, y_results)
}

// ============================================================================
// Point Plugin Functions - Pairwise Distance Operations
// ============================================================================

/// Compute Euclidean distance between two points.
#[polars_expr(output_type=Float64)]
fn point_distance(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_distance";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::Distance { other } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let series_a = &inputs[0];
    let series_b = params.column(other);
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
fn point_manhattan_distance(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_manhattan_distance";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::ManhattanDistance { other } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let series_a = &inputs[0];
    let series_b = params.column(other);
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
fn point_distance_to_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_distance_to_contour";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::DistanceToContour { contour } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let point_series = &inputs[0];
    let contour_series = params.column(contour);
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
fn point_signed_distance_to_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_signed_distance_to_contour";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::SignedDistanceToContour { contour } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let point_series = &inputs[0];
    let contour_series = params.column(contour);
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
fn point_nearest_on_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_nearest_on_contour";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::NearestOnContour { contour } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let point_series = &inputs[0];
    let contour_series = params.column(contour);
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
fn point_angle_to(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_angle_to";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::AngleTo { other } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let series_a = &inputs[0];
    let series_b = params.column(other);
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
fn point_rotate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_rotate";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::Rotate { angle, origin } = &op else {
        return Err(parsed_as_another(NAME));
    };

    let point_series = &inputs[0];
    let len = point_series.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    // Read through its reference: `origin` is optional, so its position is
    // not fixed once a per-row `angle` can also occupy an input slot.
    let origin_series = params.optional_column(origin);

    for i in 0..len {
        let point_value = point_series.get(i)?;

        point_row(
            &params,
            point_value.is_null(),
            &mut x_results,
            &mut y_results,
            || {
                // The angle may vary per row, so the trig moves into the loop.
                let angle = params.value(angle, i)?;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let (x, y) = parse_point(&point_value)?;

                // Get origin (default to 0,0)
                let (ox, oy) = match origin_series {
                    Some(col) => {
                        let origin_value = col.get(i)?;
                        if origin_value.is_null() {
                            (0.0, 0.0)
                        } else {
                            parse_point(&origin_value)?
                        }
                    }
                    None => (0.0, 0.0),
                };

                // Rotate around origin
                let dx = x - ox;
                let dy = y - oy;
                Ok((dx * cos_a - dy * sin_a + ox, dx * sin_a + dy * cos_a + oy))
            },
        )?;
    }

    build_point_series(point_series.name().clone(), x_results, y_results)
}

/// Compute midpoint between two points.
#[polars_expr(output_type_func=point_output_type)]
fn point_midpoint(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_midpoint";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::Midpoint { other } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let series_a = &inputs[0];
    let series_b = params.column(other);
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
fn point_interpolate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_interpolate";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::Interpolate { other, t } = &op else {
        return Err(parsed_as_another(NAME));
    };

    let series_a = &inputs[0];
    let series_b = params.column(other);
    let len = series_a.len();
    let mut x_results = Vec::with_capacity(len);
    let mut y_results = Vec::with_capacity(len);

    for i in 0..len {
        let value_a = series_a.get(i)?;
        let value_b = series_b.get(i)?;

        let input_is_null = value_a.is_null() || value_b.is_null();
        point_row(
            &params,
            input_is_null,
            &mut x_results,
            &mut y_results,
            || {
                let t = params.value(t, i)?;
                let (x1, y1) = parse_point(&value_a)?;
                let (x2, y2) = parse_point(&value_b)?;
                Ok((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            },
        )?;
    }

    build_point_series(series_a.name().clone(), x_results, y_results)
}

/// Check if point is within bounding box.
#[polars_expr(output_type=Boolean)]
fn point_within_bbox(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "point_within_bbox";
    let (op, params) = GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, NAME)?;
    let PointFn::WithinBbox { bbox } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let point_series = &inputs[0];
    let bbox_series = params.column(bbox);
    let len = point_series.len();
    let mut results = Vec::with_capacity(len);

    for i in 0..len {
        let point_value = point_series.get(i)?;
        let bbox_value = bbox_series.get(i)?;

        if point_value.is_null() || bbox_value.is_null() {
            results.push(None);
        } else {
            let (px, py) = parse_point(&point_value)?;
            let bbox = crate::geom_schema::parse_bbox(&bbox_value)?;

            let within = px >= bbox.x
                && px <= bbox.x + bbox.width
                && py >= bbox.y
                && py <= bbox.y + bbox.height;
            results.push(Some(within));
        }
    }

    Ok(
        BooleanChunked::from_iter_options(point_series.name().clone(), results.into_iter())
            .into_series(),
    )
}
