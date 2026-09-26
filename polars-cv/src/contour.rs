//! Contour plugin functions for polars-cv.
//!
//! This module provides Polars expression functions for contour geometry operations,
//! including measures (area, perimeter), predicates (is_convex, contains_point),
//! transforms (translate, scale, simplify), and pairwise comparisons (IoU, Dice).

use polars::prelude::*;

use crate::geom_schema::{
    bbox_anyvalue, bbox_struct_dtype, parse_bbox, point_anyvalue, point_struct_dtype,
};
use polars_arrow::array::{ListArray, PrimitiveArray, StructArray as ArrowStructArray};
use pyo3_polars::derive::polars_expr;

// Import geometry operations from view-buffer
use view_buffer::geometry::{
    contour::{BoundingBox, Contour, Point, Winding},
    label::score_contours_on_buffer,
    measures, pairwise, predicates, transforms,
};
use view_buffer::ViewBuffer;

// `contour_accessor!` is `#[macro_export]`ed, so it lives at the crate root
// regardless of module order; importing it by name avoids depending on
// `geom_arity` being declared before `contour` in lib.rs.
use crate::contour_accessor;
use crate::geom_arity::{elementwise_field, row_contours, Arity, ContourOutput};
use crate::geom_fns::{BBoxFn, ContourFn};
use crate::geom_params::{check_range, parsed_as_another, GeomKwargs, GeomParams};
use crate::ops::{ColumnRef, Param};
use view_buffer::mode::Wire;
use view_buffer::GeometryOp;

// ============================================================================
// Contour Serialization Helpers
// ============================================================================

/// Convert a Contour to a Polars AnyValue matching CONTOUR_SCHEMA.
///
/// The schema is:
/// - exterior: List[{x: Float64, y: Float64}]
/// - holes: List[List[{x: Float64, y: Float64}]] — the sole carrier of hole-ness;
///   ring winding is never interpreted as a hole signal
/// - is_closed: Boolean — reserved. Always written `true` here and ignored by
///   `parse_contour`; rings are implicitly closed.
///
/// Test-only: production contour columns are built straight into Arrow by
/// `geom_schema::contour_array` (CR-36). This per-value construction stays as
/// the independent oracle the equivalence tests compare that builder against.
#[cfg(test)]
pub fn contour_to_anyvalue(contour: &Contour) -> AnyValue<'static> {
    // Build exterior points as list of structs
    let exterior_points: Vec<AnyValue> = contour
        .exterior
        .iter()
        .map(|p| point_anyvalue(p.x, p.y))
        .collect();

    // Build holes as list of list of structs
    let holes_list: Vec<AnyValue> = contour
        .holes
        .iter()
        .map(|hole| {
            let hole_points: Vec<AnyValue> =
                hole.iter().map(|p| point_anyvalue(p.x, p.y)).collect();
            // Create a Series from hole points for the inner list
            let point_schema = point_struct_dtype();
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
    let point_schema = point_struct_dtype();
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

    // Create the outer contour struct. The field layout is declared once in
    // `geom_schema::contour_fields`; this reads it rather than re-spelling it.
    AnyValue::StructOwned(Box::new((
        vec![
            AnyValue::List(exterior_series),
            AnyValue::List(holes_series),
            AnyValue::Boolean(true), // is_closed: reserved, never read back
        ],
        crate::geom_schema::contour_fields(),
    )))
}

// ============================================================================
// Contour Parsing Helpers
// ============================================================================

/// Parse a contour from a Polars value.
///
/// The **single** Struct/List -> `Contour` parser for the whole plugin:
/// the contour namespace, the point namespace (`point.rs`), and the contour
/// source decoder (`execute.rs`) all route through it, so hole handling,
/// accepted input forms, and error text cannot diverge between consumers
/// (pinned by `parse_contour_tests` and `tests/test_contour_parsing.py`).
///
/// Accepted forms:
/// - a struct with an `exterior: List[{x, y}]` field and optional
///   `holes: List[List[{x, y}]]` (other fields, such as `is_closed`, are not
///   read). A struct without `exterior` is refused, not guessed at.
/// - a bare `List[{x, y}]` (a simple contour without holes)
pub(crate) fn parse_contour(value: &AnyValue) -> PolarsResult<Contour> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();
            let mut exterior: Option<Vec<Point>> = None;
            let mut holes: Vec<Vec<Point>> = Vec::new();

            for (i, field) in fields.iter().enumerate() {
                match field.name().as_str() {
                    "exterior" => {
                        if let Some(AnyValue::List(series)) = values.get(i) {
                            exterior = Some(extract_points_from_series(series)?);
                        }
                    }
                    "holes" => {
                        if let Some(AnyValue::List(series)) = values.get(i) {
                            holes = extract_holes_from_series(series)?;
                        }
                    }
                    _ => {}
                }
            }

            let exterior = exterior.ok_or_else(missing_exterior)?;

            Ok(Contour::with_holes(exterior, holes))
        }
        // Handle AnyValue::Struct (non-owned variant with row index and array reference)
        AnyValue::Struct(row_idx, struct_array, fields) => {
            let mut exterior: Option<Vec<Point>> = None;
            let mut holes: Vec<Vec<Point>> = Vec::new();

            for (i, field) in fields.iter().enumerate() {
                let column = struct_array.values()[i].clone();
                match field.name().as_str() {
                    "exterior" => {
                        if let Some(list_arr) = column.as_any().downcast_ref::<ListArray<i64>>() {
                            let offsets = list_arr.offsets();
                            let start = offsets[*row_idx] as usize;
                            let end = offsets[*row_idx + 1] as usize;
                            let values_arr = list_arr.values();
                            if let Some(struct_arr) =
                                values_arr.as_any().downcast_ref::<ArrowStructArray>()
                            {
                                exterior =
                                    Some(extract_points_from_struct_array(struct_arr, start, end)?);
                            }
                        }
                    }
                    "holes" => {
                        holes = extract_holes_from_arrow_array(column.as_ref(), *row_idx)?;
                    }
                    _ => {}
                }
            }

            let exterior = exterior.ok_or_else(missing_exterior)?;
            Ok(Contour::with_holes(exterior, holes))
        }
        AnyValue::List(series) => {
            // Direct list of points (simpler format)
            let points = extract_points_from_series(series)?;
            Ok(Contour::new(points))
        }
        _ => Err(polars_err!(ComputeError: "Expected Struct or List for contour, got {:?}", value)),
    }
}

fn missing_exterior() -> PolarsError {
    polars_err!(ComputeError: "Contour struct has no 'exterior' field")
}

/// Parse geometry that may be a single contour or a whole set of them.
///
/// The repack that lets one code path serve both shapes a geometry column takes:
/// `extract_contours().sink("native")` emits `List[Contour]`, while a
/// hand-written contour column is one `Struct` per row. Both arrive here and
/// leave as a set. Used by the contour *source* (whose rasterizer paints their
/// union) and by the set-level accessors (`pairwise_iou`, `correspond`,
/// `label_reduce`), which therefore accept a single contour as a one-element
/// set — the mirror of the `.contour` accessors accepting a set.
///
/// The two list forms are told apart by the element dtype — via
/// [`Arity::of`](crate::geom_arity::Arity::of), the same reading the accessors'
/// declared output types use — not by trying one and falling back: a `List`
/// whose elements are point structs is one contour's ring, anything else in a
/// `List` is a set of contours. A fallback would have to guess, and guessing
/// wrong on a contour set is what used to surface as
/// `Point struct missing 'x' field`.
///
/// Accepted forms:
/// - `List[Contour]` — a contour set (elements are parsed by [`parse_contour`])
/// - anything [`parse_contour`] accepts — a single contour, as a one-element set
/// - null — the empty set, which rasterizes to an all-background mask
pub(crate) fn parse_contour_set(value: &AnyValue) -> PolarsResult<Vec<Contour>> {
    match value {
        AnyValue::Null => Ok(Vec::new()),
        AnyValue::List(series) if !crate::geom_arity::is_point_dtype(series.dtype()) => {
            parse_contour_list(value)
        }
        _ => Ok(vec![parse_contour(value)?]),
    }
}

/// The field names a point struct may spell its coordinates with, in order.
///
/// **The single authority for "is this a point?".** Read by
/// [`extract_points_from_series`], which parses them, and by
/// [`is_point_dtype`](crate::geom_arity::is_point_dtype), which decides from the
/// dtype whether a `List` is one contour's ring or a set of contours. The two
/// used to spell the names separately, so a dtype test could admit a struct the
/// parser then rejected.
pub(crate) fn point_dtype_fields() -> [[&'static str; 2]; 2] {
    [["x", "X"], ["y", "Y"]]
}

/// Parse a list of contours from an AnyValue list expression.
pub(crate) fn parse_contour_list(value: &AnyValue) -> PolarsResult<Vec<Contour>> {
    match value {
        AnyValue::List(series) => {
            let mut contours = Vec::with_capacity(series.len());
            for i in 0..series.len() {
                let item = series.get(i)?;
                if item.is_null() {
                    continue;
                }
                contours.push(parse_contour(&item)?);
            }
            Ok(contours)
        }
        AnyValue::Null => Ok(Vec::new()),
        _ => Err(polars_err!(ComputeError: "Expected List[Contour], got {:?}", value)),
    }
}

/// Parse optional score list aligned with contour list.
/// Parse holes from a Series where each element is a ring list of points.
fn extract_holes_from_series(series: &Series) -> PolarsResult<Vec<Vec<Point>>> {
    let mut holes: Vec<Vec<Point>> = Vec::with_capacity(series.len());
    for i in 0..series.len() {
        let value = series.get(i)?;
        if value.is_null() {
            continue;
        }
        match value {
            AnyValue::List(ring_series) => {
                holes.push(extract_points_from_series(&ring_series)?);
            }
            _ => {
                return Err(polars_err!(
                    ComputeError: "Expected hole ring as List[Point], got {:?}", value
                ));
            }
        }
    }
    Ok(holes)
}

/// Parse holes from Arrow representation of nested lists at a specific row index.
fn extract_holes_from_arrow_array(
    array: &dyn polars_arrow::array::Array,
    row_idx: usize,
) -> PolarsResult<Vec<Vec<Point>>> {
    let Some(outer_list) = array.as_any().downcast_ref::<ListArray<i64>>() else {
        return Ok(Vec::new());
    };

    let outer_offsets = outer_list.offsets();
    let hole_start = outer_offsets[row_idx] as usize;
    let hole_end = outer_offsets[row_idx + 1] as usize;
    if hole_start == hole_end {
        return Ok(Vec::new());
    }

    let inner_values = outer_list.values();
    let Some(inner_list) = inner_values.as_any().downcast_ref::<ListArray<i64>>() else {
        return Err(polars_err!(ComputeError: "holes field must be List[List[Point]]"));
    };

    let inner_offsets = inner_list.offsets();
    let point_values = inner_list.values();
    let Some(point_struct_arr) = point_values.as_any().downcast_ref::<ArrowStructArray>() else {
        return Err(polars_err!(ComputeError: "holes rings must contain point structs"));
    };

    let mut holes: Vec<Vec<Point>> = Vec::with_capacity(hole_end - hole_start);
    for ring_idx in hole_start..hole_end {
        let start = inner_offsets[ring_idx] as usize;
        let end = inner_offsets[ring_idx + 1] as usize;
        holes.push(extract_points_from_struct_array(
            point_struct_arr,
            start,
            end,
        )?);
    }
    Ok(holes)
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

fn parse_numeric_series(series: &Series) -> PolarsResult<Vec<f64>> {
    let mut values = Vec::with_capacity(series.len());
    for i in 0..series.len() {
        let av = series.get(i)?;
        if av.is_null() {
            continue;
        }
        let value = av.try_extract::<f64>().map_err(
            |_| polars_err!(ComputeError: "Image/array values must be numeric, found {:?}", av),
        )?;
        values.push(value);
    }
    Ok(values)
}

fn parse_row_values_with_optional_channel(series: &Series) -> PolarsResult<Vec<f64>> {
    let mut first_non_null: Option<AnyValue> = None;
    for i in 0..series.len() {
        let item = series.get(i)?;
        if !item.is_null() {
            first_non_null = Some(item);
            break;
        }
    }
    let Some(sample) = first_non_null else {
        return Ok(Vec::new());
    };

    if matches!(sample, AnyValue::List(_) | AnyValue::Array(_, _)) {
        let mut values = Vec::with_capacity(series.len());
        for i in 0..series.len() {
            let pixel = series.get(i)?;
            if pixel.is_null() {
                continue;
            }
            let pixel_series = match pixel {
                AnyValue::List(inner) => inner,
                AnyValue::Array(inner, _) => inner,
                _ => {
                    return Err(polars_err!(
                        ComputeError: "Expected pixel channel values as list/array, found {:?}",
                        pixel
                    ))
                }
            };
            let channels = parse_numeric_series(&pixel_series)?;
            if channels.len() != 1 {
                return Err(polars_err!(
                    ComputeError: "Only single-channel row values are supported, found {} channels",
                    channels.len()
                ));
            }
            values.push(channels[0]);
        }
        return Ok(values);
    }

    parse_numeric_series(series)
}

fn parse_grid_rows(series: &Series) -> PolarsResult<Vec<Vec<f64>>> {
    let mut rows = Vec::with_capacity(series.len());
    for i in 0..series.len() {
        let row = series.get(i)?;
        if row.is_null() {
            continue;
        }
        let row_series = match row {
            AnyValue::List(inner) => inner,
            AnyValue::Array(inner, _) => inner,
            _ => {
                return Err(polars_err!(
                    ComputeError: "Image rows must be list/array values, found {:?}",
                    row
                ))
            }
        };
        rows.push(parse_row_values_with_optional_channel(&row_series)?);
    }
    if rows.windows(2).any(|w| w[0].len() != w[1].len()) {
        return Err(polars_err!(
            ComputeError: "Image rows must have uniform width"
        ));
    }
    Ok(rows)
}

/// Parse a nested list/array image column value into a `[H, W, 1]` `ViewBuffer`.
///
/// A buffer, rather than a row-of-rows grid, because that is what
/// [`score_contours_on_buffer`] consumes — the same engine entry point
/// `Pipeline.label_reduce` reaches through the graph. An empty or null value
/// yields a `[0, 0, 1]` buffer, which scores every contour 0.0.
fn parse_heatmap(value: &AnyValue) -> PolarsResult<ViewBuffer> {
    let rows: Vec<Vec<f64>> = match value {
        AnyValue::List(series) | AnyValue::Array(series, _) => {
            let mut first_non_null: Option<AnyValue> = None;
            for i in 0..series.len() {
                let item = series.get(i)?;
                if !item.is_null() {
                    first_non_null = Some(item);
                    break;
                }
            }

            match first_non_null {
                // Rows of pixels: a full [H, W] grid. A flat list of scalars is a
                // single row, and an empty/all-null column has no rows at all.
                Some(AnyValue::List(_) | AnyValue::Array(_, _)) => parse_grid_rows(series)?,
                Some(_) => vec![parse_numeric_series(series)?],
                None => Vec::new(),
            }
        }
        AnyValue::Null => Vec::new(),
        _ => {
            return Err(polars_err!(
                ComputeError: "Expected image/array values as list/array, got {:?}",
                value
            ))
        }
    };

    let height = rows.len();
    let width = rows.first().map_or(0, Vec::len);
    let data: Vec<f64> = rows.into_iter().flatten().collect();

    Ok(ViewBuffer::from_vec_with_shape(
        data,
        vec![height, width, 1],
    ))
}

fn float_list_anyvalue(values: &[f64], name: PlSmallStr) -> AnyValue<'static> {
    AnyValue::List(Series::new(name, values.to_vec()))
}

fn matrix_anyvalue(matrix: &[Vec<f64>]) -> PolarsResult<AnyValue<'static>> {
    let rows: Vec<AnyValue> = matrix
        .iter()
        .map(|row| float_list_anyvalue(row, PlSmallStr::from_static("iou_row")))
        .collect();
    let inner_dtype = DataType::List(Box::new(DataType::Float64));
    let row_series = Series::from_any_values_and_dtype(
        PlSmallStr::from_static("iou_rows"),
        &rows,
        &inner_dtype,
        false,
    )?;
    Ok(AnyValue::List(row_series))
}

fn build_pairwise_matrix_series(
    name: PlSmallStr,
    rows: Vec<AnyValue<'static>>,
) -> PolarsResult<Series> {
    let dtype = DataType::List(Box::new(DataType::List(Box::new(DataType::Float64))));
    Series::from_any_values_and_dtype(name, &rows, &dtype, true)
}

fn pairwise_iou_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let name = input_fields
        .first()
        .map(|f| f.name().clone())
        .unwrap_or_else(|| PlSmallStr::from_static("pairwise_iou"));
    Ok(Field::new(
        name,
        DataType::List(Box::new(DataType::List(Box::new(DataType::Float64)))),
    ))
}

fn optional_u32_list_anyvalue(values: &[Option<u32>], name: PlSmallStr) -> AnyValue<'static> {
    let series = UInt32Chunked::from_iter_options(name, values.iter().copied()).into_series();
    AnyValue::List(series)
}

/// The `{right_idx, overlap}` struct a correspondence result publishes.
///
/// Declared once, and read by both the output-type function and the row
/// builder below. What this replaced spelled its schema out three times per
/// entry point — six copies between the two — which is precisely how the
/// dtype a query planned against and the value it produced were free to
/// drift apart.
fn correspondence_fields() -> Vec<Field> {
    vec![
        Field::new(
            PlSmallStr::from_static("right_idx"),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            PlSmallStr::from_static("overlap"),
            DataType::List(Box::new(DataType::Float64)),
        ),
    ]
}

fn correspondence_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let name = input_fields
        .first()
        .map(|f| f.name().clone())
        .unwrap_or_else(|| PlSmallStr::from_static("correspond"));
    Ok(Field::new(name, DataType::Struct(correspondence_fields())))
}

fn correspondence_anyvalue(result: &pairwise::Correspondence) -> AnyValue<'static> {
    let right: Vec<Option<u32>> = result
        .right_idx
        .iter()
        .map(|v| v.map(|x| x as u32))
        .collect();
    AnyValue::StructOwned(Box::new((
        vec![
            optional_u32_list_anyvalue(&right, PlSmallStr::from_static("right_idx")),
            float_list_anyvalue(&result.overlap, PlSmallStr::from_static("overlap")),
        ],
        correspondence_fields(),
    )))
}

/// Parse a per-row walk order: a permutation of `0..n_left`.
///
/// Strict on purpose. A short, long, duplicated or out-of-range order is a
/// caller mistake that would otherwise show up as elements silently never
/// visited — the class of quiet degradation this codebase rejects — so it
/// fails here, naming the row.
fn parse_order_list(value: &AnyValue, n_left: usize, row: usize) -> PolarsResult<Vec<usize>> {
    let AnyValue::List(series) = value else {
        polars_bail!(ComputeError: "Expected List[UInt32] for order, got {:?}", value);
    };
    if series.len() != n_left {
        polars_bail!(ComputeError:
            "order length ({}) must match the number of left elements ({}) in row {}",
            series.len(), n_left, row
        );
    }
    let mut order = Vec::with_capacity(series.len());
    let mut seen = vec![false; n_left];
    for i in 0..series.len() {
        let item = series.get(i)?;
        let index = item.try_extract::<u32>().map_err(|_| {
            polars_err!(ComputeError:
                "order must contain non-negative integers, found {:?} in row {}", item, row)
        })? as usize;
        if index >= n_left {
            polars_bail!(ComputeError:
                "order entry {} is out of range for {} left elements in row {}",
                index, n_left, row
            );
        }
        if std::mem::replace(&mut seen[index], true) {
            polars_bail!(ComputeError:
                "order repeats entry {} in row {}; it must be a permutation", index, row
            );
        }
        order.push(index);
    }
    Ok(order)
}

/// Drive one correspondence entry point over its rows.
///
/// The contour and bbox accessors differ only in how a row parses and how its
/// overlap matrix is built, so those are the two parameters; everything else —
/// the null handling, the per-row threshold, the order, the output struct —
/// is shared. The two functions this replaced were near-verbatim duplicates.
fn correspond_rows<T>(
    inputs: &[Series],
    params: &GeomParams,
    (other, threshold, order): (&ColumnRef, &Param<f64>, &Option<ColumnRef>),
    parse: impl Fn(&AnyValue) -> PolarsResult<Vec<T>>,
    build_matrix: impl Fn(&[T], &[T]) -> Vec<Vec<f64>>,
) -> PolarsResult<Series> {
    let left_series = &inputs[0];
    // Both operands are read through their references: `order` is optional,
    // so nothing here may read a fixed position.
    let right_series = params.column(other);
    let order_series = params.optional_column(order);
    let len = left_series.len();
    let dtype = DataType::Struct(correspondence_fields());

    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);
    for i in 0..len {
        let left_value = left_series.get(i)?;
        let right_value = right_series.get(i)?;
        if left_value.is_null() || right_value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }

        // Per-row parameters cannot be range-checked once per batch, so the
        // check moves into the loop and names the offending row. A null
        // `threshold` under `on_null="null"` nulls this row instead.
        let Some(threshold) = params.row(|| {
            let threshold = params.value(threshold, i)?;
            check_range("threshold", threshold, 0.0, 1.0, i)?;
            Ok(threshold)
        })?
        else {
            rows.push(AnyValue::Null);
            continue;
        };

        let left = parse(&left_value)?;
        let right = parse(&right_value)?;

        let order = match order_series {
            Some(column) => {
                let value = column.get(i)?;
                if value.is_null() {
                    None
                } else {
                    Some(parse_order_list(&value, left.len(), i)?)
                }
            }
            None => None,
        };

        let result =
            pairwise::greedy_assign(&build_matrix(&left, &right), threshold, order.as_deref());
        rows.push(correspondence_anyvalue(&result));
    }

    Series::from_any_values_and_dtype(left_series.name().clone(), &rows, &dtype, true)
}

fn label_reduce_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let name = input_fields
        .first()
        .map(|f| f.name().clone())
        .unwrap_or_else(|| PlSmallStr::from_static("label_reduce"));
    Ok(Field::new(
        name,
        DataType::List(Box::new(DataType::Float64)),
    ))
}

// ============================================================================
// Contour Plugin Functions - Measures
// ============================================================================

// The bbox `{x, y, width, height}` struct authority (`bbox_struct_dtype`,
// `bbox_anyvalue`) and the per-row `parse_bbox` live in `geom_schema`, shared
// with the point namespace so the two cannot spell the wire format differently.

contour_accessor! {
    /// Compute contour area.
    map fn contour_area / contour_area_output_type -> |_input| DataType::Float64;
    parse GeometryOp::Area { signed };
    |contour, params, row| {
        let signed = params.value(signed, row)?;
        Ok(AnyValue::Float64(measures::area(contour, signed)))
    }
}

contour_accessor! {
    /// Compute contour perimeter.
    map fn contour_perimeter / contour_perimeter_output_type -> |_input| DataType::Float64;
    parse GeometryOp::Perimeter;
    |contour, _params, _row| Ok(AnyValue::Float64(measures::perimeter(contour)))
}

contour_accessor! {
    /// Compute winding direction.
    map fn contour_winding / contour_winding_output_type -> |_input| DataType::String;
    parse ContourFn::Winding;
    |contour, _params, _row| Ok(AnyValue::StringOwned(
        match measures::contour_winding(contour) {
            Winding::CounterClockwise => "ccw",
            Winding::Clockwise => "cw",
        }
        .into(),
    ))
}

contour_accessor! {
    /// Compute contour centroid — a `{x, y}` struct per contour.
    map fn contour_centroid / contour_centroid_output_type -> |_input| point_struct_dtype();
    parse GeometryOp::Centroid;
    |contour, _params, _row| {
        let center = measures::centroid(contour);
        Ok(point_anyvalue(center.x, center.y))
    }
}

contour_accessor! {
    /// Compute contour bounding box — an `{x, y, width, height}` struct per contour.
    map fn contour_bounding_box / contour_bounding_box_output_type -> |_input| bbox_struct_dtype();
    parse GeometryOp::BoundingBox;
    |contour, _params, _row| Ok(bbox_anyvalue(measures::bounding_box(contour)))
}

// ============================================================================
// Contour Plugin Functions - Predicates
// ============================================================================

contour_accessor! {
    /// Check if contour is convex.
    map fn contour_is_convex / contour_is_convex_output_type -> |_input| DataType::Boolean;
    parse ContourFn::IsConvex;
    |contour, _params, _row| Ok(AnyValue::Boolean(predicates::contour_is_convex(contour)))
}

/// Read one `{x, y}` struct value.
fn parse_point_value(point_value: &AnyValue) -> PolarsResult<(f64, f64)> {
    {
        {
            // Parse point from struct
            let (x, y) = match point_value {
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
            Ok((x, y))
        }
    }
}

/// Declared type of `contour_contains_point`: one bool per contour.
fn contour_contains_point_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    elementwise_field(input_fields, "contour_contains_point", DataType::Boolean)
}

/// Check if contour contains a specific point.
///
/// The one accessor with its own row loop. Its second operand is a *point*, not
/// a contour, so neither the `map` arm (one operand) nor the `zip` arm (two
/// contour operands, broadcast) describes it: the arity comes from the contour
/// column alone, while a null point nulls the whole row the way `zip_contours`
/// does rather than each element.
///
/// It still reads the arity through [`Arity::of`] and wraps through
/// [`elementwise_field`] / [`ContourOutput`], so the *decision* and the *wrapping*
/// stay single-authority — only the loop is local.
#[polars_expr(output_type_func=contour_contains_point_output_type)]
fn contour_contains_point(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "contour_contains_point";
    let (op, params) = GeomParams::parse::<ContourFn<Wire>>(inputs, kwargs, NAME)?;
    let ContourFn::ContainsPoint { point } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let contour_series = &inputs[0];
    let point_series = params.column(point);
    let arity = Arity::of(contour_series.dtype());
    let len = contour_series.len();
    let mut rows: Vec<Option<Vec<AnyValue<'static>>>> = Vec::with_capacity(len);

    for i in 0..len {
        let contour_value = contour_series.get(i)?;
        let point_value = point_series.get(i)?;
        if contour_value.is_null() || point_value.is_null() {
            rows.push(None);
            continue;
        }
        let (x, y) = parse_point_value(&point_value)?;
        let results = row_contours(&contour_value, arity)?
            .iter()
            .map(|contour| AnyValue::Boolean(predicates::contains_point(contour, x, y)))
            .collect();
        rows.push(Some(results));
    }

    AnyValue::column(
        contour_series.name().clone(),
        rows,
        arity,
        &DataType::Boolean,
    )
}

// ============================================================================
// Contour Plugin Functions - Pairwise Comparisons
// ============================================================================

/// Compute full pairwise IoU matrix between two contour sets.
#[polars_expr(output_type_func=pairwise_iou_output_type)]
fn contour_pairwise_iou(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "contour_pairwise_iou";
    let (op, params) = GeomParams::parse::<ContourFn<Wire>>(inputs, kwargs, NAME)?;
    let ContourFn::PairwiseIou { other } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let pred_series = &inputs[0];
    let gt_series = params.column(other);
    let len = pred_series.len();
    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);

    for i in 0..len {
        let preds_value = pred_series.get(i)?;
        let gts_value = gt_series.get(i)?;
        if preds_value.is_null() || gts_value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }

        let preds = parse_contour_set(&preds_value)?;
        let gts = parse_contour_set(&gts_value)?;
        let matrix = pairwise::iou_matrix(&preds, &gts);
        rows.push(matrix_anyvalue(&matrix)?);
    }

    build_pairwise_matrix_series(pred_series.name().clone(), rows)
}

/// One-to-one correspondence between two contour sets by overlap.
///
/// Generic by construction: it knows about contours and overlap, and nothing
/// about detections, confidence or true positives. `order` is a permutation
/// naming the visit sequence; supplying one derived from confidence is what
/// makes this a detection matcher, and that derivation belongs to the caller.
#[polars_expr(output_type_func=correspondence_output_type)]
fn contour_correspond(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "contour_correspond";
    let (op, params) = GeomParams::parse::<ContourFn<Wire>>(inputs, kwargs, NAME)?;
    let ContourFn::Correspond {
        other,
        threshold,
        order,
    } = &op
    else {
        return Err(parsed_as_another(NAME));
    };
    correspond_rows(
        inputs,
        &params,
        (other, threshold, order),
        parse_contour_set,
        pairwise::iou_matrix,
    )
}

/// Score each contour against a heatmap using a configurable reduction.
///
/// Delegates to the engine's [`score_contours_on_buffer`] — the same function
/// `Pipeline.label_reduce` reaches through the graph — so the two entry points
/// share their region modes, their reductions and their empty-region fallback.
#[polars_expr(output_type_func=label_reduce_output_type)]
fn contour_label_reduce(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "contour_label_reduce";
    let (op, params) = GeomParams::parse::<ContourFn<Wire>>(inputs, kwargs, NAME)?;
    let ContourFn::LabelReduce {
        image,
        reduction,
        region_mode,
    } = &op
    else {
        return Err(parsed_as_another(NAME));
    };
    let contour_series = &inputs[0];
    let heatmap_series = params.column(image);
    let len = contour_series.len();
    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);

    for i in 0..len {
        let contours_value = contour_series.get(i)?;
        let heatmap_value = heatmap_series.get(i)?;
        if contours_value.is_null() || heatmap_value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }

        // Per-row capable, matching `Pipeline.label_reduce`: neither choice
        // affects the output's shape or dtype.
        let Some((reduction, region_mode)) = params.row(|| {
            let reduction = params.value(reduction, i)?;
            let region_mode = params.value(region_mode, i)?;
            Ok((reduction, region_mode))
        })?
        else {
            rows.push(AnyValue::Null);
            continue;
        };

        let contours = parse_contour_set(&contours_value)?;
        let heatmap = parse_heatmap(&heatmap_value)?;
        let scores = score_contours_on_buffer(&heatmap, &contours, reduction, region_mode)
            .map_err(|err| polars_err!(ComputeError: "{}", err))?;
        rows.push(float_list_anyvalue(
            &scores,
            PlSmallStr::from_static("scores"),
        ));
    }

    let dtype = DataType::List(Box::new(DataType::Float64));
    Series::from_any_values_and_dtype(contour_series.name().clone(), &rows, &dtype, true)
}

contour_accessor! {
    /// Compute IoU between two contours, broadcasting a set against a single.
    zip fn contour_iou / contour_iou_output_type -> DataType::Float64;
    parse ContourFn::Iou { other };
    |a, b| Ok(AnyValue::Float64(pairwise::iou(a, b)))
}

contour_accessor! {
    /// Compute Dice coefficient between two contours.
    zip fn contour_dice / contour_dice_output_type -> DataType::Float64;
    parse ContourFn::Dice { other };
    |a, b| Ok(AnyValue::Float64(pairwise::dice(a, b)))
}

contour_accessor! {
    /// Compute Hausdorff distance between two contours.
    zip fn contour_hausdorff / contour_hausdorff_output_type -> DataType::Float64;
    parse ContourFn::Hausdorff { other };
    |a, b| Ok(AnyValue::Float64(pairwise::hausdorff_distance(a, b)))
}

// ============================================================================
// Contour Plugin Functions - Transforms
// ============================================================================

// A transform's element type is "a contour of whatever shape came in", which
// `Arity::elem_dtype` answers for both arities from one reading. The old
// `contour_transform_output_type` returned the input field verbatim — correct
// for the declaration, but its body then handed the *outer* dtype to
// `build_contour_series`, so a set could never have been built.

contour_accessor! {
    /// Translate contour by offset.
    map fn contour_translate / contour_translate_output_type
        -> |input| Arity::elem_dtype(input);
    parse GeometryOp::Translate { dx, dy };
    |contour, params, row| {
        let dx = params.value(dx, row)?;
        let dy = params.value(dy, row)?;
        Ok(transforms::translate(contour, dx, dy))
    }
}

contour_accessor! {
    /// Scale contour.
    map fn contour_scale / contour_scale_output_type
        -> |input| Arity::elem_dtype(input);
    parse GeometryOp::Scale { sx, sy, origin };
    |contour, params, row| {
        let sx = params.value(sx, row)?;
        let sy = params.value(sy, row)?;
        // Per-row capable, like `sx`/`sy` beside it: which point the scale
        // is measured from does not change the output's shape, rank or
        // dtype. This is the pipeline op's own definition, default and all.
        let scale_origin = params.value(origin, row)?;
        Ok(transforms::scale(contour, sx, sy, scale_origin))
    }
}

contour_accessor! {
    /// Simplify contour.
    map fn contour_simplify / contour_simplify_output_type
        -> |input| Arity::elem_dtype(input);
    parse GeometryOp::Simplify { tolerance };
    |contour, params, row| {
        let tolerance = params.value(tolerance, row)?;
        Ok(transforms::simplify(contour, tolerance))
    }
}

contour_accessor! {
    /// Flip contour (reverse winding).
    map fn contour_flip / contour_flip_output_type -> |input| Arity::elem_dtype(input);
    parse ContourFn::Flip;
    |contour, _params, _row| Ok(transforms::flip(contour))
}

contour_accessor! {
    /// Compute convex hull.
    map fn contour_convex_hull / contour_convex_hull_output_type
        -> |input| Arity::elem_dtype(input);
    parse GeometryOp::ConvexHull;
    |contour, _params, _row| Ok(transforms::convex_hull(contour))
}

contour_accessor! {
    /// Normalize contour coordinates to [0, 1] range.
    map fn contour_normalize / contour_normalize_output_type
        -> |input| Arity::elem_dtype(input);
    parse ContourFn::Normalize { width, height };
    |contour, params, row| {
        let ref_width = params.value(width, row)?;
        let ref_height = params.value(height, row)?;
        Ok(transforms::normalize(contour, ref_width, ref_height))
    }
}

contour_accessor! {
    /// Convert normalized coordinates to absolute pixel coordinates.
    map fn contour_to_absolute / contour_to_absolute_output_type
        -> |input| Arity::elem_dtype(input);
    parse ContourFn::ToAbsolute { width, height };
    |contour, params, row| {
        let ref_width = params.value(width, row)?;
        let ref_height = params.value(height, row)?;
        Ok(transforms::to_absolute(contour, ref_width, ref_height))
    }
}

contour_accessor! {
    /// Ensure contour has specified winding direction.
    map fn contour_ensure_winding / contour_ensure_winding_output_type
        -> |input| Arity::elem_dtype(input);
    parse ContourFn::EnsureWinding { direction };
    |contour, params, row| {
        // Per-row capable: the winding a ring is rewound to changes the
        // vertex order, not the output's shape, rank or dtype.
        //
        // `direction` is required, so there is no default to fall back to.
        // The match this replaced fell back to counter-clockwise for
        // anything it did not recognise, which meant `ensure_winding("CW")`
        // returned the *opposite* of what was asked for, silently.
        let direction = params.value(direction, row)?;
        Ok(transforms::ensure_winding(contour, direction))
    }
}

// ============================================================================
// BBox Matching Plugin Functions
// ============================================================================

// A single bbox struct parses via `geom_schema::parse_bbox` (imported above) --
// the one per-row bbox parser, shared with the point namespace.

/// Parse a List[BBOX_SCHEMA] AnyValue into a Vec<BoundingBox>.
fn parse_bbox_list(value: &AnyValue) -> PolarsResult<Vec<BoundingBox>> {
    match value {
        AnyValue::List(series) => {
            if let Ok(struct_ca) = series.struct_() {
                let x_col = struct_ca
                    .field_by_name("x")
                    .map_err(|_| polars_err!(ComputeError: "Bbox struct missing 'x' field"))?;
                let y_col = struct_ca
                    .field_by_name("y")
                    .map_err(|_| polars_err!(ComputeError: "Bbox struct missing 'y' field"))?;
                let w_col = struct_ca
                    .field_by_name("width")
                    .map_err(|_| polars_err!(ComputeError: "Bbox struct missing 'width' field"))?;
                let h_col = struct_ca
                    .field_by_name("height")
                    .map_err(|_| polars_err!(ComputeError: "Bbox struct missing 'height' field"))?;

                let x_ca = x_col
                    .f64()
                    .map_err(|_| polars_err!(ComputeError: "x must be f64"))?;
                let y_ca = y_col
                    .f64()
                    .map_err(|_| polars_err!(ComputeError: "y must be f64"))?;
                let w_ca = w_col
                    .f64()
                    .map_err(|_| polars_err!(ComputeError: "width must be f64"))?;
                let h_ca = h_col
                    .f64()
                    .map_err(|_| polars_err!(ComputeError: "height must be f64"))?;

                let mut bboxes = Vec::with_capacity(series.len());
                for i in 0..series.len() {
                    bboxes.push(BoundingBox::new(
                        x_ca.get(i).unwrap_or(0.0),
                        y_ca.get(i).unwrap_or(0.0),
                        w_ca.get(i).unwrap_or(0.0),
                        h_ca.get(i).unwrap_or(0.0),
                    ));
                }
                Ok(bboxes)
            } else {
                let mut bboxes = Vec::with_capacity(series.len());
                for i in 0..series.len() {
                    let item = series.get(i)?;
                    bboxes.push(parse_bbox(&item)?);
                }
                Ok(bboxes)
            }
        }
        _ => Err(polars_err!(ComputeError: "Expected List of bbox structs, got {:?}", value)),
    }
}

/// Pairwise IoU matrix between two sets of bounding boxes.
#[polars_expr(output_type_func=pairwise_iou_output_type)]
fn bbox_pairwise_iou(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "bbox_pairwise_iou";
    let (op, params) = GeomParams::parse::<BBoxFn<Wire>>(inputs, kwargs, NAME)?;
    let BBoxFn::PairwiseIou { other } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let pred_series = &inputs[0];
    let gt_series = params.column(other);
    let len = pred_series.len();
    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);

    for i in 0..len {
        let preds_value = pred_series.get(i)?;
        let gts_value = gt_series.get(i)?;
        if preds_value.is_null() || gts_value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }

        let preds = parse_bbox_list(&preds_value)?;
        let gts = parse_bbox_list(&gts_value)?;
        let matrix = pairwise::bbox_iou_matrix(&preds, &gts);
        rows.push(matrix_anyvalue(&matrix)?);
    }

    build_pairwise_matrix_series(pred_series.name().clone(), rows)
}

/// One-to-one correspondence between two bbox sets by overlap.
///
/// The bbox half of [`contour_correspond`], sharing its driver so the two
/// cannot disagree about the rule, the threshold, the order or the output.
#[polars_expr(output_type_func=correspondence_output_type)]
fn bbox_correspond(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "bbox_correspond";
    let (op, params) = GeomParams::parse::<BBoxFn<Wire>>(inputs, kwargs, NAME)?;
    let BBoxFn::Correspond {
        other,
        threshold,
        order,
    } = &op
    else {
        return Err(parsed_as_another(NAME));
    };
    correspond_rows(
        inputs,
        &params,
        (other, threshold, order),
        parse_bbox_list,
        pairwise::bbox_iou_matrix,
    )
}

#[cfg(test)]
mod parse_contour_tests {
    //! `parse_contour` is the single Struct/List -> Contour parser for the
    //! whole plugin (contour source decoding, point-namespace ops, and the
    //! contour namespace itself route through it). These tests pin its full
    //! contract so the consumers cannot re-diverge.

    use super::*;

    fn square_with_hole() -> Contour {
        Contour::with_holes(
            vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(0.0, 10.0),
            ],
            vec![vec![
                Point::new(4.0, 4.0),
                Point::new(6.0, 4.0),
                Point::new(6.0, 6.0),
                Point::new(4.0, 6.0),
            ]],
        )
    }

    #[test]
    fn parse_contour_struct_with_holes_round_trips() {
        let contour = square_with_hole();
        let av = contour_to_anyvalue(&contour);
        let parsed = parse_contour(&av).expect("round trip must parse");
        assert_eq!(parsed.exterior, contour.exterior);
        assert_eq!(parsed.holes, contour.holes);
    }

    #[test]
    fn parse_contour_bare_list() {
        // A bare List[{x, y}] (no wrapping struct) is a valid simple contour.
        let av = contour_to_anyvalue(&square_with_hole());
        let AnyValue::StructOwned(boxed) = av else {
            panic!("contour_to_anyvalue must build a struct");
        };
        let (values, _) = *boxed;
        let exterior_list = values[0].clone();
        assert!(matches!(exterior_list, AnyValue::List(_)));
        let parsed = parse_contour(&exterior_list).expect("bare list must parse");
        assert_eq!(parsed.exterior.len(), 4);
        assert!(parsed.holes.is_empty());
    }

    #[test]
    fn a_struct_without_an_exterior_is_refused() {
        // The ring is the `exterior` field: a struct that merely looks like a
        // contour (a `points` field, or any first list field) is not guessed
        // at. EXTENSION_TYPES_PLAN.md §3.5.
        let av = contour_to_anyvalue(&square_with_hole());
        let AnyValue::StructOwned(boxed) = av else {
            panic!("contour_to_anyvalue must build a struct");
        };
        let (values, mut fields) = *boxed;
        fields[0] = Field::new(PlSmallStr::from_static("points"), fields[0].dtype().clone());
        let renamed = AnyValue::StructOwned(Box::new((values, fields)));
        let err = parse_contour(&renamed).expect_err("a 'points' field is not an exterior");
        assert!(err.to_string().contains("exterior"), "{err}");
    }

    /// Build a `List` AnyValue over `dtype` from the given elements.
    fn list_of(elements: Vec<AnyValue<'static>>, dtype: &DataType) -> AnyValue<'static> {
        let series =
            Series::from_any_values_and_dtype(PlSmallStr::from_static("s"), &elements, dtype, true)
                .expect("test list must build");
        AnyValue::List(series)
    }

    /// The dtype `extract_contours().sink("native")` emits, element-wise.
    fn contour_dtype() -> DataType {
        // Reads the authority rather than restating it. The whole
        // `{exterior, holes, is_closed}` layout lives in
        // `geom_schema::contour_fields`; a second spelling here would only add
        // a place for a rename to miss. Python's `CONTOUR_SCHEMA` is held to
        // the same authority by `test_contour_schema_matches_the_rust_declaration`.
        DataType::Struct(crate::geom_schema::contour_fields())
    }

    #[test]
    fn parse_contour_set_reads_a_list_of_contours() {
        let a = square_with_hole();
        let b = Contour::new(vec![
            Point::new(20.0, 20.0),
            Point::new(30.0, 20.0),
            Point::new(30.0, 30.0),
        ]);
        let value = list_of(
            vec![contour_to_anyvalue(&a), contour_to_anyvalue(&b)],
            &contour_dtype(),
        );

        let parsed = parse_contour_set(&value).expect("a contour set must parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].holes.len(), 1);
        assert_eq!(parsed[1].exterior.len(), 3);
    }

    #[test]
    fn parse_contour_set_reads_a_bare_ring_as_one_contour() {
        // The dispatch is by element dtype: a list of *points* is one contour's
        // ring, not a set. Guessing here is what surfaced as "Point struct
        // missing 'x' field" when a genuine set arrived.
        let av = contour_to_anyvalue(&square_with_hole());
        let AnyValue::StructOwned(boxed) = av else {
            panic!("contour_to_anyvalue must build a struct");
        };
        let (values, _) = *boxed;
        let parsed = parse_contour_set(&values[0]).expect("a ring must parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].exterior.len(), 4);
    }

    #[test]
    fn parse_contour_set_reads_a_lone_struct_as_a_set_of_one() {
        let contour = square_with_hole();
        let parsed =
            parse_contour_set(&contour_to_anyvalue(&contour)).expect("a struct must parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].exterior, contour.exterior);
    }

    #[test]
    fn parse_contour_set_reads_null_and_empty_as_the_empty_set() {
        assert!(parse_contour_set(&AnyValue::Null)
            .expect("null must parse")
            .is_empty());
        assert!(parse_contour_set(&list_of(vec![], &contour_dtype()))
            .expect("an empty list must parse")
            .is_empty());
    }

    #[test]
    fn parse_contour_missing_exterior_errors() {
        // A struct with no list field cannot be a contour; the error names
        // the expected fields (shared verbatim by every consumer).
        let bogus = AnyValue::StructOwned(Box::new((
            vec![AnyValue::Float64(1.0)],
            vec![Field::new(
                PlSmallStr::from_static("not_a_contour"),
                DataType::Float64,
            )],
        )));
        let err = parse_contour(&bogus).expect_err("must reject").to_string();
        assert!(err.contains("'exterior'"), "{err}");
    }
}

#[cfg(test)]
mod named_param_tests {
    //! The string parameters and the argument/input binding, through the one
    //! parse every geometry function makes.
    //!
    //! `ensure_winding` and `scale(origin=)` used to parse by hand and end in
    //! `_ => <default>`, so `ensure_winding("CW")` returned *counter*-clockwise
    //! — the opposite of the request — and `scale(origin="top_left")` scaled
    //! about the centroid. Both silently. They are now typed fields of their
    //! definitions, read through `NAMED` like every other string parameter.

    use super::*;
    use view_buffer::geometry::ops::ScaleOrigin;
    use view_buffer::mode::WireOps;

    fn parse<'a, F: WireOps>(
        inputs: &'a [Series],
        name: &str,
        args: serde_json::Value,
    ) -> PolarsResult<(F, GeomParams<'a>)> {
        let kwargs: GeomKwargs =
            serde_json::from_value(serde_json::json!({"args": args, "on_null": "raise"}))
                .expect("the kwargs envelope parses");
        GeomParams::parse::<F>(inputs, kwargs, name)
    }

    fn one_column() -> [Series; 1] {
        [Series::new("c".into(), &[0i32])]
    }

    fn winding(name: &str) -> PolarsResult<ContourFn<Wire>> {
        let inputs = one_column();
        parse::<ContourFn<Wire>>(
            &inputs,
            "contour_ensure_winding",
            serde_json::json!({"direction": name}),
        )
        .map(|(op, _)| op)
    }

    #[test]
    fn a_required_parameter_rejects_a_name_the_table_does_not_hold() {
        let err = winding("CW")
            .expect_err("a miscased spelling must be rejected, not guessed")
            .to_string();
        assert!(err.contains("CW"), "the value must be named: {err}");
        assert!(
            err.contains("ccw") && err.contains("cw"),
            "the accepted spellings must be listed: {err}"
        );
    }

    #[test]
    fn a_required_parameter_has_no_default_to_fall_back_to() {
        let inputs = one_column();
        let err =
            parse::<ContourFn<Wire>>(&inputs, "contour_ensure_winding", serde_json::json!({}))
                .err()
                .expect("an absent required field is refused")
                .to_string();
        assert!(err.contains("direction"), "{err}");
    }

    #[test]
    fn the_long_winding_spellings_resolve_to_the_short_ones() {
        // Aliases in `NAMED` rather than a second table: the plugin has always
        // accepted these, and dropping them to tidy the list would have removed
        // working behaviour.
        for (name, expected) in [
            ("ccw", Winding::CounterClockwise),
            ("counterclockwise", Winding::CounterClockwise),
            ("cw", Winding::Clockwise),
            ("clockwise", Winding::Clockwise),
        ] {
            assert_eq!(
                winding(name).unwrap(),
                ContourFn::EnsureWinding {
                    direction: Param::Lit(expected)
                },
                "{name}"
            );
        }
    }

    #[test]
    fn every_scale_origin_resolves_and_an_unknown_one_does_not() {
        let inputs = one_column();
        let scale = |origin: &str| {
            parse::<GeometryOp<Wire>>(
                &inputs,
                "contour_scale",
                serde_json::json!({"sx": 2.0, "sy": 2.0, "origin": origin}),
            )
            .map(|(op, _)| op)
        };
        for (name, expected) in [
            ("centroid", ScaleOrigin::Centroid),
            ("bbox_center", ScaleOrigin::BBoxCenter),
            ("origin", ScaleOrigin::Origin),
        ] {
            let Ok(GeometryOp::Scale { origin, .. }) = scale(name) else {
                panic!("'{name}' parses as a scale");
            };
            assert_eq!(origin, Param::Lit(expected), "{name}");
        }
        let err = scale("top_left")
            .expect_err("a plausible name from another library must be rejected")
            .to_string();
        assert!(
            err.contains("top_left") && err.contains("bbox_center"),
            "{err}"
        );
    }

    /// `.contour.scale` is the pipeline op `contour_scale`, so the two cannot
    /// declare different defaults: the one declared is the centroid.
    #[test]
    fn the_accessor_and_the_op_share_one_origin_default() {
        let scale = crate::geom_fns::geom_catalog()
            .into_iter()
            .find(|d| d.namespace == "contour" && d.function.python == "scale")
            .expect("`.contour.scale` is catalogued");
        assert_eq!(scale.function.name, "contour_scale");
        let origin = scale
            .function
            .fields
            .iter()
            .find(|f| f.name == "origin")
            .unwrap();
        assert_eq!(origin.default, Some(serde_json::json!("centroid")));
    }

    /// Every input past the namespace's own column must be read by exactly
    /// one field: an unclaimed one is an operand that was dropped.
    #[test]
    fn every_extra_input_must_be_claimed_by_a_field() {
        let inputs = [
            Series::new("c".into(), &[0i32]),
            Series::new("w".into(), &[2.0f64]),
        ];
        let normalize = |width: serde_json::Value| {
            parse::<ContourFn<Wire>>(
                &inputs,
                "contour_normalize",
                serde_json::json!({"width": width, "height": 1.0}),
            )
            .map(|_| ())
        };
        assert!(normalize(serde_json::json!({"$slot": 1})).is_ok());
        let err = normalize(serde_json::json!(2.0)).unwrap_err().to_string();
        assert!(err.contains("exactly once"), "{err}");
        let err = normalize(serde_json::json!({"$slot": 2}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("'width' reads input 2"), "{err}");
    }
}
