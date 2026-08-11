//! Contour plugin functions for polars-cv.
//!
//! This module provides Polars expression functions for contour geometry operations,
//! including measures (area, perimeter), predicates (is_convex, contains_point),
//! transforms (translate, scale, simplify), and pairwise comparisons (IoU, Dice).

use polars::prelude::*;
use polars_arrow::array::{ListArray, PrimitiveArray, StructArray as ArrowStructArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::cmp::Ordering;

// Import geometry operations from view-buffer
use view_buffer::geometry::{
    contour::{BoundingBox, Contour, Point, Winding},
    label::{score_contours_on_buffer, LabelReduction, LabelRegionMode},
    measures,
    ops::ScaleOrigin,
    pairwise, predicates, transforms,
};
use view_buffer::{naming, ViewBuffer};

use crate::geom_params::{check_range, GeomParams, InputSlots};
use crate::params::NullParamPolicy;

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
            AnyValue::Boolean(true), // is_closed: reserved, never read back
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

/// Run one row of a contour operation and record its result.
///
/// Two different things make a row null: a null input contour
/// (`input_is_null`), and — under `on_null="null"` — a null per-row parameter.
/// Routing both through [`GeomParams::row`] here keeps the second from being
/// re-implemented in every operation below.
fn contour_row<T>(
    params: &GeomParams,
    input_is_null: bool,
    results: &mut Vec<Option<T>>,
    compute: impl FnOnce() -> PolarsResult<T>,
) -> PolarsResult<()> {
    let value = if input_is_null {
        None
    } else {
        params.row(compute)?
    };
    results.push(value);
    Ok(())
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
    /// IoU threshold for detection matching.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Matching strategy name.
    #[serde(default)]
    pub strategy: Option<String>,
    /// Reduction method for label scoring.
    #[serde(default)]
    pub reduction: Option<String>,
    /// Region mode for label scoring.
    #[serde(default)]
    pub region_mode: Option<String>,
    /// Maps a named input — data operand or per-row parameter — to its index
    /// in `inputs`. A parameter absent from the map is literal, read from the
    /// scalar fields above. Every input beyond the namespace's own column at
    /// index 0 must appear here; `GeomParams::new` rejects a map that does not
    /// account for all of them, so a stale caller fails loudly instead of
    /// silently dropping an operand.
    #[serde(default)]
    pub input_slots: InputSlots,
    /// What a null in a per-row parameter column means for that row: `raise`
    /// (default) fails the expression, `null` yields a null result for the
    /// affected rows. Set from Python by `_PluginNamespace.on_null` and applied
    /// by `GeomParams::row`.
    #[serde(default)]
    pub on_null: NullParamPolicy,
}

/// Parse a contour from a Polars value.
///
/// The **single** Struct/List -> `Contour` parser for the whole plugin:
/// the contour namespace, the point namespace (`point.rs`), and the contour
/// source decoder (`execute.rs`) all route through it, so hole handling,
/// accepted input forms, and error text cannot diverge between consumers
/// (pinned by `parse_contour_tests` and `tests/test_contour_parsing.py`).
///
/// Accepted forms:
/// - a struct matching `{exterior: List[{x, y}], holes: List[List[{x, y}]]}`
///   (`points` is accepted as an alias for `exterior`; with neither present,
///   the first list field is used as the exterior)
/// - a bare `List[{x, y}]` (a simple contour without holes)
pub(crate) fn parse_contour(value: &AnyValue) -> PolarsResult<Contour> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();
            let mut exterior: Option<Vec<Point>> = None;
            let mut holes: Vec<Vec<Point>> = Vec::new();

            for (i, field) in fields.iter().enumerate() {
                match field.name().as_str() {
                    "exterior" | "points" => {
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

            let exterior = if let Some(points) = exterior {
                points
            } else {
                // Backward-compatible fallback: first list field as exterior.
                let mut fallback: Option<Vec<Point>> = None;
                for av in values.iter() {
                    if let AnyValue::List(series) = av {
                        fallback = Some(extract_points_from_series(series)?);
                        break;
                    }
                }
                fallback.ok_or_else(
                    || polars_err!(ComputeError: "Contour struct missing exterior/points field"),
                )?
            };

            Ok(Contour::with_holes(exterior, holes))
        }
        // Handle AnyValue::Struct (non-owned variant with row index and array reference)
        AnyValue::Struct(row_idx, struct_array, fields) => {
            let mut exterior: Option<Vec<Point>> = None;
            let mut holes: Vec<Vec<Point>> = Vec::new();

            for (i, field) in fields.iter().enumerate() {
                let column = struct_array.values()[i].clone();
                match field.name().as_str() {
                    "exterior" | "points" => {
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

            let exterior = exterior.ok_or_else(
                || polars_err!(ComputeError: "Contour struct missing exterior/points field"),
            )?;
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

/// Parse geometry that may be a single contour or a whole set of them.
///
/// The contour source's entry point, so a mask can be rasterized from either
/// shape a geometry column takes: `extract_contours().sink("native")` emits
/// `List[Contour]`, while a hand-written contour column is one `Struct` per row.
/// Both arrive here and leave as a set; the rasterizer paints their union.
///
/// The two list forms are told apart by the element dtype, not by trying one and
/// falling back: a `List` whose elements are point structs (an `x` and a `y`
/// field) is one contour's ring, anything else in a `List` is a set of contours.
/// A fallback would have to guess, and guessing wrong on a contour set is what
/// used to surface as `Point struct missing 'x' field`.
///
/// Accepted forms:
/// - `List[Contour]` — a contour set (elements are parsed by [`parse_contour`])
/// - anything [`parse_contour`] accepts — a single contour, as a one-element set
/// - null — the empty set, which rasterizes to an all-background mask
pub(crate) fn parse_contour_set(value: &AnyValue) -> PolarsResult<Vec<Contour>> {
    match value {
        AnyValue::Null => Ok(Vec::new()),
        AnyValue::List(series) if !is_point_dtype(series.dtype()) => parse_contour_list(value),
        _ => Ok(vec![parse_contour(value)?]),
    }
}

/// Does this dtype describe a point (`{x, y}`) rather than a contour?
///
/// Reads the same field names [`extract_points_from_series`] reads, so the
/// dispatch above cannot admit something the parser then rejects.
fn is_point_dtype(dtype: &DataType) -> bool {
    let DataType::Struct(fields) = dtype else {
        return false;
    };
    let named = |wanted: [&str; 2]| fields.iter().any(|f| wanted.contains(&f.name().as_str()));
    named(["x", "X"]) && named(["y", "Y"])
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
fn parse_score_list(value: &AnyValue) -> PolarsResult<Vec<f64>> {
    match value {
        AnyValue::List(series) => {
            let mut scores = Vec::with_capacity(series.len());
            for i in 0..series.len() {
                let item = series.get(i)?;
                let score = item.try_extract::<f64>().map_err(|_| {
                    polars_err!(ComputeError: "scores must contain numeric values, found {:?}", item)
                })?;
                scores.push(score);
            }
            Ok(scores)
        }
        AnyValue::Null => Ok(Vec::new()),
        _ => Err(polars_err!(ComputeError: "Expected List[Float64] for scores, got {:?}", value)),
    }
}

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

/// Resolve a string parameter against an enum's canonical `NAMED` table.
///
/// The same table the graph path and the `enum_variants` FFI read, so the two
/// `label_reduce` entry points cannot drift apart on accepted names.
fn parse_named<T: Copy>(
    table: &[(&str, T)],
    param: &str,
    value: Option<&str>,
    default: T,
) -> PolarsResult<T> {
    let Some(name) = value else {
        return Ok(default);
    };
    require_named(table, param, name)
}

/// Resolve a string parameter that has no default, rejecting anything the
/// table does not name.
///
/// Split from [`parse_named`] rather than given a sentinel default: a
/// parameter the caller must supply has no correct value to fall back to, and
/// the two silent `_ => <default>` arms this replaced are exactly what an
/// invented fallback looks like once it ships.
fn require_named<T: Copy>(table: &[(&str, T)], param: &str, name: &str) -> PolarsResult<T> {
    naming::lookup(table, name).ok_or_else(|| {
        polars_err!(
            ComputeError: "Unsupported {} '{}'. Expected one of: {}",
            param, name, naming::names(table).join(", ")
        )
    })
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

fn optional_u32_list_anyvalue(values: &[Option<u32>], name: PlSmallStr) -> AnyValue<'static> {
    let series = UInt32Chunked::from_iter_options(name, values.iter().copied()).into_series();
    AnyValue::List(series)
}

fn u32_list_anyvalue(values: &[u32], name: PlSmallStr) -> AnyValue<'static> {
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

fn match_detections_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let name = input_fields
        .first()
        .map(|f| f.name().clone())
        .unwrap_or_else(|| PlSmallStr::from_static("match"));
    let fields = vec![
        Field::new(
            PlSmallStr::from_static("pred_idx"),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            PlSmallStr::from_static("gt_idx"),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            PlSmallStr::from_static("iou"),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new(PlSmallStr::from_static("n_preds"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_gts"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_tp"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_fp"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_fn"), DataType::UInt32),
    ];
    Ok(Field::new(name, DataType::Struct(fields)))
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

fn score_order(scores: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(b))
    });
    indices
}

// ============================================================================
// Contour Plugin Functions - Measures
// ============================================================================

/// Compute contour area.
#[polars_expr(output_type=Float64)]
fn contour_area(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;

    let series = &inputs[0];
    let len = series.len();
    let mut results = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        contour_row(&params, value.is_null(), &mut results, || {
            let signed = params.bool("signed", kwargs.signed, i)?;
            let contour = parse_contour(&value)?;
            Ok(measures::area(&contour, signed))
        })?;
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

/// Compute full pairwise IoU matrix between two contour sets.
#[polars_expr(output_type_func=pairwise_iou_output_type)]
fn contour_pairwise_iou(inputs: &[Series]) -> PolarsResult<Series> {
    let pred_series = &inputs[0];
    let gt_series = &inputs[1];
    let len = pred_series.len();
    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);

    for i in 0..len {
        let preds_value = pred_series.get(i)?;
        let gts_value = gt_series.get(i)?;
        if preds_value.is_null() || gts_value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }

        let preds = parse_contour_list(&preds_value)?;
        let gts = parse_contour_list(&gts_value)?;
        let matrix = pairwise::iou_matrix(&preds, &gts);
        rows.push(matrix_anyvalue(&matrix)?);
    }

    build_pairwise_matrix_series(pred_series.name().clone(), rows)
}

/// Match detection contour sets with greedy one-to-one IoU assignment.
#[polars_expr(output_type_func=match_detections_output_type)]
fn contour_match_detections(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;
    let pred_series = &inputs[0];
    // Both operands are looked up by name. `scores` is optional, so its
    // position is not fixed once per-row parameters can occupy input slots
    // either; `other` follows the same rule so no read here is positional.
    let gt_series = params
        .slot("other")
        .map(|idx| &inputs[idx])
        .ok_or_else(|| polars_err!(ComputeError: "missing required input 'other'"))?;
    let score_series = params.slot("scores").map(|idx| &inputs[idx]);
    let len = pred_series.len();

    if let Some(strategy) = kwargs.strategy.as_deref() {
        if strategy != "greedy" {
            return Err(polars_err!(
                ComputeError: "Unsupported strategy '{}'. Expected: greedy",
                strategy
            ));
        }
    }

    let match_dtype = DataType::Struct(vec![
        Field::new(
            PlSmallStr::from_static("pred_idx"),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            PlSmallStr::from_static("gt_idx"),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            PlSmallStr::from_static("iou"),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new(PlSmallStr::from_static("n_preds"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_gts"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_tp"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_fp"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_fn"), DataType::UInt32),
    ]);

    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);
    for i in 0..len {
        let preds_value = pred_series.get(i)?;
        let gts_value = gt_series.get(i)?;
        if preds_value.is_null() || gts_value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }

        // Per-row parameters cannot be range-checked once per batch, so the
        // check moves into the loop and names the offending row. A null
        // `threshold` under `on_null="null"` nulls this row instead.
        let Some(threshold) = params.row(|| {
            let threshold = params.f64("threshold", kwargs.threshold, 0.5, i)?;
            check_range("threshold", threshold, 0.0, 1.0, i)?;
            Ok(threshold)
        })?
        else {
            rows.push(AnyValue::Null);
            continue;
        };

        let preds = parse_contour_list(&preds_value)?;
        let gts = parse_contour_list(&gts_value)?;

        let pred_order = if let Some(scores_col) = score_series {
            let score_value = scores_col.get(i)?;
            if score_value.is_null() {
                None
            } else {
                let scores = parse_score_list(&score_value)?;
                if scores.len() != preds.len() {
                    return Err(polars_err!(
                        ComputeError:
                        "scores length ({}) must match prediction count ({}) in row {}",
                        scores.len(),
                        preds.len(),
                        i
                    ));
                }
                Some(score_order(&scores))
            }
        } else {
            None
        };

        let result = pairwise::match_detections(&preds, &gts, threshold, pred_order.as_deref());

        let pred_idx_u32: Vec<u32> = result.pred_idx.iter().map(|v| *v as u32).collect();
        let gt_idx_u32: Vec<Option<u32>> =
            result.gt_idx.iter().map(|v| v.map(|x| x as u32)).collect();

        rows.push(AnyValue::StructOwned(Box::new((
            vec![
                u32_list_anyvalue(&pred_idx_u32, PlSmallStr::from_static("pred_idx")),
                optional_u32_list_anyvalue(&gt_idx_u32, PlSmallStr::from_static("gt_idx")),
                float_list_anyvalue(&result.iou, PlSmallStr::from_static("iou")),
                AnyValue::UInt32(result.n_preds as u32),
                AnyValue::UInt32(result.n_gts as u32),
                AnyValue::UInt32(result.n_tp as u32),
                AnyValue::UInt32(result.n_fp as u32),
                AnyValue::UInt32(result.n_fn as u32),
            ],
            vec![
                Field::new(
                    PlSmallStr::from_static("pred_idx"),
                    DataType::List(Box::new(DataType::UInt32)),
                ),
                Field::new(
                    PlSmallStr::from_static("gt_idx"),
                    DataType::List(Box::new(DataType::UInt32)),
                ),
                Field::new(
                    PlSmallStr::from_static("iou"),
                    DataType::List(Box::new(DataType::Float64)),
                ),
                Field::new(PlSmallStr::from_static("n_preds"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_gts"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_tp"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_fp"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_fn"), DataType::UInt32),
            ],
        ))));
    }

    Series::from_any_values_and_dtype(pred_series.name().clone(), &rows, &match_dtype, true)
}

/// Score each contour against a heatmap using a configurable reduction.
///
/// Delegates to the engine's [`score_contours_on_buffer`] — the same function
/// `Pipeline.label_reduce` reaches through the graph — so the two entry points
/// share their region modes, their reductions and their empty-region fallback.
#[polars_expr(output_type_func=label_reduce_output_type)]
fn contour_label_reduce(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;
    let contour_series = &inputs[0];
    let heatmap_series = params
        .slot("image")
        .map(|idx| &inputs[idx])
        .ok_or_else(|| polars_err!(ComputeError: "missing required input 'image'"))?;
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
            let reduction = parse_named(
                LabelReduction::NAMED,
                "reduction",
                params.str_opt("reduction", kwargs.reduction.as_deref(), i)?,
                LabelReduction::Max,
            )?;
            let region_mode = parse_named(
                LabelRegionMode::NAMED,
                "region_mode",
                params.str_opt("region_mode", kwargs.region_mode.as_deref(), i)?,
                LabelRegionMode::Interior,
            )?;
            Ok((reduction, region_mode))
        })?
        else {
            rows.push(AnyValue::Null);
            continue;
        };

        let contours = parse_contour_list(&contours_value)?;
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
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        contour_row(&params, value.is_null(), &mut results, || {
            let dx = params.f64("dx", kwargs.dx, 0.0, i)?;
            let dy = params.f64("dy", kwargs.dy, 0.0, i)?;
            let contour = parse_contour(&value)?;
            Ok(transforms::translate(&contour, dx, dy))
        })?;
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Scale contour.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_scale(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        contour_row(&params, value.is_null(), &mut results, || {
            let sx = params.f64("sx", kwargs.sx, 1.0, i)?;
            let sy = params.f64("sy", kwargs.sy, 1.0, i)?;
            // Per-row capable, like `sx`/`sy` beside it: which point the scale
            // is measured from does not change the output's shape, rank or
            // dtype, so it meets the eligibility rule for a per-row parameter.
            // Resolved against `ScaleOrigin::NAMED` — the hand-written match
            // this replaced ended in a silent default, so `origin="top_left"`
            // scaled about the centroid and said nothing. The no-value default
            // is `Origin` because that is what the Python signature declares;
            // the two used to disagree.
            let scale_origin = parse_named(
                ScaleOrigin::NAMED,
                "origin",
                params.str_opt("origin", kwargs.origin.as_deref(), i)?,
                ScaleOrigin::Origin,
            )?;
            let contour = parse_contour(&value)?;
            Ok(transforms::scale(&contour, sx, sy, scale_origin))
        })?;
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Simplify contour.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_simplify(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        contour_row(&params, value.is_null(), &mut results, || {
            let tolerance = params.f64("tolerance", kwargs.tolerance, 1.0, i)?;
            let contour = parse_contour(&value)?;
            Ok(transforms::simplify(&contour, tolerance))
        })?;
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
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        contour_row(&params, value.is_null(), &mut results, || {
            let ref_width = params.f64("ref_width", kwargs.ref_width, 1.0, i)?;
            let ref_height = params.f64("ref_height", kwargs.ref_height, 1.0, i)?;
            let contour = parse_contour(&value)?;
            Ok(transforms::normalize(&contour, ref_width, ref_height))
        })?;
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Convert normalized coordinates to absolute pixel coordinates.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_to_absolute(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        contour_row(&params, value.is_null(), &mut results, || {
            let ref_width = params.f64("ref_width", kwargs.ref_width, 1.0, i)?;
            let ref_height = params.f64("ref_height", kwargs.ref_height, 1.0, i)?;
            let contour = parse_contour(&value)?;
            Ok(transforms::to_absolute(&contour, ref_width, ref_height))
        })?;
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

/// Ensure contour has specified winding direction.
#[polars_expr(output_type_func=contour_transform_output_type)]
fn contour_ensure_winding(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;

    let series = &inputs[0];
    let len = series.len();
    let mut results: Vec<Option<Contour>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        contour_row(&params, value.is_null(), &mut results, || {
            // Per-row capable: the winding a ring is rewound to changes the
            // vertex order, not the output's shape, rank or dtype.
            //
            // `direction` is required, so there is no default to fall back to.
            // The match this replaced fell back to counter-clockwise for
            // anything it did not recognise, which meant `ensure_winding("CW")`
            // returned the *opposite* of what was asked for, silently.
            let direction = require_named(
                Winding::NAMED,
                "winding direction",
                params
                    .str_opt("direction", kwargs.direction.as_deref(), i)?
                    .ok_or_else(
                        || polars_err!(ComputeError: "ensure_winding requires a 'direction'"),
                    )?,
            )?;
            let contour = parse_contour(&value)?;
            Ok(transforms::ensure_winding(&contour, direction))
        })?;
    }

    build_contour_series(series.name().clone(), results, series.dtype())
}

// ============================================================================
// BBox Matching Plugin Functions
// ============================================================================

/// Parse a single bbox struct AnyValue into a `BoundingBox`.
fn parse_bbox(value: &AnyValue) -> PolarsResult<BoundingBox> {
    match value {
        AnyValue::StructOwned(boxed) => {
            let (values, fields) = boxed.as_ref();
            let mut x = 0.0_f64;
            let mut y = 0.0_f64;
            let mut width = 0.0_f64;
            let mut height = 0.0_f64;
            for (i, field) in fields.iter().enumerate() {
                let f = values
                    .get(i)
                    .and_then(|v| v.try_extract::<f64>().ok())
                    .unwrap_or(0.0);
                match field.name().as_str() {
                    "x" => x = f,
                    "y" => y = f,
                    "width" => width = f,
                    "height" => height = f,
                    _ => {}
                }
            }
            Ok(BoundingBox::new(x, y, width, height))
        }
        _ => Err(polars_err!(ComputeError: "Expected bbox struct, got {:?}", value)),
    }
}

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
fn bbox_pairwise_iou(inputs: &[Series]) -> PolarsResult<Series> {
    let pred_series = &inputs[0];
    let gt_series = &inputs[1];
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

/// Match detection bbox sets with greedy one-to-one IoU assignment.
#[polars_expr(output_type_func=match_detections_output_type)]
fn bbox_match_detections(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
    let params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;
    let pred_series = &inputs[0];
    // Both operands are looked up by name. `scores` is optional, so its
    // position is not fixed once per-row parameters can occupy input slots
    // either; `other` follows the same rule so no read here is positional.
    let gt_series = params
        .slot("other")
        .map(|idx| &inputs[idx])
        .ok_or_else(|| polars_err!(ComputeError: "missing required input 'other'"))?;
    let score_series = params.slot("scores").map(|idx| &inputs[idx]);
    let len = pred_series.len();

    let match_dtype = DataType::Struct(vec![
        Field::new(
            PlSmallStr::from_static("pred_idx"),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            PlSmallStr::from_static("gt_idx"),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            PlSmallStr::from_static("iou"),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new(PlSmallStr::from_static("n_preds"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_gts"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_tp"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_fp"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("n_fn"), DataType::UInt32),
    ]);

    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);
    for i in 0..len {
        let preds_value = pred_series.get(i)?;
        let gts_value = gt_series.get(i)?;
        if preds_value.is_null() || gts_value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }

        // Per-row parameters cannot be range-checked once per batch, so the
        // check moves into the loop and names the offending row. A null
        // `threshold` under `on_null="null"` nulls this row instead.
        let Some(threshold) = params.row(|| {
            let threshold = params.f64("threshold", kwargs.threshold, 0.5, i)?;
            check_range("threshold", threshold, 0.0, 1.0, i)?;
            Ok(threshold)
        })?
        else {
            rows.push(AnyValue::Null);
            continue;
        };

        let preds = parse_bbox_list(&preds_value)?;
        let gts = parse_bbox_list(&gts_value)?;

        let pred_order = if let Some(scores_col) = score_series {
            let score_value = scores_col.get(i)?;
            if score_value.is_null() {
                None
            } else {
                let scores = parse_score_list(&score_value)?;
                if scores.len() != preds.len() {
                    return Err(polars_err!(
                        ComputeError:
                        "scores length ({}) must match prediction count ({}) in row {}",
                        scores.len(),
                        preds.len(),
                        i
                    ));
                }
                Some(score_order(&scores))
            }
        } else {
            None
        };

        let result =
            pairwise::bbox_match_detections(&preds, &gts, threshold, pred_order.as_deref());

        let pred_idx_u32: Vec<u32> = result.pred_idx.iter().map(|v| *v as u32).collect();
        let gt_idx_u32: Vec<Option<u32>> =
            result.gt_idx.iter().map(|v| v.map(|x| x as u32)).collect();

        rows.push(AnyValue::StructOwned(Box::new((
            vec![
                u32_list_anyvalue(&pred_idx_u32, PlSmallStr::from_static("pred_idx")),
                optional_u32_list_anyvalue(&gt_idx_u32, PlSmallStr::from_static("gt_idx")),
                float_list_anyvalue(&result.iou, PlSmallStr::from_static("iou")),
                AnyValue::UInt32(result.n_preds as u32),
                AnyValue::UInt32(result.n_gts as u32),
                AnyValue::UInt32(result.n_tp as u32),
                AnyValue::UInt32(result.n_fp as u32),
                AnyValue::UInt32(result.n_fn as u32),
            ],
            vec![
                Field::new(
                    PlSmallStr::from_static("pred_idx"),
                    DataType::List(Box::new(DataType::UInt32)),
                ),
                Field::new(
                    PlSmallStr::from_static("gt_idx"),
                    DataType::List(Box::new(DataType::UInt32)),
                ),
                Field::new(
                    PlSmallStr::from_static("iou"),
                    DataType::List(Box::new(DataType::Float64)),
                ),
                Field::new(PlSmallStr::from_static("n_preds"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_gts"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_tp"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_fp"), DataType::UInt32),
                Field::new(PlSmallStr::from_static("n_fn"), DataType::UInt32),
            ],
        ))));
    }

    Series::from_any_values_and_dtype(pred_series.name().clone(), &rows, &match_dtype, true)
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
    fn parse_contour_points_field_alias() {
        // "points" is accepted as an alias for "exterior".
        let av = contour_to_anyvalue(&square_with_hole());
        let AnyValue::StructOwned(boxed) = av else {
            panic!("contour_to_anyvalue must build a struct");
        };
        let (values, mut fields) = *boxed;
        fields[0] = Field::new(PlSmallStr::from_static("points"), fields[0].dtype().clone());
        let renamed = AnyValue::StructOwned(Box::new((values, fields)));
        let parsed = parse_contour(&renamed).expect("'points' alias must parse");
        assert_eq!(parsed.exterior.len(), 4);
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
        let point = DataType::Struct(vec![
            Field::new(PlSmallStr::from_static("x"), DataType::Float64),
            Field::new(PlSmallStr::from_static("y"), DataType::Float64),
        ]);
        DataType::Struct(vec![
            Field::new(
                PlSmallStr::from_static("exterior"),
                DataType::List(Box::new(point.clone())),
            ),
            Field::new(
                PlSmallStr::from_static("holes"),
                DataType::List(Box::new(DataType::List(Box::new(point)))),
            ),
            Field::new(PlSmallStr::from_static("is_closed"), DataType::Boolean),
        ])
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
        assert!(err.contains("exterior/points"), "{err}");
    }
}

#[cfg(test)]
mod named_param_tests {
    //! The two string-parameter resolvers, tested here because nothing else
    //! can reach them.
    //!
    //! `ensure_winding` and `scale(origin=)` used to parse by hand and end in
    //! `_ => <default>`, so `ensure_winding("CW")` returned *counter*-clockwise
    //! — the opposite of the request — and `scale(origin="top_left")` scaled
    //! about the centroid. Both silently. They now read `NAMED`, like every
    //! other string parameter in this file.
    //!
    //! Python validates these against `_types.Winding` / `_types.ScaleOrigin`
    //! before the kwargs are built, which is a second wall and a better error
    //! — and also means no Python test can exercise the code below. Without
    //! these, a future edit could put a silent default back in Rust and the
    //! whole suite would stay green.

    use super::*;

    #[test]
    fn a_required_parameter_rejects_a_name_the_table_does_not_hold() {
        let err = require_named(Winding::NAMED, "winding direction", "CW")
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
        // The distinction `require_named` exists for: `parse_named` answers
        // "not supplied" with a default, and a parameter the caller must
        // supply has no correct one.
        assert_eq!(
            parse_named(Winding::NAMED, "d", None, Winding::Clockwise).unwrap(),
            Winding::Clockwise
        );
        assert!(require_named(Winding::NAMED, "d", "").is_err());
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
                require_named(Winding::NAMED, "winding direction", name).unwrap(),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn every_scale_origin_resolves_and_an_unknown_one_does_not() {
        for (name, expected) in [
            ("centroid", ScaleOrigin::Centroid),
            ("bbox_center", ScaleOrigin::BBoxCenter),
            ("origin", ScaleOrigin::Origin),
        ] {
            assert_eq!(
                parse_named(
                    ScaleOrigin::NAMED,
                    "origin",
                    Some(name),
                    ScaleOrigin::Origin
                )
                .unwrap(),
                expected,
                "{name}"
            );
        }
        let err = parse_named(
            ScaleOrigin::NAMED,
            "origin",
            Some("top_left"),
            ScaleOrigin::Origin,
        )
        .expect_err("a plausible name from another library must be rejected")
        .to_string();
        assert!(
            err.contains("top_left") && err.contains("bbox_center"),
            "{err}"
        );
    }

    #[test]
    fn an_absent_origin_takes_the_default_python_declares() {
        // `Origin`, not `Centroid`: the Rust `None` arm and the Python
        // signature used to disagree about this.
        assert_eq!(
            parse_named(ScaleOrigin::NAMED, "origin", None, ScaleOrigin::Origin).unwrap(),
            ScaleOrigin::Origin
        );
    }
}
