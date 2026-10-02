//! Contour plugin functions for polars-cv.
//!
//! This module provides Polars expression functions for contour geometry operations,
//! including measures (area, perimeter), predicates (is_convex, contains_point),
//! transforms (translate, scale, simplify), and pairwise comparisons (IoU, Dice).

use polars::prelude::*;

use crate::geom_schema::{bbox_anyvalue, bbox_struct_dtype, point_anyvalue, point_struct_dtype};
use pyo3_polars::derive::polars_expr;

// Import geometry operations from view-buffer
use view_buffer::geometry::{
    contour::{Contour, Outline, Winding},
    label::score_contours_on_buffer,
    measures,
    ops::ScaleOrigin,
    pairwise, predicates, transforms,
};

// `contour_accessor!` is `#[macro_export]`ed, so it lives at the crate root
// regardless of module order; importing it by name avoids depending on
// `geom_arity` being declared before `contour` in lib.rs.
use crate::contour_accessor;
use crate::geom_arity::{elementwise_field, Arity, ContourOutput};
use crate::geom_columns::{BBoxColumn, ContourColumn, PointColumn};
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
/// - is_closed: Boolean — `true`: a [`Contour`] is always a closed region (an
///   open polyline is an `Outline::Open`, written by `contour_array`).
///
/// Test-only: production contour columns are built straight into Arrow by
/// `geom_schema::contour_array` (CR-36). This per-value construction stays as
/// the independent oracle the equivalence tests compare that builder against.
#[cfg(test)]
pub fn contour_to_anyvalue(contour: &view_buffer::geometry::contour::Contour) -> AnyValue<'static> {
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
            AnyValue::Boolean(true), // a `Contour` is a closed region
        ],
        crate::geom_schema::contour_fields(),
    )))
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
/// The contour and bbox accessors differ only in how a row's two sides are read
/// (`sides`: `None` for a null side) and how its overlap matrix is built, so
/// those are the two parameters; everything else —
/// the null handling, the per-row threshold, the order, the output struct —
/// is shared. The two functions this replaced were near-verbatim duplicates.
type Sides<T> = (Option<Vec<T>>, Option<Vec<T>>);

fn correspond_rows<T>(
    inputs: &[Series],
    params: &GeomParams,
    (threshold, order): (&Param<f64>, &Option<ColumnRef>),
    sides: impl Fn(usize) -> PolarsResult<Sides<T>> + Sync,
    build_matrix: impl Fn(&[T], &[T]) -> Vec<Vec<f64>> + Sync,
) -> PolarsResult<Series> {
    // `order` is read through its reference: it is optional, so nothing here
    // may read a fixed position.
    let order_series = params.optional_column(order);
    let rows = params.map_rows(crate::geom_calls!(), inputs[0].len(), |params, i| {
        let (Some(left), Some(right)) = sides(i)? else {
            return Ok(None);
        };
        // Per-row parameters cannot be range-checked once per batch, so the
        // check names the offending row. A null `threshold` under
        // `on_null="null"` nulls this row instead (`map_rows`).
        let threshold = params.value(threshold, i)?;
        check_range("threshold", threshold, 0.0, 1.0, i)?;
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
        Ok(Some(correspondence_anyvalue(&result)))
    })?;
    let rows: Vec<AnyValue> = rows
        .into_iter()
        .map(|r| r.unwrap_or(AnyValue::Null))
        .collect();
    let dtype = DataType::Struct(correspondence_fields());
    Series::from_any_values_and_dtype(inputs[0].name().clone(), &rows, &dtype, true)
}

/// Drive one pairwise-IoU entry point over its rows: the contour and bbox
/// forms differ only in how a row's two sides are read and how the matrix is
/// built.
fn pairwise_rows<T>(
    inputs: &[Series],
    params: &GeomParams,
    sides: impl Fn(usize) -> PolarsResult<Sides<T>> + Sync,
    build_matrix: impl Fn(&[T], &[T]) -> Vec<Vec<f64>> + Sync,
) -> PolarsResult<Series> {
    let rows = params.map_rows(crate::geom_calls!(), inputs[0].len(), |_, i| {
        let (Some(left), Some(right)) = sides(i)? else {
            return Ok(None);
        };
        matrix_anyvalue(&build_matrix(&left, &right)).map(Some)
    })?;
    let rows: Vec<AnyValue> = rows
        .into_iter()
        .map(|r| r.unwrap_or(AnyValue::Null))
        .collect();
    build_pairwise_matrix_series(inputs[0].name().clone(), rows)
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

contour_accessor! {
    /// Compute contour area.
    map fn contour_area / contour_area_output_type -> |_input| DataType::Float64;
    reads Contour;
    parse GeometryOp::Area { signed };
    |contour, params, row| {
        let signed = params.value(signed, row)?;
        Ok(AnyValue::Float64(measures::area(contour, signed)))
    }
}

contour_accessor! {
    /// Compute contour perimeter.
    map fn contour_perimeter / contour_perimeter_output_type -> |_input| DataType::Float64;
    reads Outline;
    parse GeometryOp::Perimeter;
    |outline, _params, _row| Ok(AnyValue::Float64(measures::outline_length(outline)))
}

contour_accessor! {
    /// Compute winding direction.
    map fn contour_winding / contour_winding_output_type -> |_input| DataType::String;
    reads Contour;
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
    reads Contour;
    parse GeometryOp::Centroid;
    |contour, _params, _row| {
        let center = measures::centroid(contour);
        Ok(point_anyvalue(center.x, center.y))
    }
}

contour_accessor! {
    /// Compute contour bounding box — an `{x, y, width, height}` struct per contour.
    map fn contour_bounding_box / contour_bounding_box_output_type -> |_input| bbox_struct_dtype();
    reads Outline;
    parse GeometryOp::BoundingBox;
    |outline, _params, _row| Ok(bbox_anyvalue(outline.bounding_box()))
}

// ============================================================================
// Contour Plugin Functions - Predicates
// ============================================================================

contour_accessor! {
    /// Check if contour is convex.
    map fn contour_is_convex / contour_is_convex_output_type -> |_input| DataType::Boolean;
    reads Contour;
    parse ContourFn::IsConvex;
    |contour, _params, _row| Ok(AnyValue::Boolean(predicates::contour_is_convex(contour)))
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
    let contours = ContourColumn::new(&inputs[0]);
    let points = PointColumn::new(params.column(point));
    let rows = params.map_rows(crate::geom_calls!(), inputs[0].len(), |_, i| {
        let Some(p) = points.get(i)? else {
            return Ok(None);
        };
        let Some(row) = contours.row(i)? else {
            return Ok(None);
        };
        Ok(Some(
            row.iter()
                .map(|c| AnyValue::Boolean(predicates::contains_point(c, p.x, p.y)))
                .collect(),
        ))
    })?;
    AnyValue::column(
        inputs[0].name().clone(),
        rows,
        contours.arity(),
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
    let (preds, gts) = (
        ContourColumn::new(&inputs[0]),
        ContourColumn::new(params.column(other)),
    );
    pairwise_rows(
        inputs,
        &params,
        |i| Ok((preds.row(i)?, gts.row(i)?)),
        pairwise::iou_matrix,
    )
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
    let (left, right) = (
        ContourColumn::new(&inputs[0]),
        ContourColumn::new(params.column(other)),
    );
    correspond_rows(
        inputs,
        &params,
        (threshold, order),
        |i| Ok((left.row(i)?, right.row(i)?)),
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
    let contours = ContourColumn::new(&inputs[0]);
    let heatmap_series = params.column(image);
    let rows = params.map_rows(crate::geom_calls!(), inputs[0].len(), |params, i| {
        // The heatmap decodes as the pipeline's `list`/`array` source does:
        // a grid of values, `[H, W]` or `[H, W, 1]`, refused if jagged or
        // holding a null. A null or empty heatmap is a null row.
        let Some(heatmap) =
            crate::graph::decode::decode_list_or_array_source(heatmap_series, i, None, false)
                .map_err(|e| polars_err!(ComputeError: "label_reduce image, row {}: {}", i, e))?
        else {
            return Ok(None);
        };
        let Some(contours) = contours.row(i)? else {
            return Ok(None);
        };
        // Per-row capable, matching `Pipeline.label_reduce`: neither choice
        // affects the output's shape or dtype.
        let reduction = params.value(reduction, i)?;
        let region_mode = params.value(region_mode, i)?;
        let scores = score_contours_on_buffer(&heatmap, &contours, reduction, region_mode)
            .map_err(|err| polars_err!(ComputeError: "{}", err))?;
        Ok(Some(float_list_anyvalue(
            &scores,
            PlSmallStr::from_static("scores"),
        )))
    })?;
    let rows: Vec<AnyValue> = rows
        .into_iter()
        .map(|r| r.unwrap_or(AnyValue::Null))
        .collect();
    let dtype = DataType::List(Box::new(DataType::Float64));
    Series::from_any_values_and_dtype(inputs[0].name().clone(), &rows, &dtype, true)
}

contour_accessor! {
    /// Compute IoU between two contours, broadcasting a set against a single.
    zip fn contour_iou / contour_iou_output_type -> DataType::Float64;
    reads Contour;
    parse ContourFn::Iou { other };
    |a, b, _params, _row| Ok(AnyValue::Float64(pairwise::iou(a, b)))
}

contour_accessor! {
    /// Compute Dice coefficient between two contours.
    zip fn contour_dice / contour_dice_output_type -> DataType::Float64;
    reads Contour;
    parse ContourFn::Dice { other };
    |a, b, _params, _row| Ok(AnyValue::Float64(pairwise::dice(a, b)))
}

contour_accessor! {
    /// Compute Hausdorff distance between two contours.
    zip fn contour_hausdorff / contour_hausdorff_output_type -> DataType::Float64;
    reads Outline;
    parse ContourFn::Hausdorff { other };
    |a, b, _params, _row| Ok(AnyValue::Float64(pairwise::hausdorff_distance_outlines(a, b)))
}

/// The fields of `contour_boundary_distances`'s struct, in order.
const BOUNDARY_DISTANCE_FIELDS: [&str; 5] = ["mean_a_to_b", "mean_b_to_a", "assd", "hd", "hd95"];

fn boundary_distance_fields() -> Vec<Field> {
    BOUNDARY_DISTANCE_FIELDS
        .iter()
        .map(|name| Field::new(PlSmallStr::from_static(name), DataType::Float64))
        .collect()
}

fn boundary_distances_dtype() -> DataType {
    DataType::Struct(boundary_distance_fields())
}

contour_accessor! {
    /// Point-to-edge boundary distances between two contours, broadcasting a
    /// set against a single.
    zip fn contour_boundary_distances / contour_boundary_distances_output_type
        -> boundary_distances_dtype();
    reads Outline;
    parse ContourFn::BoundaryDistances { other, sample_step };
    |a, b, params, row| {
        let step = match sample_step {
            Some(step) => {
                let step = params.value(step, row)?;
                if !(step.is_finite() && step > 0.0) {
                    polars_bail!(ComputeError:
                        "boundary_distances: sample_step must be a positive number, \
                         got {} (row {})", step, row);
                }
                Some(step)
            }
            None => None,
        };
        Ok(match pairwise::boundary_distances(a, b, step) {
            None => AnyValue::Null,
            Some(d) => AnyValue::StructOwned(Box::new((
                [d.mean_a_to_b, d.mean_b_to_a, d.assd, d.hd, d.hd95]
                    .into_iter()
                    .map(AnyValue::Float64)
                    .collect(),
                boundary_distance_fields(),
            ))),
        })
    }
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
    reads Outline;
    parse GeometryOp::Translate { dx, dy };
    |outline, params, row| {
        let dx = params.value(dx, row)?;
        let dy = params.value(dy, row)?;
        Ok(outline.map_points(|c| transforms::translate(c, dx, dy)))
    }
}

contour_accessor! {
    /// Scale contour.
    map fn contour_scale / contour_scale_output_type
        -> |input| Arity::elem_dtype(input);
    reads Outline;
    parse GeometryOp::Scale { sx, sy, origin };
    |outline, params, row| {
        let sx = params.value(sx, row)?;
        let sy = params.value(sy, row)?;
        // Per-row capable, like `sx`/`sy` beside it: which point the scale
        // is measured from does not change the output's shape, rank or
        // dtype. This is the pipeline op's own definition, default and all.
        let scale_origin = params.value(origin, row)?;
        if !outline.is_closed() && scale_origin == ScaleOrigin::Centroid {
            polars_bail!(ComputeError:
                "scale(origin=\"centroid\") needs a closed contour: an open polyline \
                 (is_closed = false) bounds no region to take a centroid of (row {}). \
                 Scale about \"bbox_center\" or \"origin\" instead",
                row
            );
        }
        Ok(outline.map_points(|c| transforms::scale(c, sx, sy, scale_origin)))
    }
}

contour_accessor! {
    /// Simplify contour.
    map fn contour_simplify / contour_simplify_output_type
        -> |input| Arity::elem_dtype(input);
    reads Outline;
    parse GeometryOp::Simplify { tolerance };
    |outline, params, row| {
        let tolerance = params.value(tolerance, row)?;
        Ok(transforms::simplify_outline(outline, tolerance))
    }
}

contour_accessor! {
    /// Flip contour (reverse winding).
    map fn contour_flip / contour_flip_output_type -> |input| Arity::elem_dtype(input);
    reads Outline;
    parse ContourFn::Flip;
    |outline, _params, _row| Ok(outline.map_points(transforms::flip))
}

contour_accessor! {
    /// Compute convex hull.
    map fn contour_convex_hull / contour_convex_hull_output_type
        -> |input| Arity::elem_dtype(input);
    reads Outline;
    parse GeometryOp::ConvexHull;
    |outline, _params, _row| Ok(transforms::convex_hull_outline(outline))
}

contour_accessor! {
    /// Normalize contour coordinates to [0, 1] range.
    map fn contour_normalize / contour_normalize_output_type
        -> |input| Arity::elem_dtype(input);
    reads Outline;
    parse ContourFn::Normalize { width, height };
    |outline, params, row| {
        let ref_width = params.value(width, row)?;
        let ref_height = params.value(height, row)?;
        Ok(outline.map_points(|c| transforms::normalize(c, ref_width, ref_height)))
    }
}

contour_accessor! {
    /// Convert normalized coordinates to absolute pixel coordinates.
    map fn contour_to_absolute / contour_to_absolute_output_type
        -> |input| Arity::elem_dtype(input);
    reads Outline;
    parse ContourFn::ToAbsolute { width, height };
    |outline, params, row| {
        let ref_width = params.value(width, row)?;
        let ref_height = params.value(height, row)?;
        Ok(outline.map_points(|c| transforms::to_absolute(c, ref_width, ref_height)))
    }
}

contour_accessor! {
    /// Ensure contour has specified winding direction.
    map fn contour_ensure_winding / contour_ensure_winding_output_type
        -> |input| Arity::elem_dtype(input);
    reads Contour;
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

/// Pairwise IoU matrix between two sets of bounding boxes.
#[polars_expr(output_type_func=pairwise_iou_output_type)]
fn bbox_pairwise_iou(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "bbox_pairwise_iou";
    let (op, params) = GeomParams::parse::<BBoxFn<Wire>>(inputs, kwargs, NAME)?;
    let BBoxFn::PairwiseIou { other } = &op else {
        return Err(parsed_as_another(NAME));
    };
    let (preds, gts) = (
        BBoxColumn::new(&inputs[0]),
        BBoxColumn::new(params.column(other)),
    );
    pairwise_rows(
        inputs,
        &params,
        |i| Ok((preds.row(i)?, gts.row(i)?)),
        pairwise::bbox_iou_matrix,
    )
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
    let (left, right) = (
        BBoxColumn::new(&inputs[0]),
        BBoxColumn::new(params.column(other)),
    );
    correspond_rows(
        inputs,
        &params,
        (threshold, order),
        |i| Ok((left.row(i)?, right.row(i)?)),
        pairwise::bbox_iou_matrix,
    )
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
