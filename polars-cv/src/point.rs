//! Point plugin functions for polars-cv.
//!
//! Coordinate transforms (normalize, translate, scale, rotate, …), distances
//! (Euclidean, Manhattan, to a contour) and predicates over a point column.
//!
//! Every function reads its operands through the geometry column readers
//! ([`PointColumn`], [`ContourColumn`], [`BBoxColumn`]) and runs its rows
//! through [`GeomParams::map_rows`] — the one row loop, split over the
//! plugin's thread pool, with the null-parameter policy — so none of them
//! parses a value or walks rows by hand. A null input row is a null result;
//! a point with a null coordinate is an error.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use view_buffer::geometry::contour::Point;
use view_buffer::geometry::{measures, predicates};

use crate::geom_calls;
use crate::geom_columns::{BBoxColumn, ContourColumn, PointColumn};
use crate::geom_fns::PointFn;
use crate::geom_params::{parsed_as_another, GeomKwargs, GeomParams};
use view_buffer::mode::Wire;

/// A per-row result of a point function, and the column it is collected into.
trait PointOutput: Sized + Send {
    fn series(name: PlSmallStr, rows: Vec<Option<Self>>) -> PolarsResult<Series>;
}

impl PointOutput for f64 {
    fn series(name: PlSmallStr, rows: Vec<Option<Self>>) -> PolarsResult<Series> {
        Ok(Float64Chunked::from_iter_options(name, rows.into_iter()).into_series())
    }
}

impl PointOutput for bool {
    fn series(name: PlSmallStr, rows: Vec<Option<Self>>) -> PolarsResult<Series> {
        Ok(BooleanChunked::from_iter_options(name, rows.into_iter()).into_series())
    }
}

/// A point, as the `{x, y}` struct of [`point_struct_dtype`](crate::geom_schema::point_struct_dtype).
impl PointOutput for Point {
    fn series(name: PlSmallStr, rows: Vec<Option<Self>>) -> PolarsResult<Series> {
        let [x, y] = crate::geom_schema::POINT_FIELD_NAMES;
        let xs = Float64Chunked::from_iter_options(x.into(), rows.iter().map(|p| p.map(|p| p.x)));
        let ys = Float64Chunked::from_iter_options(y.into(), rows.iter().map(|p| p.map(|p| p.y)));
        let mut out = StructChunked::from_series(
            name,
            rows.len(),
            [xs.into_series(), ys.into_series()].iter(),
        )?;
        // A null row is a null point, not a point of nulls.
        if rows.iter().any(Option::is_none) {
            let validity: polars_arrow::bitmap::Bitmap = rows.iter().map(Option::is_some).collect();
            out = out.with_outer_validity(Some(validity));
        }
        Ok(out.into_series())
    }
}

/// Run `row` for every row ([`GeomParams::map_rows`]) and collect the column.
fn point_rows<T: PointOutput>(
    inputs: &[Series],
    params: &GeomParams,
    calls: &crate::row_split::CallTracker,
    row: impl Fn(&GeomParams, usize) -> PolarsResult<Option<T>> + Sync,
) -> PolarsResult<Series> {
    let rows = params.map_rows(calls, inputs[0].len(), row)?;
    T::series(inputs[0].name().clone(), rows)
}

/// Output type for point transform operations (returns Point struct).
fn point_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("point"),
        crate::geom_schema::point_struct_dtype(),
    ))
}

/// The call's definition as `PointFn::$var`, or the function's error.
macro_rules! parse_as {
    ($params:ident = $inputs:ident, $kwargs:ident, $name:literal, $var:ident $({ $($field:ident),* })?) => {
        let (op, $params) = GeomParams::parse::<PointFn<Wire>>($inputs, $kwargs, $name)?;
        let PointFn::$var $({ $($field),* })? = &op else {
            return Err(parsed_as_another($name));
        };
    };
}

// ============================================================================
// Coordinate transforms
// ============================================================================

/// Normalize point coordinates to [0, 1] range.
#[polars_expr(output_type_func=point_output_type)]
fn point_normalize(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_normalize",
        Normalize { width, height }
    );
    let points = PointColumn::new(&inputs[0]);
    point_rows(inputs, &params, geom_calls!(), |params, i| {
        let Some(p) = points.get(i)? else {
            return Ok(None);
        };
        let (w, h) = (params.value(width, i)?, params.value(height, i)?);
        // Per-row dimensions cannot be validated once per batch.
        if w == 0.0 || h == 0.0 {
            polars_bail!(ComputeError: "ref_width and ref_height must be non-zero (row {})", i);
        }
        Ok(Some(Point::new(p.x / w, p.y / h)))
    })
}

/// Convert normalized coordinates to absolute pixel coordinates.
#[polars_expr(output_type_func=point_output_type)]
fn point_to_absolute(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_to_absolute",
        ToAbsolute { width, height }
    );
    let points = PointColumn::new(&inputs[0]);
    point_rows(inputs, &params, geom_calls!(), |params, i| {
        let Some(p) = points.get(i)? else {
            return Ok(None);
        };
        Ok(Some(Point::new(
            p.x * params.value(width, i)?,
            p.y * params.value(height, i)?,
        )))
    })
}

/// Translate point by offset.
#[polars_expr(output_type_func=point_output_type)]
fn point_translate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_translate",
        Translate { dx, dy }
    );
    let points = PointColumn::new(&inputs[0]);
    point_rows(inputs, &params, geom_calls!(), |params, i| {
        let Some(p) = points.get(i)? else {
            return Ok(None);
        };
        Ok(Some(Point::new(
            p.x + params.value(dx, i)?,
            p.y + params.value(dy, i)?,
        )))
    })
}

/// Scale point coordinates.
#[polars_expr(output_type_func=point_output_type)]
fn point_scale(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(params = inputs, kwargs, "point_scale", Scale { sx, sy });
    let points = PointColumn::new(&inputs[0]);
    point_rows(inputs, &params, geom_calls!(), |params, i| {
        let Some(p) = points.get(i)? else {
            return Ok(None);
        };
        Ok(Some(Point::new(
            p.x * params.value(sx, i)?,
            p.y * params.value(sy, i)?,
        )))
    })
}

/// Rotate point around an origin (default `(0, 0)`) by angle (radians).
///
/// A null `origin` value nulls its row, as a null operand does everywhere; it
/// used to rotate about `(0, 0)` instead.
#[polars_expr(output_type_func=point_output_type)]
fn point_rotate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_rotate",
        Rotate { angle, origin }
    );
    let points = PointColumn::new(&inputs[0]);
    // Read through its reference: `origin` is optional, so its position is
    // not fixed once a per-row `angle` can also occupy an input slot.
    let origins = params.optional_column(origin).map(PointColumn::new);
    point_rows(inputs, &params, geom_calls!(), |params, i| {
        let Some(p) = points.get(i)? else {
            return Ok(None);
        };
        let o = match &origins {
            Some(origins) => match origins.get(i)? {
                Some(o) => o,
                None => return Ok(None),
            },
            None => Point::new(0.0, 0.0),
        };
        let angle = params.value(angle, i)?;
        let (sin_a, cos_a) = angle.sin_cos();
        let (dx, dy) = (p.x - o.x, p.y - o.y);
        Ok(Some(Point::new(
            dx * cos_a - dy * sin_a + o.x,
            dx * sin_a + dy * cos_a + o.y,
        )))
    })
}

// ============================================================================
// Two points
// ============================================================================

/// Run `f` over the rows where both this point and the `other` point are
/// present.
fn with_other<T: PointOutput>(
    inputs: &[Series],
    params: &GeomParams,
    other: &crate::ops::ColumnRef,
    calls: &crate::row_split::CallTracker,
    f: impl Fn(&GeomParams, usize, Point, Point) -> PolarsResult<T> + Sync,
) -> PolarsResult<Series> {
    let (a, b) = (
        PointColumn::new(&inputs[0]),
        PointColumn::new(params.column(other)),
    );
    point_rows(inputs, params, calls, |params, i| {
        let (Some(p), Some(q)) = (a.get(i)?, b.get(i)?) else {
            return Ok(None);
        };
        f(params, i, p, q).map(Some)
    })
}

/// Compute Euclidean distance between two points.
#[polars_expr(output_type=Float64)]
fn point_distance(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_distance",
        Distance { other }
    );
    with_other(inputs, &params, other, geom_calls!(), |_, _, p, q| {
        Ok((q.x - p.x).hypot(q.y - p.y))
    })
}

/// Compute Manhattan (L1) distance between two points.
#[polars_expr(output_type=Float64)]
fn point_manhattan_distance(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_manhattan_distance",
        ManhattanDistance { other }
    );
    with_other(inputs, &params, other, geom_calls!(), |_, _, p, q| {
        Ok((q.x - p.x).abs() + (q.y - p.y).abs())
    })
}

/// Compute angle from this point to another in radians.
#[polars_expr(output_type=Float64)]
fn point_angle_to(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(params = inputs, kwargs, "point_angle_to", AngleTo { other });
    with_other(inputs, &params, other, geom_calls!(), |_, _, p, q| {
        Ok((q.y - p.y).atan2(q.x - p.x))
    })
}

/// Compute midpoint between two points.
#[polars_expr(output_type_func=point_output_type)]
fn point_midpoint(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_midpoint",
        Midpoint { other }
    );
    with_other(inputs, &params, other, geom_calls!(), |_, _, p, q| {
        Ok(Point::new((p.x + q.x) / 2.0, (p.y + q.y) / 2.0))
    })
}

/// Linear interpolation between two points.
#[polars_expr(output_type_func=point_output_type)]
fn point_interpolate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_interpolate",
        Interpolate { other, t }
    );
    with_other(inputs, &params, other, geom_calls!(), |params, i, p, q| {
        let t = params.value(t, i)?;
        Ok(Point::new(p.x + t * (q.x - p.x), p.y + t * (q.y - p.y)))
    })
}

// ============================================================================
// A point and a contour or a bbox
// ============================================================================

/// Run `f` over the rows where both the point and the (single) contour are
/// present.
fn with_contour<T: PointOutput>(
    inputs: &[Series],
    params: &GeomParams,
    contour: &crate::ops::ColumnRef,
    calls: &crate::row_split::CallTracker,
    f: impl Fn(Point, &view_buffer::geometry::contour::Contour) -> Option<T> + Sync,
) -> PolarsResult<Series> {
    let (points, contours) = (
        PointColumn::new(&inputs[0]),
        ContourColumn::new(params.column(contour)),
    );
    point_rows(inputs, params, calls, |_, i| {
        let Some(p) = points.get(i)? else {
            return Ok(None);
        };
        let Some(c) = contours.single(i)? else {
            return Ok(None);
        };
        Ok(f(p, &c))
    })
}

/// Compute minimum distance from point to contour boundary.
#[polars_expr(output_type=Float64)]
fn point_distance_to_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_distance_to_contour",
        DistanceToContour { contour }
    );
    with_contour(inputs, &params, contour, geom_calls!(), |p, c| {
        Some(measures::distance_to_contour(&p, c))
    })
}

/// Compute signed distance from point to contour boundary.
/// Negative if inside, positive if outside.
#[polars_expr(output_type=Float64)]
fn point_signed_distance_to_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_signed_distance_to_contour",
        SignedDistanceToContour { contour }
    );
    with_contour(inputs, &params, contour, geom_calls!(), |p, c| {
        let dist = measures::distance_to_contour(&p, c);
        Some(if predicates::contains_point(c, p.x, p.y) {
            -dist
        } else {
            dist
        })
    })
}

/// Find nearest point on contour boundary.
#[polars_expr(output_type_func=point_output_type)]
fn point_nearest_on_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_nearest_on_contour",
        NearestOnContour { contour }
    );
    with_contour(inputs, &params, contour, geom_calls!(), |p, c| {
        measures::nearest_point_on_contour(&p, c)
    })
}

/// Check if point is within bounding box.
#[polars_expr(output_type=Boolean)]
fn point_within_bbox(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_within_bbox",
        WithinBbox { bbox }
    );
    let (points, bboxes) = (
        PointColumn::new(&inputs[0]),
        BBoxColumn::new(params.column(bbox)),
    );
    point_rows(inputs, &params, geom_calls!(), |_, i| {
        let (Some(p), Some(b)) = (points.get(i)?, bboxes.single(i)?) else {
            return Ok(None);
        };
        Ok(Some(
            p.x >= b.x && p.x <= b.x + b.width && p.y >= b.y && p.y <= b.y + b.height,
        ))
    })
}
