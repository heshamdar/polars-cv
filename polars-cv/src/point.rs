//! Point plugin functions for polars-cv.
//!
//! Coordinate transforms (normalize, translate, scale, rotate, …), distances
//! (Euclidean, Manhattan, to a contour) and predicates over a point column.
//!
//! Every function reads its operands through the geometry column readers
//! ([`PointColumn`], [`ContourColumn`], [`BBoxColumn`]) and runs its rows
//! through [`map_points`] or [`zip_points`] — the one row loop, over
//! [`GeomParams::map_rows`] (split over the plugin's thread pool, with the
//! null-parameter policy) — so none of them parses a value or walks rows by
//! hand. A null input row is a null result; a point with a null coordinate is
//! an error.
//!
//! **Arity.** A point column holds one point per row, or a point set
//! (`POINT_SET_SCHEMA`, [`Arity::Set`]) per row — the `.contour` arity rule.
//! Over a set every function gives one value per point, in input order (a
//! null point gives a null in its place), as `List(elem)`. A contour or bbox
//! operand is one per row and broadcasts against the set. Two point columns
//! broadcast either way, a set against a single point; a set on both sides is
//! refused (an N x M matrix and an index-wise pairing are both plausible).
//! The declared type reads the same arities ([`unary_field`],
//! [`binary_field`]), so plan and execution cannot disagree.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use view_buffer::geometry::contour::{Contour, Outline, Point};
use view_buffer::geometry::{measures, predicates};

use crate::geom_arity::{Arity, ReadContour};
use crate::geom_calls;
use crate::geom_columns::{BBoxColumn, ContourColumn, PointColumn};
use crate::geom_fns::PointFn;
use crate::geom_params::{parsed_as_another, GeomKwargs, GeomParams};
use crate::ops::ColumnRef;
use crate::row_split::CallTracker;
use view_buffer::mode::Wire;

/// A per-point result of a point function, and the column of them.
pub(crate) trait PointOutput: Sized + Send {
    /// The element dtype.
    fn dtype() -> DataType;
    fn series(name: PlSmallStr, values: Vec<Option<Self>>) -> PolarsResult<Series>;
}

impl PointOutput for f64 {
    fn dtype() -> DataType {
        DataType::Float64
    }
    fn series(name: PlSmallStr, values: Vec<Option<Self>>) -> PolarsResult<Series> {
        Ok(Float64Chunked::from_iter_options(name, values.into_iter()).into_series())
    }
}

impl PointOutput for bool {
    fn dtype() -> DataType {
        DataType::Boolean
    }
    fn series(name: PlSmallStr, values: Vec<Option<Self>>) -> PolarsResult<Series> {
        Ok(BooleanChunked::from_iter_options(name, values.into_iter()).into_series())
    }
}

/// A point, as the `{x, y}` struct of [`point_struct_dtype`](crate::geom_schema::point_struct_dtype).
impl PointOutput for Point {
    fn dtype() -> DataType {
        crate::geom_schema::point_struct_dtype()
    }
    fn series(name: PlSmallStr, values: Vec<Option<Self>>) -> PolarsResult<Series> {
        let [x, y] = crate::geom_schema::POINT_FIELD_NAMES;
        let xs = Float64Chunked::from_iter_options(x.into(), values.iter().map(|p| p.map(|p| p.x)));
        let ys = Float64Chunked::from_iter_options(y.into(), values.iter().map(|p| p.map(|p| p.y)));
        let mut out = StructChunked::from_series(
            name,
            values.len(),
            [xs.into_series(), ys.into_series()].iter(),
        )?;
        // A null value is a null point, not a point of nulls.
        if values.iter().any(Option::is_none) {
            let validity: polars_arrow::bitmap::Bitmap =
                values.iter().map(Option::is_some).collect();
            out = out.with_outer_validity(Some(validity));
        }
        Ok(out.into_series())
    }
}

/// A point's coordinates as a pair, `Array(Float64, 2)`.
impl PointOutput for [f64; 2] {
    fn dtype() -> DataType {
        DataType::Array(Box::new(DataType::Float64), 2)
    }
    fn series(name: PlSmallStr, values: Vec<Option<Self>>) -> PolarsResult<Series> {
        use polars_arrow::array::{FixedSizeListArray, PrimitiveArray};
        let flat: Vec<f64> = values.iter().flat_map(|v| v.unwrap_or([0.0; 2])).collect();
        let validity: Option<polars_arrow::bitmap::Bitmap> = values
            .iter()
            .any(Option::is_none)
            .then(|| values.iter().map(Option::is_some).collect());
        let dtype = Self::dtype().to_arrow(CompatLevel::newest());
        let array = FixedSizeListArray::try_new(
            dtype,
            values.len(),
            PrimitiveArray::from_vec(flat).boxed(),
            validity,
        )?;
        Series::from_arrow(name, array.boxed())
    }
}

/// The column of `rows` — per row `None` (a null row) or one result per point
/// — in `arity`: the results themselves for one point per row, a list of them
/// for a set.
pub(crate) fn assemble<T: PointOutput>(
    name: PlSmallStr,
    rows: Vec<Option<Vec<Option<T>>>>,
    arity: Arity,
) -> PolarsResult<Series> {
    use polars_arrow::array::ListArray;
    use polars_arrow::offset::Offsets;

    if arity == Arity::Single {
        let values = rows
            .into_iter()
            .map(|row| row.and_then(|v| v.into_iter().next().flatten()))
            .collect();
        return T::series(name, values);
    }
    let lengths: Vec<usize> = rows
        .iter()
        .map(|r| r.as_ref().map_or(0, Vec::len))
        .collect();
    let validity: Option<polars_arrow::bitmap::Bitmap> = rows
        .iter()
        .any(Option::is_none)
        .then(|| rows.iter().map(Option::is_some).collect());
    let flat: Vec<Option<T>> = rows.into_iter().flatten().flatten().collect();
    let values = T::series(PlSmallStr::from_static("item"), flat)?
        .rechunk()
        .to_arrow(0, CompatLevel::newest());
    let offsets = Offsets::<i64>::try_from_lengths(lengths.into_iter())?;
    let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
    let list = ListArray::<i64>::try_new(dtype, offsets.into(), values, validity)?;
    let series = Series::from_arrow(name, list.boxed())?;
    let declared = arity.wrap(T::dtype());
    if series.dtype() == &declared {
        Ok(series)
    } else {
        series.strict_cast(&declared)
    }
}

/// The declared output of a function of the point column alone (any other
/// operand is one per row): `elem`, per point.
fn unary_field<T: PointOutput>(input_fields: &[Field]) -> PolarsResult<Field> {
    let input = input_fields
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "a point function takes a point column"))?;
    Ok(Field::new(
        input.name().clone(),
        Arity::of_points(input.dtype()).wrap(T::dtype()),
    ))
}

/// The declared output of a function of two point columns, the second its
/// `other` operand at input 1 ([`zip_points`] holds the call to that).
fn binary_field<T: PointOutput>(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b, ..] = input_fields else {
        polars_bail!(ComputeError: "a two-point function takes two point columns");
    };
    let arity = Arity::of_points(a.dtype()).combine(Arity::of_points(b.dtype()));
    Ok(Field::new(a.name().clone(), arity.wrap(T::dtype())))
}

fn f64_output(f: &[Field]) -> PolarsResult<Field> {
    unary_field::<f64>(f)
}
fn bool_output(f: &[Field]) -> PolarsResult<Field> {
    unary_field::<bool>(f)
}
fn point_output(f: &[Field]) -> PolarsResult<Field> {
    unary_field::<Point>(f)
}
fn pair_output(f: &[Field]) -> PolarsResult<Field> {
    unary_field::<[f64; 2]>(f)
}
fn pair_f64_output(f: &[Field]) -> PolarsResult<Field> {
    binary_field::<f64>(f)
}
fn pair_point_output(f: &[Field]) -> PolarsResult<Field> {
    binary_field::<Point>(f)
}

/// **The one row loop of the one-point-column functions.** Per row, `ctx`
/// reads what the row's points share (per-row parameters, a contour or bbox
/// operand) — `None` nulls the row, as a null operand does — and `f` maps
/// each point. Over a set, one result per point in input order.
fn map_points<C, T: PointOutput>(
    inputs: &[Series],
    params: &GeomParams,
    calls: &CallTracker,
    ctx: impl Fn(&GeomParams, usize) -> PolarsResult<Option<C>> + Sync,
    f: impl Fn(&C, Point) -> PolarsResult<Option<T>> + Sync,
) -> PolarsResult<Series> {
    let points = PointColumn::new(&inputs[0]);
    let rows = params.map_rows(calls, inputs[0].len(), |params, i| {
        let Some(row) = points.row(i)? else {
            return Ok(None);
        };
        let Some(c) = ctx(params, i)? else {
            return Ok(None);
        };
        row.into_iter()
            .map(|p| p.map_or(Ok(None), |p| f(&c, p)))
            .collect::<PolarsResult<Vec<_>>>()
            .map(Some)
    })?;
    assemble(inputs[0].name().clone(), rows, points.arity())
}

/// **The one row loop of the two-point-column functions.** A set on either
/// side broadcasts against the other side's point, each result keeping the
/// operands' order (`f(this, other)`); a set on both sides is refused.
fn zip_points<T: PointOutput>(
    inputs: &[Series],
    params: &GeomParams,
    other: &ColumnRef,
    calls: &CallTracker,
    name: &'static str,
    f: impl Fn(&GeomParams, usize, Point, Point) -> PolarsResult<T> + Sync,
) -> PolarsResult<Series> {
    // The declared type reads `other` at input 1 (its first data operand,
    // as `_GeomNamespace._call` appends them); refuse anything else rather
    // than produce an arity the declaration did not say.
    if other.0 != 1 {
        polars_bail!(ComputeError: "internal: {}'s other point is input {}, not 1", name, other.0);
    }
    let (a, b) = (
        PointColumn::new(&inputs[0]),
        PointColumn::new(params.column(other)),
    );
    if a.arity() == Arity::Set && b.arity() == Arity::Set {
        polars_bail!(ComputeError:
            "{} received a point set on both sides, which has two different \
             meanings and no default: .explode() one side to pair them row by \
             row. One side may be a set; both may not.",
            name
        );
    }
    let rows = params.map_rows(calls, inputs[0].len(), |params, i| {
        let (Some(left), Some(right)) = (a.row(i)?, b.row(i)?) else {
            return Ok(None);
        };
        let pair = |p: Option<Point>, q: Option<Point>| match (p, q) {
            (Some(p), Some(q)) => f(params, i, p, q).map(Some),
            _ => Ok(None),
        };
        let results = match (a.arity(), b.arity()) {
            (Arity::Set, _) => left.iter().map(|&p| pair(p, right[0])).collect(),
            (_, Arity::Set) => right.iter().map(|&q| pair(left[0], q)).collect(),
            (Arity::Single, Arity::Single) => pair(left[0], right[0]).map(|r| vec![r]),
        };
        results.map(Some)
    })?;
    assemble(inputs[0].name().clone(), rows, a.arity().combine(b.arity()))
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

/// No per-row context: every point of the row is mapped alone.
fn none(_: &GeomParams, _: usize) -> PolarsResult<Option<()>> {
    Ok(Some(()))
}

// ============================================================================
// Coordinates
// ============================================================================

/// The x coordinate.
#[polars_expr(output_type_func=f64_output)]
fn point_x(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(params = inputs, kwargs, "point_x", X);
    map_points(inputs, &params, geom_calls!(), none, |_, p| Ok(Some(p.x)))
}

/// The y coordinate.
#[polars_expr(output_type_func=f64_output)]
fn point_y(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(params = inputs, kwargs, "point_y", Y);
    map_points(inputs, &params, geom_calls!(), none, |_, p| Ok(Some(p.y)))
}

/// The point's coordinates as a pair, in `order`.
#[polars_expr(output_type_func=pair_output)]
fn point_to_coords(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_to_coords",
        ToCoords { order }
    );
    map_points(
        inputs,
        &params,
        geom_calls!(),
        |params, i| params.value(order, i).map(Some),
        |order, p| Ok(Some(order.pair(&p))),
    )
}

// ============================================================================
// Coordinate transforms
// ============================================================================

/// Normalize point coordinates to [0, 1] range.
#[polars_expr(output_type_func=point_output)]
fn point_normalize(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_normalize",
        Normalize { width, height }
    );
    map_points(
        inputs,
        &params,
        geom_calls!(),
        |params, i| {
            let (w, h) = (params.value(width, i)?, params.value(height, i)?);
            // Per-row dimensions cannot be validated once per batch.
            if w == 0.0 || h == 0.0 {
                polars_bail!(ComputeError: "ref_width and ref_height must be non-zero (row {})", i);
            }
            Ok(Some((w, h)))
        },
        |&(w, h), p| Ok(Some(Point::new(p.x / w, p.y / h))),
    )
}

/// Convert normalized coordinates to absolute pixel coordinates.
#[polars_expr(output_type_func=point_output)]
fn point_to_absolute(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_to_absolute",
        ToAbsolute { width, height }
    );
    map_points(
        inputs,
        &params,
        geom_calls!(),
        |params, i| Ok(Some((params.value(width, i)?, params.value(height, i)?))),
        |&(w, h), p| Ok(Some(Point::new(p.x * w, p.y * h))),
    )
}

/// Translate point by offset.
#[polars_expr(output_type_func=point_output)]
fn point_translate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_translate",
        Translate { dx, dy }
    );
    map_points(
        inputs,
        &params,
        geom_calls!(),
        |params, i| Ok(Some((params.value(dx, i)?, params.value(dy, i)?))),
        |&(dx, dy), p| Ok(Some(Point::new(p.x + dx, p.y + dy))),
    )
}

/// Scale point coordinates.
#[polars_expr(output_type_func=point_output)]
fn point_scale(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(params = inputs, kwargs, "point_scale", Scale { sx, sy });
    map_points(
        inputs,
        &params,
        geom_calls!(),
        |params, i| Ok(Some((params.value(sx, i)?, params.value(sy, i)?))),
        |&(sx, sy), p| Ok(Some(Point::new(p.x * sx, p.y * sy))),
    )
}

/// Rotate point around an origin (default `(0, 0)`) by angle (radians).
///
/// A null `origin` value nulls its row, as a null operand does everywhere; it
/// used to rotate about `(0, 0)` instead. The origin is one point per row.
#[polars_expr(output_type_func=point_output)]
fn point_rotate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_rotate",
        Rotate { angle, origin }
    );
    // Read through its reference: `origin` is optional, so its position is
    // not fixed once a per-row `angle` can also occupy an input slot.
    let origins = params.optional_column(origin).map(PointColumn::new);
    map_points(
        inputs,
        &params,
        geom_calls!(),
        |params, i| {
            let o = match &origins {
                Some(origins) => match origins.get(i)? {
                    Some(o) => o,
                    None => return Ok(None),
                },
                None => Point::new(0.0, 0.0),
            };
            Ok(Some((o, params.value(angle, i)?.sin_cos())))
        },
        |&(o, (sin_a, cos_a)), p| {
            let (dx, dy) = (p.x - o.x, p.y - o.y);
            Ok(Some(Point::new(
                dx * cos_a - dy * sin_a + o.x,
                dx * sin_a + dy * cos_a + o.y,
            )))
        },
    )
}

// ============================================================================
// Two points
// ============================================================================

/// Compute Euclidean distance between two points.
#[polars_expr(output_type_func=pair_f64_output)]
fn point_distance(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_distance",
        Distance { other }
    );
    zip_points(
        inputs,
        &params,
        other,
        geom_calls!(),
        "point_distance",
        |_, _, p, q| Ok((q.x - p.x).hypot(q.y - p.y)),
    )
}

/// Compute Manhattan (L1) distance between two points.
#[polars_expr(output_type_func=pair_f64_output)]
fn point_manhattan_distance(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_manhattan_distance",
        ManhattanDistance { other }
    );
    zip_points(
        inputs,
        &params,
        other,
        geom_calls!(),
        "point_manhattan_distance",
        |_, _, p, q| Ok((q.x - p.x).abs() + (q.y - p.y).abs()),
    )
}

/// Compute angle from this point to another in radians.
#[polars_expr(output_type_func=pair_f64_output)]
fn point_angle_to(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(params = inputs, kwargs, "point_angle_to", AngleTo { other });
    zip_points(
        inputs,
        &params,
        other,
        geom_calls!(),
        "point_angle_to",
        |_, _, p, q| Ok((q.y - p.y).atan2(q.x - p.x)),
    )
}

/// Compute midpoint between two points.
#[polars_expr(output_type_func=pair_point_output)]
fn point_midpoint(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_midpoint",
        Midpoint { other }
    );
    zip_points(
        inputs,
        &params,
        other,
        geom_calls!(),
        "point_midpoint",
        |_, _, p, q| Ok(Point::new((p.x + q.x) / 2.0, (p.y + q.y) / 2.0)),
    )
}

/// Linear interpolation between two points.
#[polars_expr(output_type_func=pair_point_output)]
fn point_interpolate(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_interpolate",
        Interpolate { other, t }
    );
    zip_points(
        inputs,
        &params,
        other,
        geom_calls!(),
        "point_interpolate",
        |params, i, p, q| {
            let t = params.value(t, i)?;
            Ok(Point::new(p.x + t * (q.x - p.x), p.y + t * (q.y - p.y)))
        },
    )
}

// ============================================================================
// A point and a contour or a bbox
// ============================================================================

/// Map each point of the row against the row's one contour, read as `C`: a
/// region ([`Contour`]) or a boundary ([`Outline`]).
fn with_contour<C: ReadContour, T: PointOutput>(
    inputs: &[Series],
    params: &GeomParams,
    contour: &ColumnRef,
    calls: &CallTracker,
    f: impl Fn(Point, &C) -> Option<T> + Sync,
) -> PolarsResult<Series> {
    let contours = ContourColumn::new(params.column(contour));
    map_points(
        inputs,
        params,
        calls,
        |_, i| contours.single_as::<C>(i),
        |c, p| Ok(f(p, c)),
    )
}

/// Compute minimum distance from point to contour boundary.
#[polars_expr(output_type_func=f64_output)]
fn point_distance_to_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_distance_to_contour",
        DistanceToContour { contour }
    );
    // A boundary measure: an open polyline is measured to its segments only.
    with_contour(inputs, &params, contour, geom_calls!(), |p, c: &Outline| {
        Some(measures::distance_to_outline(&p, c))
    })
}

/// Compute signed distance from point to contour boundary.
/// Negative if inside, positive if outside.
#[polars_expr(output_type_func=f64_output)]
fn point_signed_distance_to_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_signed_distance_to_contour",
        SignedDistanceToContour { contour }
    );
    // Inside/outside needs a region, so an open polyline is refused.
    with_contour(inputs, &params, contour, geom_calls!(), |p, c: &Contour| {
        let dist = measures::distance_to_contour(&p, c);
        Some(if predicates::contains_point(c, p.x, p.y) {
            -dist
        } else {
            dist
        })
    })
}

/// Find nearest point on contour boundary.
#[polars_expr(output_type_func=point_output)]
fn point_nearest_on_contour(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_nearest_on_contour",
        NearestOnContour { contour }
    );
    with_contour(inputs, &params, contour, geom_calls!(), |p, c: &Outline| {
        measures::nearest_point_on_outline(&p, c)
    })
}

/// Check if point is within bounding box.
#[polars_expr(output_type_func=bool_output)]
fn point_within_bbox(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    parse_as!(
        params = inputs,
        kwargs,
        "point_within_bbox",
        WithinBbox { bbox }
    );
    let bboxes = BBoxColumn::new(params.column(bbox));
    map_points(
        inputs,
        &params,
        geom_calls!(),
        |_, i| bboxes.single(i),
        |b, p| {
            Ok(Some(
                p.x >= b.x && p.x <= b.x + b.width && p.y >= b.y && p.y <= b.y + b.height,
            ))
        },
    )
}
