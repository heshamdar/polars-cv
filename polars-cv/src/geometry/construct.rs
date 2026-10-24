//! Points and contours built from plain coordinate lists.
//!
//! `polars_cv.geometry.point_from_coords` / `contour_from_coords` /
//! `contour_set_from_coords` — the inverse of `.point.to_coords()` /
//! `.contour.to_coords()`. Python casts the input to `List(Float64)` nesting
//! (so integers and `Array(_, 2)` pairs arrive alike); every pair must hold
//! exactly two non-null, finite numbers, or the row is an error naming it. The results
//! are built by the same assemblers the geometry functions use
//! ([`crate::geometry::point::assemble`], [`ContourOutput`]), so the published schema is
//! the canonical one.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use view_buffer::geometry::contour::{Contour, CoordOrder, Outline, Point};

use crate::geometry::arity::{Arity, ContourOutput};
use crate::geometry::schema::{contour_struct_dtype, point_struct_dtype};

/// Static kwargs of the constructors. Closed: an argument Python sends and
/// Rust does not read is drift.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordKwargs {
    /// `"xy"` or `"yx"`, read through [`CoordOrder`]'s own table.
    order: String,
    /// Contours only: whether each is a closed region (`false`: an open
    /// polyline).
    #[serde(default)]
    closed: Option<bool>,
}

impl CoordKwargs {
    fn order(&self) -> PolarsResult<CoordOrder> {
        view_buffer::naming::lookup(CoordOrder::NAMED, &self.order).ok_or_else(
            || polars_err!(ComputeError: "order must be \"xy\" or \"yx\", got {:?}", self.order),
        )
    }

    fn closed(&self, what: &str) -> PolarsResult<bool> {
        self.closed
            .ok_or_else(|| polars_err!(ComputeError: "internal: {} needs `closed`", what))
    }
}

/// A list column's rows, each its values as a series (`None` for null).
fn lists(ca: &ListChunked) -> impl Iterator<Item = Option<Series>> + '_ {
    (0..ca.len()).map(move |i| ca.get_as_series(i))
}

/// One `[a, b]` pair as a point, or the error naming `row`.
fn pair(values: &Series, order: CoordOrder) -> PolarsResult<Point> {
    let ca = values.f64()?;
    if ca.len() != 2 {
        polars_bail!(ComputeError:
            "a coordinate pair must hold exactly 2 numbers, got {}", ca.len());
    }
    match (ca.get(0), ca.get(1)) {
        // Every geometry reader refuses a non-finite coordinate (it has no
        // position); building one would only defer that error.
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Ok(order.point([a, b])),
        (Some(a), Some(b)) => polars_bail!(ComputeError:
            "a coordinate pair holds a non-finite number, [{}, {}]", a, b),
        _ => polars_bail!(ComputeError: "a coordinate pair holds a null"),
    }
}

/// A row's error, naming the row (`row_ref`).
fn in_row(e: PolarsError, row: usize, len: usize) -> PolarsError {
    e.wrap_msg(|m| crate::row_ref::at_row(m, row, len))
}

/// A ring of pairs as points.
fn ring(pairs: &Series, order: CoordOrder) -> PolarsResult<Vec<Point>> {
    pairs
        .list()
        .map(lists)?
        .map(|p| match p {
            Some(p) => pair(&p, order),
            None => polars_bail!(ComputeError: "a coordinate pair is null"),
        })
        .collect()
}

fn outline(points: Vec<Point>, closed: bool) -> Outline {
    if closed {
        Outline::Closed(Contour::new(points))
    } else {
        Outline::Open(points)
    }
}

fn point_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("point"),
        point_struct_dtype(),
    ))
}

fn contour_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("contour"),
        contour_struct_dtype(),
    ))
}

fn contour_set_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("contours"),
        DataType::List(Box::new(contour_struct_dtype())),
    ))
}

/// A point per `[a, b]` pair (`List(Float64)`).
#[polars_expr(output_type_func=point_dtype)]
fn geom_point_from_coords(inputs: &[Series], kwargs: CoordKwargs) -> PolarsResult<Series> {
    let order = kwargs.order()?;
    let len = inputs[0].len();
    let rows = inputs[0]
        .list()
        .map(lists)?
        .enumerate()
        .map(|(i, p)| {
            p.map(|p| pair(&p, order).map(|p| vec![Some(p)]))
                .transpose()
                .map_err(|e| in_row(e, i, len))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    crate::geometry::point::assemble(inputs[0].name().clone(), rows, Arity::Single)
}

/// A contour per ring of pairs (`List(List(Float64))`).
#[polars_expr(output_type_func=contour_dtype)]
fn geom_contour_from_coords(inputs: &[Series], kwargs: CoordKwargs) -> PolarsResult<Series> {
    let (order, closed) = (kwargs.order()?, kwargs.closed("contour_from_coords")?);
    let len = inputs[0].len();
    let rows = inputs[0]
        .list()
        .map(lists)?
        .enumerate()
        .map(|(i, r)| {
            r.map(|r| ring(&r, order).map(|pts| vec![outline(pts, closed)]))
                .transpose()
                .map_err(|e| in_row(e, i, len))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Outline::column(
        inputs[0].name().clone(),
        rows,
        Arity::Single,
        &contour_struct_dtype(),
    )
}

/// A contour set per list of rings (`List(List(List(Float64)))`).
#[polars_expr(output_type_func=contour_set_dtype)]
fn geom_contour_set_from_coords(inputs: &[Series], kwargs: CoordKwargs) -> PolarsResult<Series> {
    let (order, closed) = (kwargs.order()?, kwargs.closed("contour_set_from_coords")?);
    let len = inputs[0].len();
    let rows = inputs[0]
        .list()
        .map(lists)?
        .enumerate()
        .map(|(i, set)| {
            set.map(|set| {
                set.list()
                    .map(lists)?
                    .map(|r| match r {
                        Some(r) => ring(&r, order).map(|pts| outline(pts, closed)),
                        None => polars_bail!(ComputeError: "a ring is null"),
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })
            .transpose()
            .map_err(|e| in_row(e, i, len))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Outline::column(
        inputs[0].name().clone(),
        rows,
        Arity::Set,
        &contour_struct_dtype(),
    )
}
