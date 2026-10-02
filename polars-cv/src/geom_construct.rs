//! Points and contours built from plain coordinate lists.
//!
//! `polars_cv.geometry.point_from_coords` / `contour_from_coords` /
//! `contour_set_from_coords` — the inverse of `.point.to_coords()` /
//! `.contour.to_coords()`. Python casts the input to `List(Float64)` nesting
//! (so integers and `Array(_, 2)` pairs arrive alike); every pair must hold
//! exactly two non-null numbers, or the row is an error naming it. The results
//! are built by the same assemblers the geometry functions use
//! ([`crate::point::assemble`], [`ContourOutput`]), so the published schema is
//! the canonical one.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use view_buffer::geometry::contour::{Contour, CoordOrder, Outline, Point};

use crate::geom_arity::{Arity, ContourOutput};
use crate::geom_schema::{contour_fields, point_struct_dtype};

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
fn pair(values: &Series, order: CoordOrder, row: usize) -> PolarsResult<Point> {
    let ca = values.f64()?;
    if ca.len() != 2 {
        polars_bail!(ComputeError:
            "a coordinate pair must hold exactly 2 numbers, got {} (row {})", ca.len(), row);
    }
    match (ca.get(0), ca.get(1)) {
        (Some(a), Some(b)) => Ok(order.point([a, b])),
        _ => polars_bail!(ComputeError: "a coordinate pair holds a null (row {})", row),
    }
}

/// A ring of pairs as points.
fn ring(pairs: &Series, order: CoordOrder, row: usize) -> PolarsResult<Vec<Point>> {
    pairs
        .list()
        .map(lists)?
        .map(|p| match p {
            Some(p) => pair(&p, order, row),
            None => polars_bail!(ComputeError: "a coordinate pair is null (row {})", row),
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
        DataType::Struct(contour_fields()),
    ))
}

fn contour_set_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("contours"),
        DataType::List(Box::new(DataType::Struct(contour_fields()))),
    ))
}

/// A point per `[a, b]` pair (`List(Float64)`).
#[polars_expr(output_type_func=point_dtype)]
fn geom_point_from_coords(inputs: &[Series], kwargs: CoordKwargs) -> PolarsResult<Series> {
    let order = kwargs.order()?;
    let rows = inputs[0]
        .list()
        .map(lists)?
        .enumerate()
        .map(|(i, p)| {
            p.map(|p| pair(&p, order, i).map(|p| vec![Some(p)]))
                .transpose()
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    crate::point::assemble(inputs[0].name().clone(), rows, Arity::Single)
}

/// A contour per ring of pairs (`List(List(Float64))`).
#[polars_expr(output_type_func=contour_dtype)]
fn geom_contour_from_coords(inputs: &[Series], kwargs: CoordKwargs) -> PolarsResult<Series> {
    let (order, closed) = (kwargs.order()?, kwargs.closed("contour_from_coords")?);
    let rows = inputs[0]
        .list()
        .map(lists)?
        .enumerate()
        .map(|(i, r)| {
            r.map(|r| ring(&r, order, i).map(|pts| vec![outline(pts, closed)]))
                .transpose()
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Outline::column(
        inputs[0].name().clone(),
        rows,
        Arity::Single,
        &DataType::Struct(contour_fields()),
    )
}

/// A contour set per list of rings (`List(List(List(Float64)))`).
#[polars_expr(output_type_func=contour_set_dtype)]
fn geom_contour_set_from_coords(inputs: &[Series], kwargs: CoordKwargs) -> PolarsResult<Series> {
    let (order, closed) = (kwargs.order()?, kwargs.closed("contour_set_from_coords")?);
    let rows = inputs[0]
        .list()
        .map(lists)?
        .enumerate()
        .map(|(i, set)| {
            set.map(|set| {
                set.list()
                    .map(lists)?
                    .map(|r| match r {
                        Some(r) => ring(&r, order, i).map(|pts| outline(pts, closed)),
                        None => polars_bail!(ComputeError: "a ring is null (row {})", i),
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })
            .transpose()
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Outline::column(
        inputs[0].name().clone(),
        rows,
        Arity::Set,
        &DataType::Struct(contour_fields()),
    )
}
