//! `patch_grid`: the patches that tile each row's `height × width` image, as
//! a list of `{row, col, top, left, height, width}` per row.
//!
//! The grid is [`view_buffer::geometry::grid::PatchGrid`]'s, the one
//! definition; this function only lays it out as a column. One row in, one
//! list out, so the call stays elementwise (streaming) and Polars' `explode`
//! is what turns patches into rows. The field names are `crop`'s keywords, so
//! an exploded row feeds a crop directly.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use view_buffer::geometry::grid::{GridEdge, PatchGrid};

/// The static kwargs: the grid itself. Closed, so a misspelt keyword is an
/// error rather than a default.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchGridKwargs {
    /// `[height, width]` of every patch.
    size: [u32; 2],
    /// `[rows, cols]` between patch origins.
    stride: [u32; 2],
    /// A [`GridEdge`] spelling.
    edge: String,
}

const FIELDS: [&str; 6] = ["row", "col", "top", "left", "height", "width"];

/// One patch's struct fields, all `UInt32`, in [`FIELDS`] order.
fn cell_fields() -> Vec<Field> {
    FIELDS
        .iter()
        .map(|&n| Field::new(PlSmallStr::from_static(n), DataType::UInt32))
        .collect()
}

fn patch_grid_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let name = input_fields.first().map_or_else(
        || PlSmallStr::from_static("patch_grid"),
        |f| f.name().clone(),
    );
    Ok(Field::new(
        name,
        DataType::List(Box::new(DataType::Struct(cell_fields()))),
    ))
}

/// A `UInt32` view of a height/width input; a value that does not fit (a
/// negative or fractional size) is an error, not a wrapped or null one.
fn extent(series: &Series, what: &str) -> PolarsResult<UInt32Chunked> {
    let cast = series.strict_cast(&DataType::UInt32).map_err(|_| {
        polars_err!(ComputeError:
            "patch_grid(): {} must hold non-negative integers, got {}", what, series.dtype())
    })?;
    Ok(cast.u32()?.clone())
}

/// Every row's patches.
#[polars_expr(output_type_func=patch_grid_output_type)]
fn patch_grid(inputs: &[Series], kwargs: PatchGridKwargs) -> PolarsResult<Series> {
    use polars_arrow::array::{ListArray, PrimitiveArray, StructArray};
    use polars_arrow::offset::Offsets;

    let [height, width] = inputs else {
        polars_bail!(ComputeError: "patch_grid() takes a height and a width column");
    };
    let edge = view_buffer::naming::lookup(GridEdge::NAMED, &kwargs.edge).ok_or_else(|| {
        polars_err!(ComputeError:
            "patch_grid(): unknown edge {:?} (expected one of {:?})",
            kwargs.edge, view_buffer::naming::names(GridEdge::NAMED))
    })?;
    let grid = PatchGrid::new(kwargs.size, kwargs.stride, edge)
        .map_err(|e| polars_err!(ComputeError: "patch_grid(): {}", e))?;
    let (heights, widths) = (extent(height, "height")?, extent(width, "width")?);
    let n = heights.len().max(widths.len());
    let broadcast = |ca: &UInt32Chunked, i: usize| ca.get(if ca.len() == 1 { 0 } else { i });
    if heights.len() != widths.len() && heights.len() != 1 && widths.len() != 1 {
        polars_bail!(ComputeError: "patch_grid(): height and width have different lengths");
    }

    let [ph, pw] = grid.size();
    let mut columns: [Vec<u32>; 6] = Default::default();
    let mut lengths = Vec::with_capacity(n);
    let mut valid = Vec::with_capacity(n);
    for i in 0..n {
        match (broadcast(&heights, i), broadcast(&widths, i)) {
            (Some(h), Some(w)) => {
                let cells = grid.cells(h, w);
                lengths.push(cells.len());
                valid.push(true);
                for c in cells {
                    for (col, v) in columns
                        .iter_mut()
                        .zip([c.row, c.col, c.top, c.left, ph, pw])
                    {
                        col.push(v);
                    }
                }
            }
            _ => {
                lengths.push(0);
                valid.push(false);
            }
        }
    }

    let arrow_fields: Vec<_> = cell_fields()
        .iter()
        .map(|f| f.to_arrow(CompatLevel::newest()))
        .collect();
    let len = columns[0].len();
    let values = StructArray::try_new(
        ArrowDataType::Struct(arrow_fields),
        len,
        columns
            .into_iter()
            .map(|v| PrimitiveArray::from_vec(v).boxed())
            .collect(),
        None,
    )?;
    let validity = valid
        .contains(&false)
        .then(|| valid.into_iter().collect::<polars_arrow::bitmap::Bitmap>());
    let offsets = Offsets::<i64>::try_from_lengths(lengths.into_iter())?;
    let dtype =
        ListArray::<i64>::default_datatype(polars_arrow::array::Array::dtype(&values).clone());
    let list = ListArray::<i64>::try_new(dtype, offsets.into(), values.boxed(), validity)?;
    Series::from_arrow(height.name().clone(), list.boxed())
}
