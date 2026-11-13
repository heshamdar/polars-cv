//! The `.bbox` accessors that measure or transform each box (`BBoxFn`), and
//! non-maximum suppression over a row's box set.
//!
//! A box column holds one box per row or a set ([`Arity::of_bboxes`]), read
//! through [`BBoxColumn`], which refuses a null, non-finite or negative-size
//! field. Each per-box accessor maps over whichever it is given, as the
//! contour accessors do, and assembles its column through [`ContourOutput`],
//! so the declared type ([`bbox_field`]) and the data wrap alike. The
//! arithmetic is [`BoundingBox`]'s.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use view_buffer::geometry::contour::{BoundingBox, Contour};
use view_buffer::geometry::pairwise;
use view_buffer::mode::Wire;

use crate::namespaces::arity::{Arity, ContourOutput};
use crate::namespaces::columns::BBoxColumn;
use crate::namespaces::fns::BBoxFn;
use crate::namespaces::params::{parsed_as_another, GeomKwargs, GeomParams};
use crate::namespaces::schema::{bbox_anyvalue, bbox_struct_dtype, contour_struct_dtype};

/// The declared type of a per-box accessor whose result per box is `elem`.
fn bbox_field(input_fields: &[Field], name: &'static str, elem: DataType) -> PolarsResult<Field> {
    let input = input_fields
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "{} takes a bbox column", name))?;
    Ok(Field::new(
        input.name().clone(),
        Arity::of_bboxes(input.dtype()).wrap(elem),
    ))
}

/// Run `compute` on every box of every row, keeping the column's arity.
fn map_bboxes<R: ContourOutput + Send>(
    series: &Series,
    params: &GeomParams,
    elem: DataType,
    compute: impl Fn(&BoundingBox, &GeomParams, usize) -> PolarsResult<R> + Sync,
) -> PolarsResult<Series> {
    let column = BBoxColumn::new(series);
    let rows = params.map_rows(series.len(), |params, i| {
        let Some(boxes) = column.row(i)? else {
            return Ok(None);
        };
        boxes
            .iter()
            .map(|b| compute(b, params, i))
            .collect::<PolarsResult<Vec<_>>>()
            .map(Some)
    })?;
    R::column(
        series.name().clone(),
        rows,
        Arity::of_bboxes(series.dtype()),
        &elem,
    )
}

/// A per-box accessor: its declared type and its plugin function, parsing
/// its arguments as `BBoxFn::$var`.
macro_rules! bbox_accessor {
    (
        $(#[$meta:meta])*
        map fn $name:ident / $out_ty:ident -> $elem:expr;
        parse $var:ident $({ $($field:ident),* })?;
        |$b:ident, $params:ident, $row:ident| $body:expr
    ) => {
        fn $out_ty(input_fields: &[Field]) -> PolarsResult<Field> {
            bbox_field(input_fields, stringify!($name), $elem)
        }
        $(#[$meta])*
        #[polars_expr(output_type_func=$out_ty)]
        fn $name(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
            let (op, geom_params) =
                GeomParams::parse::<BBoxFn<Wire>>(inputs, kwargs, stringify!($name))?;
            let BBoxFn::$var $({ $($field),* })? = &op else {
                return Err(parsed_as_another(stringify!($name)));
            };
            map_bboxes(&inputs[0], &geom_params, $elem, |$b, $params, $row| $body)
        }
    };
}

/// A box result, or the refusal `why` names.
fn bbox(
    result: Option<BoundingBox>,
    why: impl FnOnce() -> String,
) -> PolarsResult<AnyValue<'static>> {
    match result {
        Some(b) => Ok(bbox_anyvalue(Some(b))),
        None => Err(polars_err!(ComputeError: "{}", why())),
    }
}

/// An image size a box is read against: `width` and `height` at least 0
/// (positive when `positive`).
fn image_size(width: f64, height: f64, positive: bool) -> PolarsResult<(f64, f64)> {
    let ok = |v: f64| if positive { v > 0.0 } else { v >= 0.0 };
    if !(ok(width) && ok(height)) {
        let bound = if positive { "positive" } else { "at least 0" };
        polars_bail!(ComputeError:
            "width and height must be {bound}, got {width} x {height}");
    }
    Ok((width, height))
}

bbox_accessor! {
    /// Each box's area.
    map fn bbox_area / bbox_area_output_type -> DataType::Float64;
    parse Area;
    |b, _params, _row| Ok(AnyValue::Float64(b.area()))
}

bbox_accessor! {
    /// Each box's four numbers in a format.
    map fn bbox_to_coords / bbox_to_coords_output_type
        -> DataType::Array(Box::new(DataType::Float64), 4);
    parse ToCoords { format };
    |b, params, row| {
        let coords = b.coords(params.value(format, row)?);
        Ok(AnyValue::Array(Series::new(PlSmallStr::EMPTY, coords.to_vec()), 4))
    }
}

bbox_accessor! {
    /// Each box as a closed contour.
    map fn bbox_to_contour / bbox_to_contour_output_type -> contour_struct_dtype();
    parse ToContour;
    |b, _params, _row| Ok(Contour::new(b.corners()))
}

bbox_accessor! {
    /// Each box moved.
    map fn bbox_translate / bbox_translate_output_type -> bbox_struct_dtype();
    parse Translate { dx, dy };
    |b, params, row| Ok(bbox_anyvalue(Some(
        b.translated(params.value(dx, row)?, params.value(dy, row)?),
    )))
}

bbox_accessor! {
    /// Each box scaled about the image origin.
    map fn bbox_scale / bbox_scale_output_type -> bbox_struct_dtype();
    parse Scale { sx, sy };
    |b, params, row| {
        let (sx, sy) = (params.value(sx, row)?, params.value(sy, row)?);
        bbox(b.scaled(sx, sy), || {
            format!("scale factors must be at least 0, got sx={sx}, sy={sy}")
        })
    }
}

bbox_accessor! {
    /// Each box in the [0, 1] coordinates of its image.
    map fn bbox_normalize / bbox_normalize_output_type -> bbox_struct_dtype();
    parse Normalize { width, height };
    |b, params, row| {
        let (w, h) = image_size(params.value(width, row)?, params.value(height, row)?, true)?;
        bbox(b.scaled(1.0 / w, 1.0 / h), || unreachable!("positive sizes scale"))
    }
}

bbox_accessor! {
    /// Each box in the pixel coordinates of its image.
    map fn bbox_to_absolute / bbox_to_absolute_output_type -> bbox_struct_dtype();
    parse ToAbsolute { width, height };
    |b, params, row| {
        let (w, h) = image_size(params.value(width, row)?, params.value(height, row)?, false)?;
        bbox(b.scaled(w, h), || unreachable!("sizes at least 0 scale"))
    }
}

bbox_accessor! {
    /// The part of each box inside its image.
    map fn bbox_clip / bbox_clip_output_type -> bbox_struct_dtype();
    parse Clip { width, height };
    |b, params, row| {
        let (w, h) = image_size(params.value(width, row)?, params.value(height, row)?, false)?;
        Ok(bbox_anyvalue(Some(b.clipped(w, h))))
    }
}

bbox_accessor! {
    /// Each box grown (or shrunk) on every side.
    map fn bbox_pad / bbox_pad_output_type -> bbox_struct_dtype();
    parse Pad { amount };
    |b, params, row| {
        let amount = params.value(amount, row)?;
        bbox(b.padded(amount), || format!(
            "padding a {} x {} box by {amount} leaves it a negative size",
            b.width, b.height
        ))
    }
}

bbox_accessor! {
    /// Each box scaled about its centre.
    map fn bbox_expand / bbox_expand_output_type -> bbox_struct_dtype();
    parse Expand { factor };
    |b, params, row| {
        let factor = params.value(factor, row)?;
        bbox(b.expanded(factor), || format!("factor must be at least 0, got {factor}"))
    }
}

fn bbox_nms_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let input = input_fields
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "bbox_nms takes a bbox column"))?;
    Ok(Field::new(
        input.name().clone(),
        DataType::List(Box::new(DataType::UInt32)),
    ))
}

/// Greedy non-maximum suppression over each row's box set: the kept
/// positions, highest score first ([`pairwise::bbox_nms`]).
///
/// The positions index the row's list as stored, so every box must be
/// present: a null box in the set would shift them, and is refused, as is a
/// score list of another length, a null or non-finite score, or a threshold
/// outside [0, 1].
#[polars_expr(output_type_func=bbox_nms_output_type)]
fn bbox_nms(inputs: &[Series], kwargs: GeomKwargs) -> PolarsResult<Series> {
    const NAME: &str = "bbox_nms";
    let (op, params) = GeomParams::parse::<BBoxFn<Wire>>(inputs, kwargs, NAME)?;
    let BBoxFn::Nms {
        scores,
        iou_threshold,
    } = &op
    else {
        return Err(parsed_as_another(NAME));
    };
    let series = &inputs[0];
    if Arity::of_bboxes(series.dtype()) != Arity::Set {
        polars_bail!(ComputeError:
            "nms takes a set of boxes per row (List[BBOX_SCHEMA]); this column holds \
             one box per row. Group the boxes into a list first (`.implode()` over \
             an image's rows)."
        );
    }
    let stored = series.list()?;
    let boxes = BBoxColumn::new(series);
    let score_lists = params
        .column(scores)
        .strict_cast(&DataType::List(Box::new(DataType::Float64)))
        .map_err(|_| {
            polars_err!(ComputeError:
                "nms scores must be a list of numbers per row, got {}",
                params.column(scores).dtype()
            )
        })?;
    let score_lists = score_lists.list()?;
    let rows = params.map_rows(series.len(), |params, i| {
        let (Some(set), Some(row_scores)) = (boxes.row(i)?, score_lists.get_as_series(i)) else {
            return Ok(None);
        };
        let stored_len = stored.get_as_series(i).map_or(0, |s| s.len());
        if set.len() != stored_len {
            polars_bail!(ComputeError:
                "nms returns positions in the row's box list, so every box must be \
                 present: {} of {stored_len} are null",
                stored_len - set.len()
            );
        }
        if row_scores.len() != set.len() {
            polars_bail!(ComputeError:
                "nms needs one score per box: {} boxes, {} scores",
                set.len(),
                row_scores.len()
            );
        }
        let values = row_scores
            .f64()?
            .iter()
            .map(|s| match s {
                Some(s) if s.is_finite() => Ok(s),
                Some(s) => Err(polars_err!(ComputeError: "nms scores must be finite, got {s}")),
                None => Err(polars_err!(ComputeError: "nms scores must not be null")),
            })
            .collect::<PolarsResult<Vec<f64>>>()?;
        let threshold = params.value(iou_threshold, i)?;
        if !(0.0..=1.0).contains(&threshold) {
            polars_bail!(ComputeError: "iou_threshold must be in [0, 1], got {threshold}");
        }
        let kept: Vec<u32> = pairwise::bbox_nms(&set, &values, threshold)
            .into_iter()
            .map(|k| k as u32)
            .collect();
        Ok(Some(vec![AnyValue::List(Series::new(
            PlSmallStr::EMPTY,
            kept,
        ))]))
    })?;
    <AnyValue<'static> as ContourOutput>::column(
        series.name().clone(),
        rows,
        Arity::Single,
        &DataType::List(Box::new(DataType::UInt32)),
    )
}
