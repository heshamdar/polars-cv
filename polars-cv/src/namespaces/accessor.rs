//! The op-backed `.contour` accessors: each is a pipeline step, run as one.
//!
//! An accessor whose function is a pipeline op (`.contour.offset(...)` is
//! `Pipeline.offset(...)`) parses the op's definition, resolves it per row
//! through [`GeomParams::step`] (the op's check, as `TypedOp::resolve` runs
//! it) and executes it with [`execute_geometry_op`] on the row's outlines —
//! the pipeline executor's own dispatch. So the two surfaces cannot answer,
//! refuse or name a refusal differently; each accessor's only code is how it
//! presents the step's output, its [`Form`].
//!
//! [`op_accessors!`](crate::op_accessors) declares them, and in the same
//! statement [`OP_ACCESSORS`](crate::namespaces::contour::OP_ACCESSORS), the table the
//! geometry catalogue and the build check read.

use polars::prelude::*;
use std::sync::Arc;

use crate::graph::node_output::NodeOutput;
use view_buffer::geometry::contour::Outline;
use view_buffer::mode::Wire;
use view_buffer::GeometryOp;

use crate::namespaces::arity::{Arity, ContourOutput};
use crate::namespaces::columns::ContourColumn;
use crate::namespaces::exec::execute_geometry_op;
use crate::namespaces::params::{GeomKwargs, GeomParams};
use crate::namespaces::schema::{
    bbox_anyvalue, bbox_struct_dtype, contour_struct_dtype, point_anyvalue, point_struct_dtype,
};
use view_buffer::geometry::contour::BoundingBox;

/// How an op-backed accessor presents its step's output. The pipeline gives
/// a measure as one flat vector per row; an accessor gives one value per
/// member, in the column's own arity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// A number per member (`area`).
    Number,
    /// An `{x, y}` point per member (`centroid`).
    Point,
    /// An `{x, y, width, height}` box per member (`bounding_box`).
    BBox,
    /// A contour per member (`translate`), in the column's arity.
    Contour,
    /// A contour set per row, whatever arity came in (`offset`): how many
    /// is the data's.
    Set,
}

impl Form {
    /// One member's type.
    fn elem(self) -> DataType {
        match self {
            Form::Number => DataType::Float64,
            Form::Point => point_struct_dtype(),
            Form::BBox => bbox_struct_dtype(),
            Form::Contour | Form::Set => contour_struct_dtype(),
        }
    }

    /// The result's arity over an input column of `input`.
    fn arity(self, input: &DataType) -> Arity {
        match self {
            Form::Set => Arity::Set,
            _ => Arity::of(input),
        }
    }

    /// One member's value from its slice of a measure vector.
    fn value(self, member: &[f64]) -> AnyValue<'static> {
        match (self, member) {
            (Form::Point, &[x, y]) => point_anyvalue(x, y),
            (Form::BBox, &[x, y, width, height]) => bbox_anyvalue(Some(BoundingBox {
                x,
                y,
                width,
                height,
            })),
            (_, &[v]) => AnyValue::Float64(v),
            _ => unreachable!("chunked by Form::width"),
        }
    }

    /// How many numbers of a measure vector are one member's.
    fn width(self) -> usize {
        match self {
            Form::Point => 2,
            Form::BBox => 4,
            _ => 1,
        }
    }
}

/// The form of the op-backed accessor `name`, from [`OP_ACCESSORS`].
///
/// [`OP_ACCESSORS`]: crate::namespaces::contour::OP_ACCESSORS
fn form_of(name: &str) -> PolarsResult<Form> {
    crate::namespaces::contour::OP_ACCESSORS
        .iter()
        .find(|(op, _, _)| *op == name)
        .map(|(_, _, form)| *form)
        .ok_or_else(
            || polars_err!(ComputeError: "internal: '{}' is not an op-backed accessor", name),
        )
}

/// The declared output of the op-backed accessor `name`.
pub(crate) fn field(input_fields: &[Field], name: &'static str) -> PolarsResult<Field> {
    let form = form_of(name)?;
    let input = input_fields
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "{} takes a contour column", name))?;
    let dtype = form.arity(input.dtype()).wrap(form.elem());
    Ok(Field::new(input.name().clone(), dtype))
}

/// Run the op-backed accessor `name`: each row's outlines through the step
/// its arguments resolve to for that row, presented in its [`Form`].
pub(crate) fn run(
    inputs: &[Series],
    kwargs: GeomKwargs,
    name: &'static str,
) -> PolarsResult<Series> {
    let form = form_of(name)?;
    let (op, params) = GeomParams::parse::<GeometryOp<Wire>>(inputs, kwargs, name)?;
    let column = ContourColumn::new(&inputs[0]);
    let arity = form.arity(inputs[0].dtype());
    let out_name = inputs[0].name().clone();
    match form {
        Form::Contour | Form::Set => {
            let rows = step_rows(
                &column,
                inputs[0].len(),
                &op,
                &params,
                |output, _| match output {
                    NodeOutput::Contours(c) => Ok(Arc::unwrap_or_clone(c)),
                    other => Err(produced(name, &other)),
                },
            )?;
            Outline::column(out_name, rows, arity, &form.elem())
        }
        Form::Number | Form::Point | Form::BBox => {
            let rows = step_rows(&column, inputs[0].len(), &op, &params, |output, members| {
                let NodeOutput::Vector(values) = &output else {
                    return Err(produced(name, &output));
                };
                if values.len() != members * form.width() {
                    polars_bail!(ComputeError:
                        "internal: {} measured {} values for {} members",
                        name, values.len(), members
                    );
                }
                Ok(values.chunks(form.width()).map(|m| form.value(m)).collect())
            })?;
            AnyValue::column(out_name, rows, arity, &form.elem())
        }
    }
}

/// Each row's step output, presented by `present` (which also gets the
/// row's member count), in row order; a null row (or a null per-row value
/// under `on_null("null")`) is `None`.
fn step_rows<R: Send>(
    column: &ContourColumn<'_>,
    len: usize,
    op: &GeometryOp<Wire>,
    params: &GeomParams,
    present: impl Fn(NodeOutput, usize) -> PolarsResult<Vec<R>> + Sync,
) -> PolarsResult<Vec<Option<Vec<R>>>> {
    // As the pipeline resolves a step: once when no field is per-row (its
    // literals were checked as the expression was built), else per row.
    let mut per_row = false;
    op.visit_inputs(&mut |_, _| per_row = true);
    let fixed = if per_row {
        None
    } else {
        Some(params.step(op, None)?)
    };
    params.map_rows(len, |params, row| {
        let Some(outlines) = column.outlines(row)? else {
            return Ok(None);
        };
        let resolved;
        let step = match &fixed {
            Some(step) => step,
            None => {
                resolved = params.step(op, Some(row))?;
                &resolved
            }
        };
        let members = outlines.len();
        let output = execute_geometry_op(NodeOutput::from_contours(outlines), step)
            .map_err(|e| polars_err!(ComputeError: "{}", e))?;
        present(output, members).map(Some)
    })
}

fn produced(name: &str, output: &NodeOutput) -> PolarsError {
    polars_err!(ComputeError:
        "internal: {} produced {:?}, which its form does not present",
        name, output.domain()
    )
}

/// Declare the op-backed `.contour` accessors and, in the same statement,
/// [`OP_ACCESSORS`](crate::namespaces::contour::OP_ACCESSORS): each line is the plugin
/// function (named for its op), its output-type function, its Python method
/// and its [`Form`]. A function and its table entry cannot be written apart.
#[macro_export]
macro_rules! op_accessors {
    ($($(#[$meta:meta])* op fn $name:ident / $out_ty:ident => $method:literal, $form:expr;)*) => {
        /// The `.contour` accessor methods that are pipeline ops:
        /// `(op, method, form)`. Each is that [`GeometryOp`] variant — its
        /// fields, defaults and doc — run as the pipeline's step
        /// (`crate::namespaces::accessor`).
        pub const OP_ACCESSORS: &[(&str, &str, $crate::namespaces::accessor::Form)] =
            &[$((stringify!($name), $method, $form)),*];
        $(
            fn $out_ty(input_fields: &[Field]) -> PolarsResult<Field> {
                $crate::namespaces::accessor::field(input_fields, stringify!($name))
            }
            $(#[$meta])*
            #[polars_expr(output_type_func=$out_ty)]
            fn $name(
                inputs: &[Series],
                kwargs: $crate::namespaces::params::GeomKwargs,
            ) -> PolarsResult<Series> {
                $crate::namespaces::accessor::run(inputs, kwargs, stringify!($name))
            }
        )*
    };
}
