//! Typed op parameters. [`Param`], [`Literal`], [`ColumnRef`], [`NodeRef`]
//! and the catalogue types live with the engine ops in `view_buffer::mode`;
//! the per-row [`Values`] of one row of the plugin's parameter columns live
//! here.

use polars::prelude::*;
pub use view_buffer::mode::{
    literal_field, ColumnRef, FieldType, Literal, NodeRef, Param, Resolve, Values, SLOT_KEY,
};
use view_buffer::naming::{WireKind, WireScalar, WireValue};

use crate::params::ParamCtx;

/// One row of the plugin's parameter columns: where a `Wire` op's slots are
/// read when it is resolved for that row.
pub struct RowValues<'a> {
    pub row: usize,
    pub ctx: &'a ParamCtx<'a>,
}

impl Values for RowValues<'_> {
    type Error = PolarsError;

    fn value<T: WireScalar>(&self, slot: usize) -> PolarsResult<T> {
        let (row, ctx) = (self.row, self.ctx);
        let col = ctx.col(slot)?;
        let value = match T::KIND {
            WireKind::Int => WireValue::Int(col.get_i64(row, ctx)?),
            WireKind::Float => WireValue::Float(col.get_f64(row, ctx)?),
            WireKind::Bool => WireValue::Bool(col.get_bool(row, ctx)?),
            WireKind::Name => WireValue::Str(col.get_str(row, ctx)?),
        };
        T::from_wire(value).map_err(|e| {
            polars_err!(ComputeError:
                "Parameter column '{}' at row {}: {}", col.name(), row, e)
        })
    }
}

/// `Param::resolve` for one row, for the typed fields read outside an op —
/// the geometry namespaces' kwargs.
pub trait ParamExt<T> {
    /// The value at `row`: the literal, or the bound column's value.
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<T>;
}

impl<T: WireScalar> ParamExt<T> for Param<T> {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<T> {
        Resolve::resolve(self, &RowValues { row, ctx })
    }
}
