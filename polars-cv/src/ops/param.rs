//! Typed op parameters. [`Param`], [`Literal`] and the catalogue types live
//! with the engine ops in `view_buffer::mode`; the two graph-level field
//! kinds — [`ColumnRef`] and [`NodeRef`] — and the per-row [`Values`] of one
//! row of the plugin's parameter columns live here.

use polars::prelude::*;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub use view_buffer::mode::{
    as_slot, literal_field, FieldType, Literal, Param, Resolve, TypeDesc, Values, SLOT_KEY,
};
use view_buffer::naming::{WireKind, WireScalar, WireValue};

use crate::params::ParamCtx;

/// An input column the step reads as *data* — `label_reduce`'s contour set —
/// rather than a parameter value resolved per row. Always a slot: a literal
/// has nowhere to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnRef(pub usize);

/// Another graph node, by id: the operand of a binary op, a mask, a merged
/// channel. Graph topology, so never per-row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRef(pub String);

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
        // Transitional (C4b): a not-yet-combined op resolved for its rules.
        if ctx.is_planning() {
            return Ok(T::planning_value());
        }
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

/// `Param::resolve` for one row, as the not-yet-combined typed ops call it.
pub trait ParamExt<T> {
    /// The value at `row`: the literal, or the bound column's value.
    ///
    /// At plan time (`ParamCtx::planning`) a per-row parameter has no row to
    /// read, so it takes an arbitrary valid value. That is sound only because
    /// such a parameter is per-row eligible, i.e. has no effect on the schema.
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<T>;
}

impl<T: WireScalar> ParamExt<T> for Param<T> {
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<T> {
        match *self {
            Param::Lit(v) => Ok(v),
            Param::Slot(_) if ctx.is_planning() => Ok(T::planning_value()),
            Param::Slot(_) => Resolve::resolve(self, &RowValues { row, ctx }),
        }
    }
}

impl Serialize for ColumnRef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Param::<i64>::Slot(self.0).serialize(s)
    }
}

impl<'de> Deserialize<'de> for ColumnRef {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        match as_slot(&value) {
            Some(slot) => slot.map(ColumnRef).map_err(D::Error::custom),
            None => Err(D::Error::custom(format!(
                "this parameter is an input column and must be a Polars \
                 expression, got the literal {value}"
            ))),
        }
    }
}

impl Serialize for NodeRef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for NodeRef {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match serde_json::Value::deserialize(d)? {
            serde_json::Value::String(id) => Ok(NodeRef(id)),
            other => Err(D::Error::custom(format!(
                "expected a graph node id (a string), got {other}"
            ))),
        }
    }
}

impl FieldType for ColumnRef {
    fn describe() -> TypeDesc {
        TypeDesc::Column
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        f(self.0);
    }
}

impl FieldType for NodeRef {
    fn describe() -> TypeDesc {
        TypeDesc::Node
    }
    fn visit_slots(&self, _f: &mut dyn FnMut(usize)) {}
}
