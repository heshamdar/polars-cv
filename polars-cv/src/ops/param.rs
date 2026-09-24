//! Typed op parameters: [`Param<T>`] (per-row capable) and [`Literal<T>`]
//! (structural), and the [`FieldType`] trait the catalogue describes them by.
//!
//! The wire form is the value itself — `224`, `"bilinear"`, `[1.0, 0.0]` — or,
//! for a per-row parameter, `{"$slot": n}`: the index of the plugin input
//! column that carries it. Deserialization is hand-written rather than
//! `#[serde(untagged)]`, which is ambiguous for integer `T` and replaces every
//! error with "data did not match any variant".
//!
//! The eligibility rule (a parameter may be per-row iff it has no effect on
//! the output shape, rank or dtype) is a type here: a `$slot` in a
//! [`Literal`] is a deserialization error.

use polars::prelude::*;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use view_buffer::naming::{WireKind, WireScalar, WireValue};

use crate::params::ParamCtx;

/// The wire key marking a per-row parameter.
pub const SLOT_KEY: &str = "$slot";

/// A parameter that is a literal or, per row, an input column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Param<T> {
    /// A value fixed for every row.
    Lit(T),
    /// The plugin input column at this position holds the value for each row.
    Slot(usize),
}

/// A structural parameter: always a literal, never per-row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Literal<T>(pub T);

/// An input column the step reads as *data* — `label_reduce`'s contour set —
/// rather than a parameter value resolved per row. Always a slot: a literal
/// has nowhere to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnRef(pub usize);

/// Another graph node, by id: the operand of a binary op, a mask, a merged
/// channel. Graph topology, so never per-row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRef(pub String);

impl<T: WireScalar> Param<T> {
    /// The value at `row`: the literal, or the bound column's value.
    ///
    /// Under a plan-time probe (`ParamCtx::probe`) a named enum or flag cannot
    /// be read from the integer placeholder, so it takes an arbitrary valid
    /// value. That is sound only because such a parameter is per-row
    /// eligible, i.e. has no effect on the schema being probed.
    pub fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<T> {
        let idx = match *self {
            Param::Lit(v) => return Ok(v),
            Param::Slot(idx) => idx,
        };
        let col = ctx.col(idx)?;
        let value = match T::KIND {
            WireKind::Int => WireValue::Int(col.get_i64(row, ctx)?),
            WireKind::Float => WireValue::Float(col.get_f64(row, ctx)?),
            WireKind::Bool | WireKind::Name if ctx.is_probe() => return Ok(T::probe_value()),
            WireKind::Bool => WireValue::Bool(col.get_bool(row, ctx)?),
            WireKind::Name => WireValue::Str(col.get_str(row, ctx)?),
        };
        T::from_wire(value).map_err(|e| {
            polars_err!(ComputeError:
                "Parameter column '{}' at row {}: {}", col.name(), row, e)
        })
    }
}

impl<T: WireScalar> Literal<T> {
    /// The value.
    pub fn get(&self) -> T {
        self.0
    }
}

fn scalar_from_json<T: WireScalar>(value: &serde_json::Value) -> Result<T, String> {
    use serde_json::Value;
    let wire = match value {
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => WireValue::Int(i),
            (None, Some(_)) if n.is_u64() => {
                return Err(format!("{n} is out of range for {}", T::PY_TYPE))
            }
            (None, Some(f)) => WireValue::Float(f),
            (None, None) => return Err(format!("{n} is not a representable number")),
        },
        Value::Bool(b) => WireValue::Bool(*b),
        Value::String(s) => WireValue::Str(s),
        other => return Err(format!("expected {}, got {other}", T::PY_TYPE)),
    };
    T::from_wire(wire)
}

fn serialize_scalar<T: WireScalar, S: Serializer>(v: T, s: S) -> Result<S::Ok, S::Error> {
    match v.to_wire() {
        WireValue::Int(i) => s.serialize_i64(i),
        WireValue::Float(f) => s.serialize_f64(f),
        WireValue::Bool(b) => s.serialize_bool(b),
        WireValue::Str(n) => s.serialize_str(n),
    }
}

/// `Some(n)` when `value` is exactly `{"$slot": n}`.
fn as_slot(value: &serde_json::Value) -> Option<Result<usize, String>> {
    let obj = value.as_object()?;
    let idx = obj.get(SLOT_KEY)?;
    if obj.len() != 1 {
        return Some(Err(format!(
            "a slot is exactly {{\"{SLOT_KEY}\": n}}, got {value}"
        )));
    }
    Some(
        idx.as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .ok_or_else(|| format!("slot index must be a non-negative integer, got {idx}")),
    )
}

impl<T: WireScalar> Serialize for Param<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match *self {
            Param::Lit(v) => serialize_scalar(v, s),
            Param::Slot(idx) => {
                use serde::ser::SerializeMap;
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(SLOT_KEY, &idx)?;
                map.end()
            }
        }
    }
}

impl<'de, T: WireScalar> Deserialize<'de> for Param<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        if let Some(slot) = as_slot(&value) {
            return slot.map(Param::Slot).map_err(D::Error::custom);
        }
        scalar_from_json(&value)
            .map(Param::Lit)
            .map_err(D::Error::custom)
    }
}

impl<T: WireScalar> Serialize for Literal<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serialize_scalar(self.0, s)
    }
}

impl<'de, T: WireScalar> Deserialize<'de> for Literal<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        if as_slot(&value).is_some() {
            return Err(D::Error::custom(
                "this parameter is structural (it fixes the output shape, rank or \
                 dtype at planning time) and cannot be a per-row expression",
            ));
        }
        scalar_from_json(&value)
            .map(Literal)
            .map_err(D::Error::custom)
    }
}

/// `#[serde(deserialize_with)]` for a plain field read as a [`Literal`]: the
/// graph's policies, which have no op to hang a `Literal<T>` on but must parse
/// through the same `NAMED` table rather than a serde `rename_all` beside it.
pub fn literal_field<'de, D: Deserializer<'de>, T: WireScalar>(d: D) -> Result<T, D::Error> {
    Literal::<T>::deserialize(d).map(|l| l.0)
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

/// How the catalogue describes a field's type to the Python generator.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TypeDesc {
    /// One value. `per_row` is false for a [`Literal`].
    Scalar {
        per_row: bool,
        /// `int`, `float`, `bool`, or a named enum's name.
        py: &'static str,
        /// Every accepted spelling, for a named enum.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        variants: Vec<&'static str>,
    },
    /// May be absent (`None` in Python, omitted on the wire).
    Optional { inner: Box<TypeDesc> },
    /// Exactly `len` elements.
    Array { len: usize, inner: Box<TypeDesc> },
    /// Any number of elements.
    List { inner: Box<TypeDesc> },
    /// One of several shapes, told apart by the value (see the owning type).
    OneOf { options: Vec<TypeDesc> },
    /// Another graph node ([`NodeRef`]); Python passes the operand expression.
    Node,
    /// An input column read as data ([`ColumnRef`]); Python passes an
    /// expression, never a value.
    Column,
    /// A string-to-string map (`source(cloud_options=)`).
    Map,
}

/// A type an op field may have.
///
/// Implemented for the parameter building blocks and their compositions, so
/// the catalogue and the slot visitor follow the field's actual type.
pub trait FieldType {
    /// The field's catalogue description.
    fn describe() -> TypeDesc;
    /// Call `f` with every slot the value reads.
    fn visit_slots(&self, f: &mut dyn FnMut(usize));
}

fn scalar_desc<T: WireScalar>(per_row: bool) -> TypeDesc {
    TypeDesc::Scalar {
        per_row,
        py: T::PY_TYPE,
        variants: T::spellings(),
    }
}

impl<T: WireScalar> FieldType for Param<T> {
    fn describe() -> TypeDesc {
        scalar_desc::<T>(true)
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        if let Param::Slot(idx) = self {
            f(*idx);
        }
    }
}

impl<T: WireScalar> FieldType for Literal<T> {
    fn describe() -> TypeDesc {
        scalar_desc::<T>(false)
    }
    fn visit_slots(&self, _f: &mut dyn FnMut(usize)) {}
}

/// Literal text (a path root, say): never per-row.
impl FieldType for String {
    fn describe() -> TypeDesc {
        TypeDesc::Scalar {
            per_row: false,
            py: "str",
            variants: Vec::new(),
        }
    }
    fn visit_slots(&self, _f: &mut dyn FnMut(usize)) {}
}

impl FieldType for std::collections::HashMap<String, String> {
    fn describe() -> TypeDesc {
        TypeDesc::Map
    }
    fn visit_slots(&self, _f: &mut dyn FnMut(usize)) {}
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

impl<F: FieldType> FieldType for Option<F> {
    fn describe() -> TypeDesc {
        TypeDesc::Optional {
            inner: Box::new(F::describe()),
        }
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        if let Some(v) = self {
            v.visit_slots(f);
        }
    }
}

impl<F: FieldType, const N: usize> FieldType for [F; N] {
    fn describe() -> TypeDesc {
        TypeDesc::Array {
            len: N,
            inner: Box::new(F::describe()),
        }
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        self.iter().for_each(|v| v.visit_slots(f));
    }
}

impl<F: FieldType> FieldType for Vec<F> {
    fn describe() -> TypeDesc {
        TypeDesc::List {
            inner: Box::new(F::describe()),
        }
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        self.iter().for_each(|v| v.visit_slots(f));
    }
}
