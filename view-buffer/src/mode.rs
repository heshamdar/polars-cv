//! Op modes: one op definition, read as the wire's typed op or executed.
//!
//! An engine op is generic over a [`Mode`]: a parameter that may be per-row is
//! a `M::V<T>` field, a structural one a `M::L<T>` field.
//!
//! - [`Exec`] (the default type parameter) is the op the kernels run:
//!   `V<T> = L<T> = T`, so `ImageOpKind::Resize { width, .. }` reads a plain
//!   `u32` exactly as before.
//! - [`Wire`] is the typed op the builder sends and the planner reads:
//!   `V<T> = Param<T>` (a literal or a per-row slot), `L<T> = Literal<T>`.
//!   Every plan-time rule is implemented once, generically, and evaluated on
//!   the `Wire` op directly: a literal is [`Sym::Known`], a slot
//!   [`Sym::PerRow`], so no value is ever invented to read a rule.
//!
//! [`Resolve`] maps a `Wire` op to its `Exec` op for one row, reading each
//! slot through [`Values`]; `#[derive(Resolve)]` writes it field by field.
//!
//! The wire form of a parameter is the value itself — `224`, `"bilinear"`,
//! `[1.0, 0.0]` — or, per row, `{"$slot": n}`: the index of the plugin input
//! column that carries it, or `{"$this": tree}`: a number computed from the
//! buffer the op reads ([`ThisExpr`]). Deserialization is hand-written rather than
//! `#[serde(untagged)]`, which is ambiguous for integer `T` and replaces every
//! error with "data did not match any variant". The eligibility rule (a
//! parameter may be per-row iff it has no effect on the output rank, dtype or
//! domain; a size it sets is then unknown at plan time) is a type: a `$slot`
//! in a [`Literal`] is a deserialization error.

use core::fmt::Debug;
use std::sync::Arc;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::naming::{WireKind, WireScalar, WireValue};
use crate::ops::Sym;

/// How an op holds its parameters (see the module docs).
pub trait Mode: Copy + Clone + Debug + PartialEq + Eq + 'static {
    /// A parameter that may be per-row.
    type V<T: WireScalar>: Clone + Debug + PartialEq;
    /// A structural parameter: always a literal.
    type L<T: WireScalar>: Clone + Debug + PartialEq;
    /// A per-row-capable parameter as a shape argument.
    fn sym<T: WireScalar>(value: &Self::V<T>) -> Sym<T>;
    /// A structural parameter's value.
    fn lit<T: WireScalar>(value: &Self::L<T>) -> T;
}

/// The executed op: every parameter is its value (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exec;

/// The wire's typed op: parameters are literals or per-row slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wire;

impl Mode for Exec {
    type V<T: WireScalar> = T;
    type L<T: WireScalar> = T;
    fn sym<T: WireScalar>(value: &T) -> Sym<T> {
        Sym::Known(*value)
    }
    fn lit<T: WireScalar>(value: &T) -> T {
        *value
    }
}

impl Mode for Wire {
    type V<T: WireScalar> = Param<T>;
    type L<T: WireScalar> = Literal<T>;
    fn sym<T: WireScalar>(value: &Param<T>) -> Sym<T> {
        value.sym()
    }
    fn lit<T: WireScalar>(value: &Literal<T>) -> T {
        value.0
    }
}

/// A size parameter as a shape argument.
pub fn size<M: Mode>(value: &M::V<u32>) -> Sym<usize> {
    match M::sym(value) {
        Sym::Known(n) => Sym::Known(n as usize),
        Sym::PerRow => Sym::PerRow,
    }
}

/// A per-row-capable parameter's value, when it is known (always, for an
/// `Exec` op; for a `Wire` op, when it is a literal). A rule that reads a
/// value reads it through this and says nothing about one it does not know.
pub fn known<M: Mode, T: WireScalar>(value: &M::V<T>) -> Option<T> {
    M::sym(value).known()
}

/// The wire key marking a per-row parameter.
pub const SLOT_KEY: &str = "$slot";

/// The wire key marking a parameter derived from the buffer the op reads.
pub const THIS_KEY: &str = "$this";

/// A parameter that is a literal or, per row, an input column or a number
/// derived from the buffer the op reads.
#[derive(Debug, Clone, PartialEq)]
pub enum Param<T> {
    /// A value fixed for every row.
    Lit(T),
    /// The plugin input column at this position holds the value for each row.
    Slot(usize),
    /// Computed, per row, from attributes of the buffer the op reads
    /// (`polars_cv.this.height`, ...). Only a numeric parameter takes one.
    This(Arc<ThisExpr>),
}

impl<T: Copy> Param<T> {
    /// This parameter as a shape argument: its value when literal, `PerRow`
    /// when a column or the row's shape supplies it.
    pub fn sym(&self) -> Sym<T> {
        match self {
            Param::Lit(v) => Sym::Known(*v),
            Param::Slot(_) | Param::This(_) => Sym::PerRow,
        }
    }
}

impl Param<u32> {
    /// A size parameter as a shape argument (see [`sym`](Self::sym)).
    pub fn size(&self) -> Sym<usize> {
        match self.sym() {
            Sym::Known(v) => Sym::Known(v as usize),
            Sym::PerRow => Sym::PerRow,
        }
    }
}

/// What a parameter reads per row, besides its literal: an input column, or
/// the shape of the buffer its op reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowInput {
    /// The plugin input column at this position.
    Slot(usize),
    /// The shape of the op's input ([`Param::This`]).
    Shape,
}

/// A number computed from attributes of the buffer an op reads
/// (`polars_cv.this`). On the wire, an attribute's name (`"height"`), a
/// number, or `[op, a, b]` with `op` one of `+ - * / //` (`//` floors),
/// under [`THIS_KEY`].
#[derive(Debug, Clone, PartialEq)]
pub enum ThisExpr {
    /// An attribute of the input.
    Attr(ThisAttr),
    /// A finite number.
    Num(f64),
    /// `a op b`.
    Op(ThisOp, Box<(ThisExpr, ThisExpr)>),
}

/// An attribute of the buffer an op reads, known before the op runs (its
/// shape): every one is read from the shape the executor carries through a
/// node's steps, so none needs the buffer's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThisAttr {
    /// Axis 0 of an `[H, W]` or `[H, W, C]` buffer.
    Height,
    /// Axis 1 of an `[H, W]` or `[H, W, C]` buffer.
    Width,
    /// Axis 2 of an `[H, W, C]` buffer; 1 for an `[H, W]` one.
    Channels,
    /// The rank, of any buffer.
    Ndim,
}

impl ThisAttr {
    /// Each attribute and its wire (and Python) name.
    const NAMES: [(ThisAttr, &'static str); 4] = [
        (ThisAttr::Height, "height"),
        (ThisAttr::Width, "width"),
        (ThisAttr::Channels, "channels"),
        (ThisAttr::Ndim, "ndim"),
    ];

    fn name(self) -> &'static str {
        Self::NAMES
            .iter()
            .find(|(attr, _)| *attr == self)
            .map(|(_, n)| *n)
            .expect("every attribute has a name")
    }

    /// The attribute of a buffer of `shape`.
    fn of(self, shape: &[usize]) -> Result<usize, String> {
        match (self, shape) {
            (ThisAttr::Ndim, _) => Ok(shape.len()),
            (ThisAttr::Height, [h, _] | [h, _, _]) => Ok(*h),
            (ThisAttr::Width, [_, w] | [_, w, _]) => Ok(*w),
            (ThisAttr::Channels, [_, _]) => Ok(1),
            (ThisAttr::Channels, [_, _, c]) => Ok(*c),
            (attr, _) => Err(format!(
                "this.{} reads an [H, W] or [H, W, C] buffer, and this op's input \
                 has shape {shape:?}",
                attr.name()
            )),
        }
    }
}

/// A [`ThisExpr`] operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThisOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
}

impl ThisOp {
    /// Each operator and its wire (and Python) spelling.
    const SPELLINGS: [(ThisOp, &'static str); 5] = [
        (ThisOp::Add, "+"),
        (ThisOp::Sub, "-"),
        (ThisOp::Mul, "*"),
        (ThisOp::Div, "/"),
        (ThisOp::FloorDiv, "//"),
    ];

    fn spelling(self) -> &'static str {
        Self::SPELLINGS
            .iter()
            .find(|(op, _)| *op == self)
            .map(|(_, s)| *s)
            .expect("every operator has a spelling")
    }

    /// How tightly it binds, as in Python (for display).
    fn precedence(self) -> u8 {
        match self {
            ThisOp::Add | ThisOp::Sub => 1,
            ThisOp::Mul | ThisOp::Div | ThisOp::FloorDiv => 2,
        }
    }
}

impl ThisExpr {
    /// Parse the wire tree.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        use serde_json::Value;
        match value {
            Value::String(s) => ThisAttr::NAMES
                .iter()
                .find(|(_, n)| n == s)
                .map(|(attr, _)| ThisExpr::Attr(*attr))
                .ok_or_else(|| {
                    let known: Vec<&str> = ThisAttr::NAMES.iter().map(|(_, n)| *n).collect();
                    format!("unknown attribute of `this` {s:?}, expected one of {known:?}")
                }),
            Value::Number(n) => match n.as_f64() {
                Some(f) if f.is_finite() => Ok(ThisExpr::Num(f)),
                _ => Err(format!("{n} is not a finite number")),
            },
            Value::Array(items) if items.len() == 3 => {
                let op = items[0]
                    .as_str()
                    .and_then(|s| ThisOp::SPELLINGS.iter().find(|(_, sp)| *sp == s))
                    .map(|(op, _)| *op)
                    .ok_or_else(|| {
                        let known: Vec<&str> = ThisOp::SPELLINGS.iter().map(|(_, s)| *s).collect();
                        format!("unknown operator {}, expected one of {known:?}", items[0])
                    })?;
                let a = Self::from_json(&items[1])?;
                let b = Self::from_json(&items[2])?;
                Ok(ThisExpr::Op(op, Box::new((a, b))))
            }
            other => Err(format!(
                "a `this` expression is an attribute name, a number or [op, a, b], got {other}"
            )),
        }
    }

    /// The wire tree.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ThisExpr::Attr(attr) => attr.name().into(),
            ThisExpr::Num(f) => serde_json::Value::from(*f),
            ThisExpr::Op(op, args) => {
                serde_json::json!([op.spelling(), args.0.to_json(), args.1.to_json()])
            }
        }
    }

    /// The value over a buffer of `shape`.
    pub fn eval(&self, shape: &[usize]) -> Result<f64, String> {
        Ok(match self {
            ThisExpr::Attr(attr) => attr.of(shape)? as f64,
            ThisExpr::Num(f) => *f,
            ThisExpr::Op(op, args) => {
                let (a, b) = (args.0.eval(shape)?, args.1.eval(shape)?);
                match op {
                    ThisOp::Add => a + b,
                    ThisOp::Sub => a - b,
                    ThisOp::Mul => a * b,
                    ThisOp::Div | ThisOp::FloorDiv if b == 0.0 => {
                        return Err(format!("{self}: division by zero"))
                    }
                    ThisOp::Div => a / b,
                    ThisOp::FloorDiv => (a / b).floor(),
                }
            }
        })
    }

    /// Written as an operand of `parent`, parenthesised where Python would
    /// need it to read the same: a looser operator, or on the right an equal
    /// one that does not associate (`a - (b - c)`, `a / (b * c)`).
    fn fmt_operand(
        &self,
        f: &mut core::fmt::Formatter<'_>,
        parent: ThisOp,
        right: bool,
    ) -> core::fmt::Result {
        let wrap = match self {
            ThisExpr::Op(op, _) => {
                let (child, outer) = (op.precedence(), parent.precedence());
                child < outer
                    || (right
                        && child == outer
                        && !(*op == parent && matches!(parent, ThisOp::Add | ThisOp::Mul)))
            }
            _ => false,
        };
        if wrap {
            write!(f, "({self})")
        } else {
            write!(f, "{self}")
        }
    }
}

impl core::fmt::Display for ThisExpr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ThisExpr::Attr(attr) => write!(f, "this.{}", attr.name()),
            ThisExpr::Num(n) => write!(f, "{n}"),
            ThisExpr::Op(op, args) => {
                args.0.fmt_operand(f, *op, false)?;
                write!(f, " {} ", op.spelling())?;
                args.1.fmt_operand(f, *op, true)
            }
        }
    }
}

/// `expr`'s value as a `T`, over `shape`, the shape of the buffer the op
/// reads (`None` where none is known before the op runs). An integer
/// parameter takes the value rounded to nearest (half away from zero), so
/// `0.29 * this.height` at a height of 100 is 29, not the 28 its float error
/// truncates to.
pub fn this_value<T: WireScalar>(expr: &ThisExpr, shape: Option<&[usize]>) -> Result<T, String> {
    let Some(shape) = shape else {
        return Err(format!(
            "{expr}: `this` needs the shape of the buffer this op reads, which is \
             not known before the op runs (its input is not a buffer, or an \
             earlier op's output size depends on the data)"
        ));
    };
    let value = expr.eval(shape)?;
    if !value.is_finite() {
        return Err(format!("{expr} is {value} for an input of shape {shape:?}"));
    }
    let wire = match T::KIND {
        WireKind::Float => WireValue::Float(value),
        WireKind::Int => {
            let rounded = value.round();
            if rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
                return Err(format!("{expr} is {value}, out of range"));
            }
            WireValue::Int(rounded as i64)
        }
        WireKind::Bool | WireKind::Name => {
            return Err(this_kind_error::<T>());
        }
    };
    T::from_wire(wire).map_err(|e| format!("{expr} = {value} for an input of shape {shape:?}: {e}"))
}

/// The refusal of a `this` expression where the parameter is not a number.
fn this_kind_error<T: WireScalar>() -> String {
    format!(
        "`this` attributes give a number, and this parameter takes a {}",
        T::PY_TYPE
    )
}

/// A structural parameter: always a literal, never per-row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Literal<T>(pub T);

impl<T: WireScalar> Literal<T> {
    /// The value.
    pub fn get(&self) -> T {
        self.0
    }
}

/// An input column the step reads as *data* — `label_reduce`'s contour set —
/// rather than a parameter value resolved per row. Always a slot: a literal
/// has nowhere to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnRef(pub usize);

/// Another graph node, by id: the operand of a binary op, a mask, a merged
/// channel. Graph topology, so never per-row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRef(pub String);

/// Where a `Wire` op's per-row values come from: one row of the plugin's
/// parameter columns.
pub trait Values {
    /// The error a missing, null or unparseable value raises.
    type Error;
    /// The value slot `slot` holds for this row.
    fn value<T: WireScalar>(&self, slot: usize) -> Result<T, Self::Error>;
    /// The value `expr` takes over the buffer the op reads ([`this_value`]).
    fn this<T: WireScalar>(&self, expr: &ThisExpr) -> Result<T, Self::Error>;
    /// `error`, raised resolving the field `field`, saying which field it was:
    /// `#[derive(Resolve)]` calls it for every named field, so a bad per-row
    /// value names the parameter it was bound to (`left`), not only the
    /// column it came from. Required, so an implementation decides.
    fn in_field(&self, field: &'static str, error: Self::Error) -> Self::Error;
}

/// What `#[derive(Ops)]` gives a family's `Wire` form, for code generic
/// over families (the family's inherent methods of the same names).
pub trait WireOps: Sized {
    /// The op `name` from its wire fields; `None` when the family has none.
    fn from_wire(name: &str, fields: serde_json::Value) -> Option<Result<Self, String>>;
    /// The op's wire name; `None` for an engine-internal variant.
    fn wire_name(&self) -> Option<&'static str>;
    /// The op's wire fields; `None` for an engine-internal variant.
    fn wire_fields(&self) -> Option<serde_json::Map<String, serde_json::Value>>;
    /// Call `f(field, input)` for every row input a field reads.
    fn visit_inputs(&self, f: &mut dyn FnMut(&'static str, RowInput));
    /// Every op's catalogue entry, in declaration order.
    fn catalog() -> Vec<OpDesc>;
}

/// No per-row values: resolves an all-literal op, and refuses a slot.
pub struct Literals;

impl Values for Literals {
    type Error = String;
    fn value<T: WireScalar>(&self, slot: usize) -> Result<T, String> {
        Err(format!("slot {slot} read where only literals resolve"))
    }
    fn this<T: WireScalar>(&self, expr: &ThisExpr) -> Result<T, String> {
        this_value(expr, None)
    }
    fn in_field(&self, field: &'static str, error: String) -> String {
        format!("{field}: {error}")
    }
}

/// A `Wire` value's executed form for one row.
pub trait Resolve {
    /// The `Exec` form.
    type Exec;
    /// Read every slot through `values`.
    fn resolve<V: Values>(&self, values: &V) -> Result<Self::Exec, V::Error>;
}

impl<T: WireScalar> Resolve for Param<T> {
    type Exec = T;
    fn resolve<V: Values>(&self, values: &V) -> Result<T, V::Error> {
        match self {
            Param::Lit(v) => Ok(*v),
            Param::Slot(slot) => values.value(*slot),
            Param::This(expr) => values.this(expr),
        }
    }
}

impl Resolve for NodeRef {
    type Exec = NodeRef;
    fn resolve<V: Values>(&self, _values: &V) -> Result<NodeRef, V::Error> {
        Ok(self.clone())
    }
}

impl Resolve for ColumnRef {
    type Exec = ColumnRef;
    fn resolve<V: Values>(&self, _values: &V) -> Result<ColumnRef, V::Error> {
        Ok(*self)
    }
}

impl<T: WireScalar> Resolve for Literal<T> {
    type Exec = T;
    fn resolve<V: Values>(&self, _values: &V) -> Result<T, V::Error> {
        Ok(self.0)
    }
}

impl<R: Resolve> Resolve for Option<R> {
    type Exec = Option<R::Exec>;
    fn resolve<V: Values>(&self, values: &V) -> Result<Self::Exec, V::Error> {
        self.as_ref().map(|r| r.resolve(values)).transpose()
    }
}

impl<R: Resolve> Resolve for Vec<R> {
    type Exec = Vec<R::Exec>;
    fn resolve<V: Values>(&self, values: &V) -> Result<Self::Exec, V::Error> {
        self.iter().map(|r| r.resolve(values)).collect()
    }
}

impl<R: Resolve, const N: usize> Resolve for [R; N] {
    type Exec = [R::Exec; N];
    fn resolve<V: Values>(&self, values: &V) -> Result<Self::Exec, V::Error> {
        let mut out = Vec::with_capacity(N);
        for r in self {
            out.push(r.resolve(values)?);
        }
        match out.try_into() {
            Ok(array) => Ok(array),
            Err(_) => unreachable!("one element per element"),
        }
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
        Value::String(s) => WireValue::str(s),
        other => return Err(format!("expected {}, got {other}", T::PY_TYPE)),
    };
    T::from_wire(wire)
}

fn serialize_scalar<T: WireScalar, S: Serializer>(v: T, s: S) -> Result<S::Ok, S::Error> {
    match v.to_wire() {
        WireValue::Int(i) => s.serialize_i64(i),
        WireValue::Float(f) => s.serialize_f64(f),
        WireValue::Bool(b) => s.serialize_bool(b),
        WireValue::Str(n) => s.serialize_str(&n),
    }
}

/// `Some(n)` when `value` is exactly `{"$slot": n}`.
pub fn as_slot(value: &serde_json::Value) -> Option<Result<usize, String>> {
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

/// `Some(expr)` when `value` is exactly `{"$this": tree}`.
pub fn as_this(value: &serde_json::Value) -> Option<Result<ThisExpr, String>> {
    let obj = value.as_object()?;
    let tree = obj.get(THIS_KEY)?;
    if obj.len() != 1 {
        return Some(Err(format!(
            "a `this` expression is exactly {{\"{THIS_KEY}\": tree}}, got {value}"
        )));
    }
    Some(ThisExpr::from_json(tree))
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
            Param::This(ref expr) => {
                use serde::ser::SerializeMap;
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(THIS_KEY, &expr.to_json())?;
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
        if let Some(expr) = as_this(&value) {
            if !matches!(T::KIND, WireKind::Int | WireKind::Float) {
                return Err(D::Error::custom(this_kind_error::<T>()));
            }
            return expr
                .map(|e| Param::This(Arc::new(e)))
                .map_err(D::Error::custom);
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
        if as_slot(&value).is_some() || as_this(&value).is_some() {
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

/// `#[serde(deserialize_with)]` for a plain field read as a [`Literal`]: a
/// setting with no op to hang a `Literal<T>` on that must still parse through
/// its `NAMED` table rather than a serde `rename_all` beside it.
pub fn literal_field<'de, D: Deserializer<'de>, T: WireScalar>(d: D) -> Result<T, D::Error> {
    Literal::<T>::deserialize(d).map(|l| l.0)
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
    /// Another graph node; Python passes the operand expression.
    Node,
    /// An input column read as data; Python passes an expression, never a
    /// value.
    Column,
    /// A string-to-string map (`source(cloud_options=)`).
    Map,
}

/// A type an op field may have (its `Wire` form).
///
/// Implemented for the parameter building blocks and their compositions, so
/// the catalogue and the slot visitor follow the field's actual type.
pub trait FieldType {
    /// The field's catalogue description.
    fn describe() -> TypeDesc;
    /// Call `f` with every row input the value reads.
    fn visit_inputs(&self, f: &mut dyn FnMut(RowInput));
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
    fn visit_inputs(&self, f: &mut dyn FnMut(RowInput)) {
        match self {
            Param::Lit(_) => {}
            Param::Slot(idx) => f(RowInput::Slot(*idx)),
            Param::This(_) => f(RowInput::Shape),
        }
    }
}

impl<T: WireScalar> FieldType for Literal<T> {
    fn describe() -> TypeDesc {
        scalar_desc::<T>(false)
    }
    fn visit_inputs(&self, _f: &mut dyn FnMut(RowInput)) {}
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
    fn visit_inputs(&self, _f: &mut dyn FnMut(RowInput)) {}
}

impl FieldType for std::collections::HashMap<String, String> {
    fn describe() -> TypeDesc {
        TypeDesc::Map
    }
    fn visit_inputs(&self, _f: &mut dyn FnMut(RowInput)) {}
}

impl<F: FieldType> FieldType for Option<F> {
    fn describe() -> TypeDesc {
        TypeDesc::Optional {
            inner: Box::new(F::describe()),
        }
    }
    fn visit_inputs(&self, f: &mut dyn FnMut(RowInput)) {
        if let Some(v) = self {
            v.visit_inputs(f);
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
    fn visit_inputs(&self, f: &mut dyn FnMut(RowInput)) {
        self.iter().for_each(|v| v.visit_inputs(f));
    }
}

impl<F: FieldType> FieldType for Vec<F> {
    fn describe() -> TypeDesc {
        TypeDesc::List {
            inner: Box::new(F::describe()),
        }
    }
    fn visit_inputs(&self, f: &mut dyn FnMut(RowInput)) {
        self.iter().for_each(|v| v.visit_inputs(f));
    }
}

/// One field of an op, as the catalogue describes it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldDesc {
    pub name: &'static str,
    pub doc: &'static str,
    /// The Python signature default; absent means required (or `None` for an
    /// optional field).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(rename = "type")]
    pub ty: TypeDesc,
}

/// One op in the catalogue.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpDesc {
    /// The wire name.
    pub name: &'static str,
    /// The Python method name.
    pub python: &'static str,
    pub visibility: Visibility,
    pub doc: &'static str,
    pub fields: Vec<FieldDesc>,
}

/// Where the Python builder exposes an op: `#[op(visibility = ...)]`, naming
/// a variant (default `Public`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// A generated `Pipeline` method (and its lazy forwarder).
    Public,
    /// A generated `LazyPipelineExpr` method only: it reads another node.
    LazyOnly,
    /// No generated method: reached through hand-written sugar, or planned
    /// internally.
    Internal,
}

/// Read field `name` out of an op's wire object, naming the field (and the
/// element within it) in any error: `'size[1]': 7.5 is not an integer`.
/// `None` when the field is absent (or `null`); the caller decides whether
/// that is allowed.
pub fn take_field<T: serde::de::DeserializeOwned>(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<Option<T>, String> {
    match fields.remove(name) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => serde_path_to_error::deserialize(value)
            .map(Some)
            .map_err(|e| {
                let path = e.path().to_string();
                if path == "." {
                    format!("'{name}': {}", e.inner())
                } else {
                    format!("'{name}{}': {}", path.trim_start_matches('.'), e.inner())
                }
            }),
    }
}

/// An absent field's declared default (`#[param(default = ...)]`), read as
/// the field. `every_field_default_is_a_valid_value` holds every declared
/// default to parse, so the error is a definition mistake, named as such.
pub fn default_field<T: serde::de::DeserializeOwned>(
    name: &str,
    default: serde_json::Value,
) -> Result<T, String> {
    serde_json::from_value(default)
        .map_err(|e| format!("'{name}': the declared default is not a valid value: {e}"))
}

/// The error for a required field that is absent.
pub fn missing_field(name: &str) -> String {
    format!("missing field `{name}`")
}

/// The error for a key no field of the op has, when any remains.
pub fn refuse_unknown_fields(
    fields: &serde_json::Map<String, serde_json::Value>,
    expected: &[&str],
) -> Result<(), String> {
    match fields.keys().next() {
        None => Ok(()),
        Some(key) => {
            let expected: Vec<String> = expected.iter().map(|f| format!("`{f}`")).collect();
            Err(if expected.is_empty() {
                format!("unknown field `{key}`, there are no fields")
            } else {
                format!(
                    "unknown field `{key}`, expected one of {}",
                    expected.join(", ")
                )
            })
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
    fn visit_inputs(&self, f: &mut dyn FnMut(RowInput)) {
        f(RowInput::Slot(self.0));
    }
}

impl FieldType for NodeRef {
    fn describe() -> TypeDesc {
        TypeDesc::Node
    }
    fn visit_inputs(&self, _f: &mut dyn FnMut(RowInput)) {}
}

#[cfg(test)]
mod this_tests {
    use super::*;
    use serde_json::json;

    fn tree(v: serde_json::Value) -> ThisExpr {
        ThisExpr::from_json(&v).unwrap()
    }

    #[test]
    fn the_wire_tree_round_trips_and_evaluates() {
        let wire = json!(["+", ["*", ["-", "height", 4.0], 0.5], ["//", "width", 3.0]]);
        let expr = tree(wire.clone());
        assert_eq!(expr.to_json(), wire);
        // (100 - 4) * 0.5 + floor(10 / 3)
        assert_eq!(expr.eval(&[100, 10, 3]).unwrap(), 51.0);
    }

    #[test]
    fn each_attribute_reads_its_part_of_the_shape() {
        for (name, shape, want) in [
            ("height", &[4, 5, 3][..], 4.0),
            ("width", &[4, 5][..], 5.0),
            ("channels", &[4, 5, 3][..], 3.0),
            ("channels", &[4, 5][..], 1.0),
            ("ndim", &[7][..], 1.0),
            ("ndim", &[1, 2, 3, 4][..], 4.0),
        ] {
            assert_eq!(
                tree(json!(name)).eval(shape).unwrap(),
                want,
                "{name} {shape:?}"
            );
        }
    }

    /// The same text Python's `repr` gives, so an error quotes the
    /// expression as the user wrote it.
    #[test]
    fn it_displays_as_python_reads_it() {
        for (wire, text) in [
            (
                json!(["*", ["-", "height", 4], 0.5]),
                "(this.height - 4) * 0.5",
            ),
            (
                json!(["/", "height", ["+", "width", 1]]),
                "this.height / (this.width + 1)",
            ),
            (
                json!(["-", "height", ["-", "width", 1]]),
                "this.height - (this.width - 1)",
            ),
            (
                json!(["+", "height", ["+", "width", 1]]),
                "this.height + this.width + 1",
            ),
            (
                json!(["-", ["-", "height", "width"], 1]),
                "this.height - this.width - 1",
            ),
        ] {
            assert_eq!(tree(wire).to_string(), text);
        }
    }

    #[test]
    fn a_malformed_tree_is_refused() {
        for (wire, says) in [
            (json!(["%", "height", 2]), "unknown operator"),
            (json!(["+", "height"]), "a `this` expression is"),
            (json!("depth"), "unknown attribute of `this` \"depth\""),
        ] {
            let err = ThisExpr::from_json(&wire).unwrap_err();
            assert!(err.contains(says), "{err}");
        }
    }

    /// An integer parameter rounds to nearest: 0.29 * 100 is
    /// 28.999999999999996, which truncating would make 28.
    #[test]
    fn an_integer_rounds_to_nearest_and_a_float_is_exact() {
        let expr = tree(json!(["*", 0.29, "height"]));
        assert_eq!(this_value::<u32>(&expr, Some(&[100, 7])).unwrap(), 29);
        let half = tree(json!(["/", "width", 2]));
        assert_eq!(this_value::<u32>(&half, Some(&[1, 5, 3])).unwrap(), 3);
        assert_eq!(this_value::<f32>(&half, Some(&[1, 5])).unwrap(), 2.5);
    }

    #[test]
    fn the_shape_must_be_known_and_fit_the_attribute() {
        let h = tree(json!("height"));
        let err = this_value::<u32>(&h, None).unwrap_err();
        assert!(err.contains("not known before the op runs"), "{err}");
        for shape in [&[9][..], &[1, 2, 3, 4]] {
            let err = this_value::<u32>(&h, Some(shape)).unwrap_err();
            assert!(
                err.contains("this.height reads an [H, W] or [H, W, C]"),
                "{err}"
            );
        }
        let err = this_value::<u32>(
            &tree(json!(["/", "height", ["-", "width", "width"]])),
            Some(&[2, 2]),
        )
        .unwrap_err();
        assert!(
            err.contains("this.height / (this.width - this.width): division by zero"),
            "{err}"
        );
        let err = this_value::<u32>(&tree(json!(["-", "height", 9])), Some(&[2, 2])).unwrap_err();
        assert!(err.contains("negative"), "{err}");
    }

    #[test]
    fn only_a_per_row_numeric_parameter_takes_one() {
        let wire = json!({"$this": ["/", "height", 2]});
        let p: Param<u32> = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(p.sym(), Sym::PerRow);
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            json!({"$this": ["/", "height", 2.0]})
        );
        let err = serde_json::from_value::<Param<bool>>(wire.clone()).unwrap_err();
        assert!(
            err.to_string().contains("`this` attributes give a number"),
            "{err}"
        );
        let err = serde_json::from_value::<Literal<u32>>(wire).unwrap_err();
        assert!(err.to_string().contains("structural"), "{err}");
    }

    #[test]
    fn it_is_visited_as_a_shape_read() {
        let p: Param<f32> = serde_json::from_value(json!({"$this": "width"})).unwrap();
        let mut seen = Vec::new();
        p.visit_inputs(&mut |input| seen.push(input));
        assert_eq!(seen, [RowInput::Shape]);
    }
}
