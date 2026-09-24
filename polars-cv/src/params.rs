//! Parameter value types for expression resolution.
//!
//! This module handles the resolution of parameter values that can be either
//! literals (known at planning time) or expressions (resolved per-row).
//!
//! An expression parameter arrives as a [`ParamValue::Slot`]: the absolute
//! index of its column among the plugin's input series, assigned by the Python
//! `SlotTable`. Per-row resolution is a direct indexed read through a typed
//! accessor ([`ParamCol`]); no names are involved.

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::HashMap;

/// What a **null** in a per-row expression parameter column means.
///
/// Per-row parameters are read from ordinary Polars columns, which may contain
/// nulls. This says whether a missing parameter aborts the query or simply
/// yields a missing result for the rows it affects.
///
/// Deliberately separate from [`RowErrorPolicy`](crate::graph::RowErrorPolicy):
/// under [`Null`](Self::Null) a null parameter is *not* an error, so it records
/// no `_error` message under `null_with_message`, and opting into it does not
/// weaken reporting for decode, encode or genuine operation failures.
///
/// There is deliberately no "fallback default" variant: `pl.col("w").fill_null(1.0)`
/// already expresses that in Polars itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NullParamPolicy {
    /// A null parameter fails the expression (the default).
    #[default]
    Raise,
    /// A null parameter makes the node produce no output for that row, which
    /// propagates as a null exactly like a null *input* already does.
    Null,
}

// As with `RowErrorPolicy`, these must agree with the `rename_all` above;
// `null_param_policy_names_match_serde` checks that they do.
view_buffer::naming::named_variants!(NullParamPolicy {
    "raise" => Raise,
    "null" => Null,
});

/// A parameter value: a literal, or a per-row input column by position.
///
/// Wire form (hand-written (de)serialization below — see its docs):
/// `{"type": "literal", "value": …}` or `{"$slot": n}`.
#[derive(Debug, Clone)]
pub enum ParamValue {
    /// A literal value known at planning time.
    Literal {
        /// The literal value.
        value: serde_json::Value,
    },

    /// A per-row value: the plugin input column at this absolute index
    /// (root columns first, expression parameters after — the Python
    /// `SlotTable` assigns the positions).
    Slot {
        /// Absolute index into the plugin input series.
        idx: usize,
    },

    /// A pre-parsed nested parameter list.
    ///
    /// Produced by graph compilation from a `Literal` whose JSON value is an
    /// array of `ParamValue` dicts (a `warp_affine` matrix, a `reshape`
    /// shape), so per-row resolution reads the parsed elements directly (via
    /// [`ParamValue::as_param_slice`]) instead of re-deserializing the JSON
    /// every row. The introspection path (`op_schema`) does not compile the
    /// graph and keeps the `Literal` JSON form, so `as_param_list` handles
    /// both. Serializes back to the `Literal` form.
    List(Vec<ParamValue>),
}

/// The key that marks a slot reference on the wire.
const SLOT_KEY: &str = "$slot";

impl Serialize for ParamValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(2))?;
        match self {
            ParamValue::Literal { value } => {
                map.serialize_entry("type", "literal")?;
                map.serialize_entry("value", value)?;
            }
            ParamValue::Slot { idx } => map.serialize_entry(SLOT_KEY, idx)?,
            ParamValue::List(items) => {
                map.serialize_entry("type", "literal")?;
                map.serialize_entry("value", items)?;
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ParamValue {
    /// Exactly two shapes are accepted, each closed: `{"type": "literal",
    /// "value": v}` and `{"$slot": n}`. Anything else — including the removed
    /// name-keyed `{"type": "expr", "col": …}` form — is an error naming both.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        ParamValue::from_wire(&value).map_err(D::Error::custom)
    }
}

impl ParamValue {
    /// Parse one wire parameter (see the `Deserialize` impl).
    pub fn from_wire(value: &serde_json::Value) -> Result<Self, String> {
        let expected = || {
            format!(
                "expected a parameter {{\"type\": \"literal\", \"value\": …}} or \
                 {{\"{SLOT_KEY}\": n}}, got {value}"
            )
        };
        let obj = value.as_object().ok_or_else(expected)?;
        match (
            obj.len(),
            obj.get(SLOT_KEY),
            obj.get("type"),
            obj.get("value"),
        ) {
            (1, Some(idx), None, None) => idx
                .as_u64()
                .map(|idx| ParamValue::Slot { idx: idx as usize })
                .ok_or_else(|| format!("slot index must be a non-negative integer, got {idx}")),
            (2, None, Some(t), Some(v)) if t == "literal" => {
                Ok(ParamValue::Literal { value: v.clone() })
            }
            _ => Err(expected()),
        }
    }

    /// Whether a JSON value is a wire parameter (either accepted shape).
    pub fn is_wire_param(value: &serde_json::Value) -> bool {
        value.as_object().is_some_and(|obj| {
            obj.contains_key(SLOT_KEY) || obj.get("type").is_some_and(|t| t == "literal")
        })
    }
}

impl ParamValue {
    /// Check if this parameter is fully literal (statically resolvable at compile
    /// time with an empty context).
    ///
    /// Nested parameter lists (a `warp_affine` matrix, a `reshape` shape) are
    /// hoisted into [`ParamValue::List`] by graph compilation
    /// (`graph::compiled::bind_param`) *before* this classification runs, so a
    /// bare `Literal` never holds nested sub-params and is always fully literal;
    /// a `List` is literal iff every element is (a `Slot` element
    /// makes the whole op dynamic, so it re-resolves per row).
    pub fn is_literal(&self) -> bool {
        match self {
            ParamValue::Literal { .. } => true,
            ParamValue::List(items) => items.iter().all(ParamValue::is_literal),
            ParamValue::Slot { .. } => false,
        }
    }

    /// Look up this param's bound column in the context.
    fn slot_col<'c, 'a>(&self, ctx: &'c ParamCtx<'a>) -> PolarsResult<&'c ParamCol<'a>> {
        match self {
            ParamValue::Slot { idx } => ctx.col(*idx),
            ParamValue::List(_) => Err(polars_err!(ComputeError:
                "Internal error: a nested list parameter cannot be resolved as a scalar"
            )),
            ParamValue::Literal { .. } => {
                unreachable!("slot_col called on literal")
            }
        }
    }

    /// Resolve this parameter to a concrete i64 value.
    pub fn resolve_i64(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<i64> {
        match self {
            ParamValue::Literal { value } => value.as_i64().ok_or_else(
                || polars_err!(ComputeError: "Expected integer literal, got {:?}", value),
            ),
            _ => self.slot_col(ctx)?.get_i64(row_idx, ctx),
        }
    }

    /// Resolve this parameter to a concrete usize value.
    pub fn resolve_usize(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<usize> {
        let value = self.resolve_i64(row_idx, ctx)?;
        if value < 0 {
            return Err(polars_err!(ComputeError: "Value {} cannot be negative", value));
        }
        Ok(value as usize)
    }

    /// Resolve this parameter to a concrete f64 value.
    pub fn resolve_f64(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<f64> {
        match self {
            ParamValue::Literal { value } => value.as_f64().ok_or_else(
                || polars_err!(ComputeError: "Expected float literal, got {:?}", value),
            ),
            _ => self.slot_col(ctx)?.get_f64(row_idx, ctx),
        }
    }

    /// Resolve this parameter to a concrete f32 value.
    pub fn resolve_f32(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<f32> {
        self.resolve_f64(row_idx, ctx).map(|v| v as f32)
    }

    /// Resolve this parameter to a literal string.
    ///
    /// Deliberately literal-only: this is the accessor for **structural**
    /// string parameters — `cast(dtype)`, `normalize(method/out_dtype)`,
    /// `histogram(output)` — which feed dtype inference and so must be fixed
    /// at planning time. Non-structural enums resolve through
    /// [`resolve_str`](Self::resolve_str) instead.
    pub fn resolve_string(&self) -> PolarsResult<&str> {
        match self {
            ParamValue::Literal { value } => value.as_str().ok_or_else(
                || polars_err!(ComputeError: "Expected string literal, got {:?}", value),
            ),
            ParamValue::Slot { .. } | ParamValue::List(_) => Err(polars_err!(ComputeError:
                    "This string parameter is structural (it fixes the output \
                     dtype at planning time) and cannot be an expression")),
        }
    }

    /// Resolve this parameter to a string, per row when it is bound to a column.
    ///
    /// For enum parameters with no shape or dtype effect, where a per-row value
    /// is meaningful. Returns `None` under a plan-time probe context, telling
    /// the caller to use its default (see [`ParamCtx::probe`]).
    pub fn resolve_str<'a>(
        &'a self,
        row_idx: usize,
        ctx: &ParamCtx<'a>,
    ) -> PolarsResult<Option<&'a str>> {
        match self {
            ParamValue::Literal { value } => value.as_str().map(Some).ok_or_else(
                || polars_err!(ComputeError: "Expected string literal, got {:?}", value),
            ),
            _ if ctx.is_probe() => Ok(None),
            _ => self.slot_col(ctx)?.get_str(row_idx, ctx).map(Some),
        }
    }

    /// Resolve this parameter to a boolean, per row when bound to a column.
    ///
    /// Returns `None` under a plan-time probe context, as [`resolve_str`] does.
    pub fn resolve_bool(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<Option<bool>> {
        match self {
            ParamValue::Literal { value } => value.as_bool().map(Some).ok_or_else(
                || polars_err!(ComputeError: "Expected boolean literal, got {:?}", value),
            ),
            _ if ctx.is_probe() => Ok(None),
            _ => self.slot_col(ctx)?.get_bool(row_idx, ctx).map(Some),
        }
    }

    /// The already-bound elements of a compiled nested list, if this is one.
    ///
    /// The zero-copy fast path for per-row resolution: a `List` (produced once at
    /// compile time) is iterated directly, no per-row JSON parse or allocation.
    /// Returns `None` for the `Literal` JSON form (the introspection path), whose
    /// caller falls back to [`ParamValue::as_param_list`].
    pub fn as_param_slice(&self) -> Option<&[ParamValue]> {
        match self {
            ParamValue::List(items) => Some(items),
            _ => None,
        }
    }

    /// Get value as an owned list of ParamValue (for reshape / warp_affine).
    ///
    /// Handles both the compiled `List` form and the `Literal` JSON-array form
    /// (used by the un-compiled introspection path). Prefer [`as_param_slice`]
    /// on the per-row hot path to avoid the allocation.
    pub fn as_param_list(&self) -> PolarsResult<Vec<ParamValue>> {
        match self {
            ParamValue::List(items) => Ok(items.clone()),
            ParamValue::Literal { value } => {
                let arr = value.as_array().ok_or_else(
                    || polars_err!(ComputeError: "Expected array literal, got {:?}", value),
                )?;

                arr.iter()
                    .map(|v| {
                        // Each element in the array is itself a ParamValue dict
                        serde_json::from_value(v.clone()).map_err(
                            |e| polars_err!(ComputeError: "Invalid param value in array: {}", e),
                        )
                    })
                    .collect()
            }
            ParamValue::Slot { .. } => {
                Err(polars_err!(ComputeError: "Array parameters cannot be expressions"))
            }
        }
    }

    /// Borrow the elements of a per-element parameter list.
    ///
    /// Handles both forms transparently: the compiled `List` is borrowed with
    /// no allocation (the per-row hot path), while the un-compiled `Literal`
    /// JSON-array form used by the introspection path is parsed into
    /// `owned`, which the caller supplies as scratch storage.
    pub fn param_elements<'p>(
        &'p self,
        owned: &'p mut Option<Vec<ParamValue>>,
    ) -> PolarsResult<&'p [ParamValue]> {
        match self.as_param_slice() {
            Some(slice) => Ok(slice),
            None => Ok(owned.insert(self.as_param_list()?)),
        }
    }

    /// Resolve a per-element parameter list to `f32`s at `row_idx`.
    ///
    /// The list *length* stays structural — it fixes a kernel size or channel
    /// count at planning time — while each element may be a per-row
    /// expression, the same encoding `reshape` and `warp_affine` use.
    pub fn resolve_f32_list(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<Vec<f32>> {
        let mut owned = None;
        self.param_elements(&mut owned)?
            .iter()
            .map(|p| p.resolve_f32(row_idx, ctx))
            .collect()
    }

    /// Resolve a per-element parameter list to `usize`s at `row_idx`.
    ///
    /// For value-carrying lists such as `channel_swap`'s permutation, where the
    /// element *count* is structural but the values are not. (Axis lists, which
    /// reorder dimensions, are typed `Literal` lists on their ops.)
    pub fn resolve_usize_list(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<Vec<usize>> {
        let mut owned = None;
        self.param_elements(&mut owned)?
            .iter()
            .map(|p| p.resolve_usize(row_idx, ctx))
            .collect()
    }
}

// ============================================================================
// Per-call parameter context
// ============================================================================

/// A typed, pre-downcast view of one input column used as a dynamic parameter.
///
/// Built once per plugin call so per-row reads are a direct `ChunkedArray::get`
/// on the concrete dtype, with no `AnyValue` round-trip for the numeric
/// fast paths.
enum TypedCol<'a> {
    U8(&'a UInt8Chunked),
    I8(&'a Int8Chunked),
    U16(&'a UInt16Chunked),
    I16(&'a Int16Chunked),
    U32(&'a UInt32Chunked),
    I32(&'a Int32Chunked),
    U64(&'a UInt64Chunked),
    I64(&'a Int64Chunked),
    F32(&'a Float32Chunked),
    F64(&'a Float64Chunked),
    /// String columns backing per-row enum parameters (`filter`, `mode`, …).
    Str(&'a StringChunked),
    /// Boolean columns backing per-row flag parameters.
    Bool(&'a BooleanChunked),
    /// Non-primitive columns (structs, lists, …): fall back to `AnyValue`.
    Other(&'a Series),
}

/// One input column wrapped for per-row parameter access.
pub struct ParamCol<'a> {
    series: &'a Series,
    typed: TypedCol<'a>,
    /// Scalar broadcasting: when an expression is an aggregation (like
    /// `.max()`), Polars passes a single-element series; that value applies
    /// to every row, matching Polars' contextual broadcasting behavior.
    broadcast: bool,
}

impl<'a> ParamCol<'a> {
    fn new(series: &'a Series) -> Self {
        let typed = match series.dtype() {
            DataType::UInt8 => TypedCol::U8(series.u8().unwrap()),
            DataType::Int8 => TypedCol::I8(series.i8().unwrap()),
            DataType::UInt16 => TypedCol::U16(series.u16().unwrap()),
            DataType::Int16 => TypedCol::I16(series.i16().unwrap()),
            DataType::UInt32 => TypedCol::U32(series.u32().unwrap()),
            DataType::Int32 => TypedCol::I32(series.i32().unwrap()),
            DataType::UInt64 => TypedCol::U64(series.u64().unwrap()),
            DataType::Int64 => TypedCol::I64(series.i64().unwrap()),
            DataType::Float32 => TypedCol::F32(series.f32().unwrap()),
            DataType::Float64 => TypedCol::F64(series.f64().unwrap()),
            DataType::String => TypedCol::Str(series.str().unwrap()),
            DataType::Boolean => TypedCol::Bool(series.bool().unwrap()),
            _ => TypedCol::Other(series),
        };
        ParamCol {
            series,
            typed,
            broadcast: series.len() == 1,
        }
    }

    /// The input column's name, for error messages.
    pub fn name(&self) -> &str {
        self.series.name()
    }

    /// The effective row index after scalar broadcasting.
    #[inline]
    fn value_index(&self, row_idx: usize) -> usize {
        if self.broadcast {
            0
        } else {
            row_idx
        }
    }

    /// The single point where a null per-row parameter is handled.
    ///
    /// Every null in an expression parameter — numeric, enum, flag, or list
    /// element, for any of the ~70 operations — reaches this function, so the
    /// policy is a shared mechanism rather than something each op re-declares.
    ///
    /// Both policies return `Err`, which is what makes this safe: resolution
    /// short-circuits, so no placeholder value can reach an operation. Under
    /// [`NullParamPolicy::Null`] the context is additionally flagged, and the
    /// caller that owns the row (`graph::compiled`, or `GeomParams::row`) turns
    /// that flag into a null result instead of propagating the error.
    fn on_null(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsError {
        if ctx.null_policy() == NullParamPolicy::Null {
            ctx.flag_null();
        }
        polars_err!(ComputeError:
            "Parameter column '{}' has a null value at row {}",
            self.series.name(), row_idx
        )
    }

    fn cast_err(&self, row_idx: usize, target: &str) -> PolarsError {
        polars_err!(ComputeError:
            "Parameter column '{}' value at row {} cannot be represented as {}",
            self.series.name(), row_idx, target
        )
    }

    /// Read the value at `row_idx` as i64 (truncating floats, like
    /// `AnyValue::try_extract`).
    pub fn get_i64(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<i64> {
        let idx = self.value_index(row_idx);
        let value: Option<i64> = match &self.typed {
            TypedCol::U8(ca) => ca.get(idx).map(i64::from),
            TypedCol::I8(ca) => ca.get(idx).map(i64::from),
            TypedCol::U16(ca) => ca.get(idx).map(i64::from),
            TypedCol::I16(ca) => ca.get(idx).map(i64::from),
            TypedCol::U32(ca) => ca.get(idx).map(i64::from),
            TypedCol::I32(ca) => ca.get(idx).map(i64::from),
            TypedCol::U64(ca) => match ca.get(idx) {
                Some(v) => Some(i64::try_from(v).map_err(|_| self.cast_err(row_idx, "i64"))?),
                None => None,
            },
            TypedCol::I64(ca) => ca.get(idx),
            TypedCol::F32(ca) => match ca.get(idx) {
                Some(v) => {
                    Some(float_to_i64(v as f64).ok_or_else(|| self.cast_err(row_idx, "i64"))?)
                }
                None => None,
            },
            TypedCol::F64(ca) => match ca.get(idx) {
                Some(v) => Some(float_to_i64(v).ok_or_else(|| self.cast_err(row_idx, "i64"))?),
                None => None,
            },
            // A Boolean column for a numeric parameter is a mis-routed
            // expression, not a 0/1 value the user asked for.
            TypedCol::Bool(_) | TypedCol::Str(_) => return Err(self.cast_err(row_idx, "i64")),
            // Route the fallback path's null through `on_null` too, rather than
            // letting `try_extract` report it as a cast failure.
            TypedCol::Other(s) => match s.get(idx)? {
                AnyValue::Null => None,
                v => return v.try_extract::<i64>(),
            },
        };
        value.ok_or_else(|| self.on_null(row_idx, ctx))
    }

    /// Read the value at `row_idx` as f64.
    pub fn get_f64(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<f64> {
        let idx = self.value_index(row_idx);
        let value: Option<f64> = match &self.typed {
            TypedCol::U8(ca) => ca.get(idx).map(f64::from),
            TypedCol::I8(ca) => ca.get(idx).map(f64::from),
            TypedCol::U16(ca) => ca.get(idx).map(f64::from),
            TypedCol::I16(ca) => ca.get(idx).map(f64::from),
            TypedCol::U32(ca) => ca.get(idx).map(f64::from),
            TypedCol::I32(ca) => ca.get(idx).map(f64::from),
            TypedCol::U64(ca) => ca.get(idx).map(|v| v as f64),
            TypedCol::I64(ca) => ca.get(idx).map(|v| v as f64),
            TypedCol::F32(ca) => ca.get(idx).map(f64::from),
            TypedCol::F64(ca) => ca.get(idx),
            TypedCol::Bool(_) | TypedCol::Str(_) => return Err(self.cast_err(row_idx, "f64")),
            TypedCol::Other(s) => match s.get(idx)? {
                AnyValue::Null => None,
                v => return v.try_extract::<f64>(),
            },
        };
        value.ok_or_else(|| self.on_null(row_idx, ctx))
    }

    /// Read the value at `row_idx` as a string, for per-row enum parameters.
    ///
    /// Only a genuine string column is accepted: a numeric column here means
    /// the user routed the wrong expression to an enum parameter, which must
    /// be an error rather than a silent fallback to the default.
    pub fn get_str(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<&'a str> {
        let idx = self.value_index(row_idx);
        match &self.typed {
            TypedCol::Str(ca) => ca.get(idx).ok_or_else(|| self.on_null(row_idx, ctx)),
            _ => Err(polars_err!(ComputeError:
                "Parameter column '{}' must be a String column for an enum \
                 parameter, got {}",
                self.series.name(), self.series.dtype()
            )),
        }
    }

    /// Read the value at `row_idx` as a boolean, for per-row flag parameters.
    pub fn get_bool(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<bool> {
        let idx = self.value_index(row_idx);
        match &self.typed {
            TypedCol::Bool(ca) => ca.get(idx).ok_or_else(|| self.on_null(row_idx, ctx)),
            _ => Err(polars_err!(ComputeError:
                "Parameter column '{}' must be a Boolean column for a flag \
                 parameter, got {}",
                self.series.name(), self.series.dtype()
            )),
        }
    }

    /// Read the value at `row_idx` as an `AnyValue`, for columns carrying
    /// *data* rather than a parameter value — currently only `label_reduce`'s
    /// contour operand, which travels by column name instead of as a
    /// `ParamValue`.
    ///
    /// Deliberately outside [`NullParamPolicy`]: it returns `AnyValue::Null`
    /// rather than routing through [`on_null`](Self::on_null), and its caller
    /// gives that its own meaning (an empty score vector). Do not reach for
    /// this accessor for an actual parameter — the typed accessors are what
    /// make the null policy uniform.
    pub fn get_any(&self, row_idx: usize) -> PolarsResult<AnyValue<'a>> {
        self.series.get(self.value_index(row_idx))
    }
}

/// Truncating float→int conversion matching `NumCast` semantics: `None` when
/// the value is not representable (NaN, ±inf, out of range).
fn float_to_i64(v: f64) -> Option<i64> {
    if v.is_nan() || v < i64::MIN as f64 || v >= i64::MAX as f64 {
        return None;
    }
    Some(v.trunc() as i64)
}

/// Per-call parameter context: typed accessors over the plugin's input series.
///
/// Indexed by the absolute input position that [`ParamValue::Slot`] was bound
/// to at graph-compile time. Built once per plugin call (per morsel).
#[derive(Default)]
pub struct ParamCtx<'a> {
    cols: Vec<ParamCol<'a>>,
    probe: bool,
    null_policy: NullParamPolicy,
    /// Set by [`ParamCol::on_null`] when a null was read under
    /// [`NullParamPolicy::Null`]. `Cell` because resolvers take `&ParamCtx`;
    /// each row range of a call builds its own context and reads the flag on
    /// its own thread, so no `Sync` bound is introduced.
    null_hit: Cell<bool>,
}

impl<'a> ParamCtx<'a> {
    /// Build a context over every plugin input series, applying `policy` to
    /// null parameter values.
    ///
    /// Source columns are included (slots never point at them, but absolute
    /// indexing keeps the binding trivial and collision-free).
    pub fn with_null_policy(inputs: &'a [Series], policy: NullParamPolicy) -> Self {
        ParamCtx {
            cols: inputs.iter().map(ParamCol::new).collect(),
            probe: false,
            null_policy: policy,
            null_hit: Cell::new(false),
        }
    }

    /// Build a *plan-time probe* context (see `lib.rs::op_infer_shape`).
    ///
    /// Shape probing binds every expression parameter to an integer
    /// placeholder so it can detect which output dimensions depend on a
    /// per-row value. A dynamic enum or flag parameter cannot consume an
    /// integer, so accessors ask [`is_probe`](Self::is_probe) and substitute
    /// the parameter's documented default instead.
    ///
    /// That substitution is sound only because a parameter may become per-row
    /// exclusively when it has **no effect on output shape, rank, or dtype**
    /// (see `get::req_enum`), so which variant probing picks cannot change the
    /// inferred schema. Signalling this explicitly — rather than inferring it
    /// from the placeholder's dtype — keeps real execution strict: a user who
    /// routes an integer column into an enum parameter still gets an error.
    pub fn probe(inputs: &'a [Series]) -> Self {
        ParamCtx {
            cols: inputs.iter().map(ParamCol::new).collect(),
            probe: true,
            // Probe placeholders are synthesised non-null integers, so the
            // policy is unreachable here; `Raise` keeps probing strict.
            null_policy: NullParamPolicy::Raise,
            null_hit: Cell::new(false),
        }
    }

    /// Whether this is a plan-time shape probe rather than real execution.
    pub fn is_probe(&self) -> bool {
        self.probe
    }

    /// The policy this context applies to null parameter values.
    pub fn null_policy(&self) -> NullParamPolicy {
        self.null_policy
    }

    /// Record that a null parameter was read under [`NullParamPolicy::Null`].
    pub(crate) fn flag_null(&self) {
        self.null_hit.set(true);
    }

    /// Clear the null flag before a unit of work whose outcome will be tested
    /// with [`took_null`](Self::took_null).
    pub fn clear_null(&self) {
        self.null_hit.set(false);
    }

    /// Whether the work since the last [`clear_null`](Self::clear_null) read a
    /// null parameter that the policy says should yield a null result.
    ///
    /// Only meaningful alongside an `Err`: `on_null` always returns an error so
    /// resolution short-circuits, and this distinguishes "null parameter, null
    /// result wanted" from a genuine failure.
    pub fn took_null(&self) -> bool {
        self.null_hit.get()
    }

    /// An empty context, for resolving all-literal op specs.
    pub fn empty() -> Self {
        ParamCtx::default()
    }

    /// Look up a bound column by slot index.
    pub fn col(&self, idx: usize) -> PolarsResult<&ParamCol<'a>> {
        self.cols.get(idx).ok_or_else(|| {
            polars_err!(ComputeError:
                "Parameter slot {} out of bounds ({} input columns); \
                 expression parameter column was not passed to the plugin",
                idx, self.cols.len()
            )
        })
    }
}

/// One op's parameter map, plus the record of which names were looked up.
///
/// A legacy op (`LegacyOpSpec`, the not-yet-typed half of `OpSpec`) is the
/// exception to this crate's `deny_unknown_fields` rule — its params are a
/// free map — so the wire format cannot refuse a parameter
/// no operation understands. Nothing else refused one either: `scale` and
/// `clamp` both accepted an `out_dtype` that entered the op's identity (and so
/// the CSE and compiled-graph cache keys) and was then read by no `resolve_op`
/// arm and no dtype rule.
///
/// This type closes that hole at the only door into the arms. Every read goes
/// through [`OpParams::get`] or [`OpParams::contains_key`] and is recorded;
/// [`resolve_op`](crate::execute::resolve_op) then rejects anything the arm did
/// not touch. An arm cannot opt out, because it never receives the underlying
/// map — which is the point: a guard listing the parameters each op must
/// remember to read would go stale the day someone adds one it has not heard
/// of.
///
/// Reads are recorded in a bitmask over the map's key order rather than a set
/// of names, so tracking allocates nothing: `resolve_op` runs per row for ops
/// with expression parameters, and ops carry a handful of parameters at most.
pub struct OpParams<'a> {
    map: &'a HashMap<String, ParamValue>,
    read: Cell<u64>,
}

impl<'a> OpParams<'a> {
    /// The widest parameter map the read-tracking bitmask can cover.
    const CAPACITY: usize = 64;

    /// Wrap an op's parameters for tracked access.
    pub fn new(map: &'a HashMap<String, ParamValue>) -> Self {
        Self {
            map,
            read: Cell::new(0),
        }
    }

    /// Record `name` as read, if the map carries it.
    fn mark(&self, name: &str) {
        if let Some(idx) = self.map.keys().position(|k| k == name) {
            if idx < Self::CAPACITY {
                self.read.set(self.read.get() | (1u64 << idx));
            }
        }
    }

    /// Look up a parameter, recording the read.
    pub fn get(&self, name: &str) -> Option<&'a ParamValue> {
        self.mark(name);
        self.map.get(name)
    }

    /// Test for a parameter's presence, recording the read.
    pub fn contains_key(&self, name: &str) -> bool {
        self.mark(name);
        self.map.contains_key(name)
    }

    /// Record `name` as read without returning it.
    ///
    /// For parameters an arm legitimately does not consume because a layer
    /// above it does: `rasterize`'s `shape_ref` names another graph node, and
    /// is resolved by `CompiledGraph::compile` before the op is reached. The
    /// acknowledgement is explicit and lives on the arm, so the parameter is
    /// *declared* as belonging to the op rather than special-cased inside the
    /// checker, where it would read as a hole in the rule.
    pub fn acknowledge(&self, name: &str) {
        self.mark(name);
    }

    /// Parameter names present on the wire that nothing read.
    ///
    /// Empty is the only acceptable result; see [`OpParams`].
    pub fn unread(&self) -> PolarsResult<Vec<&'a str>> {
        if self.map.len() > Self::CAPACITY {
            // Refusing to answer beats answering wrongly: past `CAPACITY` the
            // mask cannot represent the read, and reporting those names as
            // unread would be a false accusation while ignoring them would be
            // a silent blind spot.
            polars_bail!(ComputeError:
                "operation has {} parameters, more than the {} the parameter-use \
                 checker can track; raise OpParams::CAPACITY",
                self.map.len(), Self::CAPACITY);
        }
        let read = self.read.get();
        Ok(self
            .map
            .keys()
            .enumerate()
            .filter(|(idx, _)| read & (1u64 << idx) == 0)
            .map(|(_, k)| k.as_str())
            .collect())
    }
}

/// Shared accessors for optional and enum-valued operation parameters.
///
/// These implement the **single parameter failure policy** for `resolve_op`:
/// an *absent* optional parameter takes its documented default, while a
/// parameter that is *present but invalid* — unknown enum string, wrong type,
/// out-of-range value, or a per-row expression that fails to resolve — is
/// always an error. Helpers never swallow a resolution error into a default
/// (guarded by `execute::strict_param_tests`).
///
/// A **null** per-row value is the one thing that is not covered here, because
/// it is not a matter of validity: it is governed by [`NullParamPolicy`] at
/// [`ParamCol::on_null`], the layer below. Under `Raise` it reaches these
/// helpers as an ordinary error and the policy above applies unchanged; under
/// `Null` the error still propagates out of `resolve_op`, but the caller that
/// owns the row recognises it and nulls the row instead. Either way, no helper
/// here may turn a null into its default — that would silently compute a wrong
/// result for a missing input.
pub mod get {
    use super::{OpParams, ParamCtx, ParamValue};
    use polars::prelude::*;

    /// Every helper here reads through the tracking wrapper, so using one is
    /// what records the parameter as consumed. There is no untracked overload.
    type Params<'a> = OpParams<'a>;

    fn named(name: &str, e: PolarsError) -> PolarsError {
        polars_err!(ComputeError: "parameter '{}': {}", name, e)
    }

    /// Optional boolean. Booleans are structural: only a literal
    /// `true`/`false` is accepted — strings, numbers, and expressions error
    /// instead of silently reading as `false`.
    pub fn opt_bool(params: &Params<'_>, name: &str, default: bool) -> PolarsResult<bool> {
        match params.get(name) {
            None => Ok(default),
            Some(ParamValue::Literal {
                value: serde_json::Value::Bool(b),
            }) => Ok(*b),
            Some(other) => Err(polars_err!(ComputeError:
                "parameter '{}' must be a boolean literal (true/false), got {:?}",
                name, other
            )),
        }
    }

    /// Optional u32 for a **structural** parameter that fixes the output
    /// shape/length (e.g. `perceptual_hash(hash_size)`). Literal-only: a bound
    /// expression `Slot` (or any non-literal form) errors, because letting the
    /// value vary per row would desync the plan-time schema from the data.
    pub fn opt_u32_literal(params: &Params<'_>, name: &str, default: u32) -> PolarsResult<u32> {
        match params.get(name) {
            None => Ok(default),
            Some(ParamValue::Literal { value }) => {
                let v = value.as_i64().ok_or_else(|| {
                    named(
                        name,
                        polars_err!(ComputeError: "expected an integer literal, got {:?}", value),
                    )
                })?;
                u32::try_from(v).map_err(|_| {
                    named(
                        name,
                        polars_err!(ComputeError: "value {} out of range for u32", v),
                    )
                })
            }
            Some(_) => Err(structural_literal_only(name)),
        }
    }

    /// Optional usize structural parameter where absence is meaningful (e.g. a
    /// reduction `axis`: absent = "global"). Literal-only: the axis fixes the
    /// output rank at plan time, so a bound expression `Slot` errors rather
    /// than silently changing rank per row.
    pub fn maybe_usize_literal(params: &Params<'_>, name: &str) -> PolarsResult<Option<usize>> {
        match params.get(name) {
            None => Ok(None),
            Some(ParamValue::Literal { value }) => {
                let v = value.as_i64().ok_or_else(|| {
                    named(
                        name,
                        polars_err!(ComputeError: "expected an integer literal, got {:?}", value),
                    )
                })?;
                if v < 0 {
                    return Err(named(
                        name,
                        polars_err!(ComputeError: "value {} cannot be negative", v),
                    ));
                }
                Ok(Some(v as usize))
            }
            Some(_) => Err(structural_literal_only(name)),
        }
    }

    /// The uniform error for a structural parameter given a per-row expression.
    fn structural_literal_only(name: &str) -> PolarsError {
        polars_err!(ComputeError:
            "parameter '{}' is structural (it fixes the output shape/rank at \
             planning time) and must be a literal, not a per-row expression",
            name)
    }

    /// Optional f64 where absence is meaningful (e.g. `min_area`: absent
    /// means "no filter"). Present-but-invalid still errors.
    pub fn maybe_f64(
        params: &Params<'_>,
        name: &str,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<Option<f64>> {
        params
            .get(name)
            .map(|p| p.resolve_f64(row_idx, ctx).map_err(|e| named(name, e)))
            .transpose()
    }

    /// Optional f64 with a default for absence.
    pub fn opt_f64(
        params: &Params<'_>,
        name: &str,
        default: f64,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<f64> {
        params
            .get(name)
            .map(|p| p.resolve_f64(row_idx, ctx).map_err(|e| named(name, e)))
            .transpose()
            .map(|v| v.unwrap_or(default))
    }

    /// Optional u8 with a default for absence; range-checked so 300 errors
    /// instead of silently truncating.
    pub fn opt_u8(
        params: &Params<'_>,
        name: &str,
        default: u8,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<u8> {
        opt_u8_value(params.get(name), name, default, row_idx, ctx)
    }

    /// [`opt_u8`] for a parameter the caller already holds.
    ///
    /// `SourceSpec` keeps its per-row parameters in named fields rather than a
    /// map, so it cannot look one up by name — but it must not grow a second
    /// copy of this logic, or a future change to null handling or range
    /// checking would have to land in two places.
    pub fn opt_u8_value(
        param: Option<&ParamValue>,
        name: &str,
        default: u8,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<u8> {
        match param {
            None => Ok(default),
            Some(p) => {
                let v = p.resolve_i64(row_idx, ctx).map_err(|e| named(name, e))?;
                u8::try_from(v).map_err(|_| {
                    polars_err!(ComputeError:
                        "parameter '{}' must be in 0..=255, got {}", name, v)
                })
            }
        }
    }

    /// Required enum-valued parameter, resolved per row against a canonical
    /// `NAMED`-style table (plus parser-only aliases). Unknown values error
    /// with the canonical names listed.
    ///
    /// `default` is used when the parameter is bound to a column under a
    /// plan-time probe context, where no real value exists yet. Only pass a
    /// parameter through here when its value has **no effect on output shape,
    /// rank, or dtype** — that invariant is what makes probing with the
    /// default sound (see [`ParamCtx::probe`]). Structural string parameters
    /// must keep using [`ParamValue::resolve_string`], which rejects
    /// expressions outright.
    pub fn req_enum<T: Copy>(
        params: &Params<'_>,
        name: &str,
        canonical: &[(&str, T)],
        aliases: &[(&str, T)],
        default: T,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<T> {
        let param = params
            .get(name)
            .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: {}", name))?;
        let Some(s) = param
            .resolve_str(row_idx, ctx)
            .map_err(|e| named(name, e))?
        else {
            return Ok(default);
        };
        view_buffer::naming::lookup(canonical, s)
            .or_else(|| view_buffer::naming::lookup(aliases, s))
            .ok_or_else(|| {
                polars_err!(ComputeError:
                    "parameter '{}': unknown value '{}', expected one of {:?}",
                    name, s, view_buffer::naming::names(canonical)
                )
            })
    }

    /// Optional enum-valued parameter with a default for absence.
    pub fn opt_enum<T: Copy>(
        params: &Params<'_>,
        name: &str,
        canonical: &[(&str, T)],
        aliases: &[(&str, T)],
        default: T,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<T> {
        if params.contains_key(name) {
            req_enum(params, name, canonical, aliases, default, row_idx, ctx)
        } else {
            Ok(default)
        }
    }

    /// Required enum-valued parameter that must be a literal.
    ///
    /// For enums that *are* structural — they feed dtype inference or fix the
    /// output length — so a per-row value would desync the lazy schema from
    /// the produced data. Rejects a bound expression slot as defense in depth,
    /// mirroring `opt_u32_literal` / `maybe_usize_literal` for numbers.
    pub fn req_enum_literal<T: Copy>(
        params: &Params<'_>,
        name: &str,
        canonical: &[(&str, T)],
        aliases: &[(&str, T)],
    ) -> PolarsResult<T> {
        let param = params
            .get(name)
            .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: {}", name))?;
        // Reject an expression here, then reuse the shared lookup. An empty
        // probe context is safe precisely because the literal-only check above
        // means no slot can be read.
        param.resolve_string().map_err(|e| named(name, e))?;
        req_enum(
            params,
            name,
            canonical,
            aliases,
            canonical[0].1,
            0,
            &ParamCtx::empty(),
        )
    }

    /// Optional literal-only enum parameter with a default for absence.
    pub fn opt_enum_literal<T: Copy>(
        params: &Params<'_>,
        name: &str,
        canonical: &[(&str, T)],
        aliases: &[(&str, T)],
        default: T,
    ) -> PolarsResult<T> {
        if params.contains_key(name) {
            req_enum_literal(params, name, canonical, aliases)
        } else {
            Ok(default)
        }
    }

    /// Optional boolean parameter, resolved per row.
    ///
    /// The per-row counterpart of [`opt_bool`], for flags with no shape or
    /// dtype effect (e.g. `apply_mask(invert)`). Flags that
    /// *do* change the output shape — `rotate(expand)` — must keep using
    /// [`opt_bool`], which is literal-only.
    pub fn opt_bool_dyn(
        params: &Params<'_>,
        name: &str,
        default: bool,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<bool> {
        match params.get(name) {
            None => Ok(default),
            Some(p) => Ok(p
                .resolve_bool(row_idx, ctx)
                .map_err(|e| named(name, e))?
                .unwrap_or(default)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `NAMED` spelling must parse through serde to the variant it names.
    ///
    /// The twin of `row_error_policy_names_match_serde` in `graph::types`, for
    /// the same reason: serde's `rename_all` reads the wire while `NAMED` tells
    /// Python what to write, and nothing else compares the two. A rename on one
    /// side would leave Python sending a value the graph cannot parse.
    ///
    /// Only this direction is checked. The reverse — that serde accepts
    /// *nothing* `NAMED` does not publish — would need to enumerate serde's
    /// accepted spellings, which it does not expose; a `#[serde(alias)]`
    /// added to a variant would therefore pass unpublished to Python. Say so
    /// rather than claim a round trip this does not make.
    ///
    /// A second copy of a *test* over a different type, not a second copy of a
    /// fact — the registry cannot express "deserialize this" generically,
    /// because it stores name functions rather than the types themselves.
    #[test]
    fn null_param_policy_names_match_serde() {
        for (name, expected) in NullParamPolicy::NAMED {
            let parsed: NullParamPolicy = serde_json::from_str(&format!("\"{name}\""))
                .unwrap_or_else(|e| panic!("serde rejects the NAMED spelling {name:?}: {e}"));
            assert_eq!(parsed, *expected, "{name} parses to the wrong variant");
        }
    }

    #[test]
    fn test_literal_i64() {
        let param = ParamValue::Literal {
            value: serde_json::json!(42),
        };
        assert!(param.is_literal());
        assert_eq!(param.resolve_i64(0, &ParamCtx::empty()).unwrap(), 42);
    }

    #[test]
    fn plain_literal_array_is_literal() {
        // A normalize mean/std or flip-axes array of plain scalars is literal.
        let param = ParamValue::Literal {
            value: serde_json::json!([0.485, 0.456, 0.406]),
        };
        assert!(param.is_literal());
    }

    #[test]
    fn test_literal_f64() {
        let param = ParamValue::Literal {
            value: serde_json::json!(1.5),
        };
        assert_eq!(param.resolve_f64(0, &ParamCtx::empty()).unwrap(), 1.5);
    }

    #[test]
    fn test_literal_string() {
        let param = ParamValue::Literal {
            value: serde_json::json!("hello"),
        };
        assert_eq!(param.resolve_string().unwrap(), "hello");
    }

    #[test]
    fn test_slot_typed_read() {
        let s = Series::new("h".into(), &[10i64, 20, 30]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(2, &ctx).unwrap(), 30);
        assert_eq!(param.resolve_i64(1, &ctx).unwrap(), 20);
    }

    #[test]
    fn test_slot_broadcast_scalar() {
        // A one-element series (aggregation result) broadcasts to all rows.
        let s = Series::new("h".into(), &[7i32]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(0, &ctx).unwrap(), 7);
        assert_eq!(param.resolve_i64(99, &ctx).unwrap(), 7);
    }

    #[test]
    fn test_slot_null_value_errors() {
        let s = Series::new("h".into(), &[Some(1i64), None]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(0, &ctx).unwrap(), 1);
        assert!(param.resolve_i64(1, &ctx).is_err());
        // Under `Raise` the context is never flagged, so callers cannot
        // mistake a genuine failure for a null-parameter row.
        assert!(!ctx.took_null());
    }

    #[test]
    fn test_null_policy_flags_the_context() {
        // Under `Null` the read still errors — so resolution short-circuits
        // and no placeholder value reaches an op — but the context is flagged
        // so the row's owner can null it instead of propagating.
        let s = Series::new("h".into(), &[Some(1i64), None]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Null);
        let param = ParamValue::Slot { idx: 0 };

        ctx.clear_null();
        assert_eq!(param.resolve_i64(0, &ctx).unwrap(), 1);
        assert!(!ctx.took_null());

        ctx.clear_null();
        assert!(param.resolve_i64(1, &ctx).is_err());
        assert!(ctx.took_null());
    }

    #[test]
    fn test_null_policy_does_not_flag_other_failures() {
        // A wrong-dtype column is a user error, not a null: it must stay an
        // error under `Null` rather than silently nulling the row.
        let s = Series::new("h".into(), &["not-a-number"]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Null);
        let param = ParamValue::Slot { idx: 0 };

        ctx.clear_null();
        assert!(param.resolve_i64(0, &ctx).is_err());
        assert!(!ctx.took_null());
    }

    #[test]
    fn test_null_policy_covers_every_accessor() {
        // Numeric, enum and flag parameters all route through `on_null`.
        let ints = Series::new("i".into(), &[None::<i64>]);
        let floats = Series::new("f".into(), &[None::<f64>]);
        let strings = Series::new("s".into(), &[None::<&str>]);
        let bools = Series::new("b".into(), &[None::<bool>]);
        let inputs = vec![ints, floats, strings, bools];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Null);

        for idx in 0..4 {
            let param = ParamValue::Slot { idx };
            ctx.clear_null();
            let result: PolarsResult<()> = match idx {
                0 => param.resolve_i64(0, &ctx).map(|_| ()),
                1 => param.resolve_f64(0, &ctx).map(|_| ()),
                2 => param.resolve_str(0, &ctx).map(|_| ()),
                _ => param.resolve_bool(0, &ctx).map(|_| ()),
            };
            assert!(result.is_err(), "slot {idx} should error");
            assert!(ctx.took_null(), "slot {idx} should flag the null");
        }
    }

    #[test]
    fn test_null_policy_covers_non_primitive_columns() {
        // The `TypedCol::Other` fallback used to report `try_extract`'s cast
        // error for a null, bypassing the null path entirely.
        let s = Series::new("d".into(), &[None::<i64>])
            .cast(&DataType::Duration(TimeUnit::Milliseconds))
            .unwrap();
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Null);
        let param = ParamValue::Slot { idx: 0 };

        ctx.clear_null();
        assert!(param.resolve_i64(0, &ctx).is_err());
        assert!(ctx.took_null());
    }

    #[test]
    fn test_slot_float_truncates_like_try_extract() {
        let s = Series::new("h".into(), &[3.9f64, -2.7]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(0, &ctx).unwrap(), 3);
        assert_eq!(param.resolve_i64(1, &ctx).unwrap(), -2);
    }

    #[test]
    fn test_structural_literal_resolvers_reject_bound_slots() {
        use super::get;
        use std::collections::HashMap;

        // A structural param that reached the resolver as a per-row slot must
        // error — letting it vary per row would desync the plan-time schema
        // from the data.
        let bad = ParamValue::Slot { idx: 0 };
        let mut params: HashMap<String, ParamValue> = HashMap::new();
        params.insert("axis".into(), bad.clone());
        let err = get::maybe_usize_literal(&OpParams::new(&params), "axis").unwrap_err();
        assert!(err.to_string().contains("structural"), "got: {err}");

        let mut params2: HashMap<String, ParamValue> = HashMap::new();
        params2.insert("hash_size".into(), bad);
        let err = get::opt_u32_literal(&OpParams::new(&params2), "hash_size", 64).unwrap_err();
        assert!(err.to_string().contains("structural"), "got: {err}");

        // A literal still resolves normally, and absence yields the default.
        let mut lit: HashMap<String, ParamValue> = HashMap::new();
        lit.insert(
            "axis".into(),
            ParamValue::Literal {
                value: serde_json::json!(2),
            },
        );
        assert_eq!(
            get::maybe_usize_literal(&OpParams::new(&lit), "axis").unwrap(),
            Some(2)
        );
        let empty: HashMap<String, ParamValue> = HashMap::new();
        assert_eq!(
            get::opt_u32_literal(&OpParams::new(&empty), "hash_size", 64).unwrap(),
            64
        );
    }

    #[test]
    fn the_wire_form_is_a_literal_or_a_slot() {
        let lit: ParamValue = serde_json::from_str(r#"{"type": "literal", "value": 7}"#).unwrap();
        assert!(matches!(lit, ParamValue::Literal { ref value } if value == &serde_json::json!(7)));
        let slot: ParamValue = serde_json::from_str(r#"{"$slot": 3}"#).unwrap();
        assert!(matches!(slot, ParamValue::Slot { idx: 3 }));
        for (param, wire) in [
            (lit, serde_json::json!({"type": "literal", "value": 7})),
            (slot, serde_json::json!({"$slot": 3})),
        ] {
            assert_eq!(serde_json::to_value(&param).unwrap(), wire);
        }
    }

    #[test]
    fn a_nested_list_serializes_as_its_literal_form() {
        let list = ParamValue::List(vec![
            ParamValue::Literal {
                value: serde_json::json!(1.0),
            },
            ParamValue::Slot { idx: 2 },
        ]);
        assert_eq!(
            serde_json::to_value(&list).unwrap(),
            serde_json::json!({"type": "literal", "value": [
                {"type": "literal", "value": 1.0}, {"$slot": 2}
            ]})
        );
    }

    #[test]
    fn anything_else_on_the_wire_is_rejected() {
        for bad in [
            // The removed name-keyed expression form.
            r#"{"type": "expr", "col": "h"}"#,
            r#"{"$slot": -1}"#,
            r#"{"$slot": 1, "type": "literal"}"#,
            r#"{"type": "literal", "value": 1, "extra": 2}"#,
            r#"{"type": "literal"}"#,
            r#"7"#,
        ] {
            let err = serde_json::from_str::<ParamValue>(bad).unwrap_err();
            assert!(
                err.to_string().contains("expected a parameter")
                    || err.to_string().contains("slot index"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn test_param_list_slice_and_owned() {
        // A compiled List is borrowed via as_param_slice (hot path) and cloned
        // via as_param_list; the Literal JSON form has no slice but parses.
        let list = ParamValue::List(vec![
            ParamValue::Literal {
                value: serde_json::json!(1.0),
            },
            ParamValue::Slot { idx: 2 },
        ]);
        assert_eq!(list.as_param_slice().map(|s| s.len()), Some(2));
        assert_eq!(list.as_param_list().unwrap().len(), 2);
        assert!(!list.is_literal()); // contains a Slot

        let json_form = ParamValue::Literal {
            value: serde_json::json!([{"type": "literal", "value": 1.0}]),
        };
        assert!(json_form.as_param_slice().is_none());
        assert_eq!(json_form.as_param_list().unwrap().len(), 1);
    }
}
