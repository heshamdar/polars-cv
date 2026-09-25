//! The per-call view of expression-parameter columns.
//!
//! A per-row parameter arrives as `{"$slot": n}` (a typed
//! [`Param<T>`](crate::ops::Param)): the absolute index of its column among the
//! plugin's input series, assigned by the Python `SlotTable`. Per-row
//! resolution is a direct indexed read through a typed accessor ([`ParamCol`]);
//! no names are involved. [`NullParamPolicy`] says what a null in such a column
//! means.

use polars::prelude::*;
use std::cell::Cell;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NullParamPolicy {
    /// A null parameter fails the expression (the default).
    #[default]
    Raise,
    /// A null parameter makes the node produce no output for that row, which
    /// propagates as a null exactly like a null *input* already does.
    Null,
}

view_buffer::naming::named_variants!(NullParamPolicy: "What a null in a per-row expression parameter means.\n\nDeliberately separate from :class:`RowErrorPolicy`: under ``NULL`` a null\nparameter is not an error, so it records no ``_error`` message and does not\nweaken reporting for genuine decode/encode/operation failures.\n- RAISE: a null parameter fails the expression.\n- NULL: the affected node produces no output for that row." {
    "raise" => Raise,
    "null" => Null,
});

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
    /// contour operand (a `ColumnRef`).
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
/// Indexed by the absolute input position a `{"$slot": n}` names (see
/// [`Param`](crate::ops::Param)). Built once per plugin call (per morsel).
#[derive(Default)]
pub struct ParamCtx<'a> {
    cols: Vec<ParamCol<'a>>,
    /// Resolving an op at plan time, for its rules (see [`planning`](Self::planning)).
    planning: bool,
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
            planning: false,
            null_policy: policy,
            null_hit: Cell::new(false),
        }
    }

    /// A *plan-time* context: an op is resolved without any row to read its
    /// rules (domain, dtype, rank, channels, identity), which is sound because
    /// a parameter may be per-row exclusively when it has **no effect on
    /// output shape, rank, or dtype** (a `Literal<T>` field cannot hold a
    /// slot). Every per-row parameter therefore resolves to
    /// [`WireScalar::planning_value`](view_buffer::naming::WireScalar::planning_value).
    /// The shape is not read this way: it is symbolic
    /// ([`TypedOp::shape`](crate::ops::TypedOp::shape)).
    pub fn planning() -> Self {
        ParamCtx {
            cols: Vec::new(),
            planning: true,
            // No column is read, so the policy is unreachable; `Raise` keeps
            // planning strict.
            null_policy: NullParamPolicy::Raise,
            null_hit: Cell::new(false),
        }
    }

    /// Whether this resolves an op at plan time rather than for a row.
    pub fn is_planning(&self) -> bool {
        self.planning
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_typed_read() {
        let s = Series::new("h".into(), &[10i64, 20, 30]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ctx.col(0).unwrap();
        assert_eq!(param.get_i64(2, &ctx).unwrap(), 30);
        assert_eq!(param.get_i64(1, &ctx).unwrap(), 20);
    }

    #[test]
    fn test_slot_broadcast_scalar() {
        // A one-element series (aggregation result) broadcasts to all rows.
        let s = Series::new("h".into(), &[7i32]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ctx.col(0).unwrap();
        assert_eq!(param.get_i64(0, &ctx).unwrap(), 7);
        assert_eq!(param.get_i64(99, &ctx).unwrap(), 7);
    }

    #[test]
    fn test_slot_null_value_errors() {
        let s = Series::new("h".into(), &[Some(1i64), None]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ctx.col(0).unwrap();
        assert_eq!(param.get_i64(0, &ctx).unwrap(), 1);
        assert!(param.get_i64(1, &ctx).is_err());
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
        let param = ctx.col(0).unwrap();

        ctx.clear_null();
        assert_eq!(param.get_i64(0, &ctx).unwrap(), 1);
        assert!(!ctx.took_null());

        ctx.clear_null();
        assert!(param.get_i64(1, &ctx).is_err());
        assert!(ctx.took_null());
    }

    #[test]
    fn test_null_policy_does_not_flag_other_failures() {
        // A wrong-dtype column is a user error, not a null: it must stay an
        // error under `Null` rather than silently nulling the row.
        let s = Series::new("h".into(), &["not-a-number"]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Null);
        let param = ctx.col(0).unwrap();

        ctx.clear_null();
        assert!(param.get_i64(0, &ctx).is_err());
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
            ctx.clear_null();
            let result: PolarsResult<()> = match idx {
                0 => ctx.col(idx).and_then(|c| c.get_i64(0, &ctx)).map(|_| ()),
                1 => ctx.col(idx).and_then(|c| c.get_f64(0, &ctx)).map(|_| ()),
                2 => ctx.col(idx).and_then(|c| c.get_str(0, &ctx)).map(|_| ()),
                _ => ctx.col(idx).and_then(|c| c.get_bool(0, &ctx)).map(|_| ()),
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
        let param = ctx.col(0).unwrap();

        ctx.clear_null();
        assert!(param.get_i64(0, &ctx).is_err());
        assert!(ctx.took_null());
    }

    #[test]
    fn test_slot_float_truncates_like_try_extract() {
        let s = Series::new("h".into(), &[3.9f64, -2.7]);
        let inputs = vec![s];
        let ctx = ParamCtx::with_null_policy(&inputs, NullParamPolicy::Raise);
        let param = ctx.col(0).unwrap();
        assert_eq!(param.get_i64(0, &ctx).unwrap(), 3);
        assert_eq!(param.get_i64(1, &ctx).unwrap(), -2);
    }
}
