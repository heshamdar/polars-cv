//! Parameter value types for expression resolution.
//!
//! This module handles the resolution of parameter values that can be either
//! literals (known at planning time) or expressions (resolved per-row).
//!
//! Expression parameters are referenced by column name in the serialized graph
//! JSON. At graph **compile** time (see `graph::compiled`) every `Expr` param
//! is bound to a [`ParamValue::Slot`] — an integer index into the plugin's
//! input series — so per-row resolution is a direct indexed read through a
//! typed accessor ([`ParamCol`]) instead of a string-keyed map lookup plus
//! `AnyValue` extraction.

use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// A parameter value that can be either a literal or an expression reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ParamValue {
    /// A literal value known at planning time.
    #[serde(rename = "literal")]
    Literal {
        /// The literal value.
        value: serde_json::Value,
    },

    /// A reference to a column expression, resolved at execution time.
    ///
    /// This is the wire form. It must be bound to a [`ParamValue::Slot`]
    /// during graph compilation before per-row resolution.
    #[serde(rename = "expr")]
    Expr {
        /// Column name to resolve.
        #[serde(default)]
        col: Option<String>,
    },

    /// A compile-time-bound reference to an input column by index.
    ///
    /// Never serialized — produced from `Expr` by graph compilation
    /// (`graph::compiled::CompiledGraph`). The index points into the plugin's
    /// input series slice (source columns first, expression columns after).
    #[serde(skip)]
    Slot {
        /// Absolute index into the plugin input series.
        idx: usize,
    },

    /// A pre-parsed, already-bound nested parameter list.
    ///
    /// Never serialized — produced by graph compilation from a `Literal` whose
    /// JSON value is an array of `ParamValue` dicts (a `warp_affine` matrix, a
    /// `reshape` shape). Parsing and slot-binding happen once at compile time so
    /// per-row resolution reads the already-bound elements directly (via
    /// [`ParamValue::as_param_slice`]) instead of re-deserializing the JSON every
    /// row. The introspection path (`op_schema`) does not compile the graph and
    /// keeps the `Literal` JSON form, so `as_param_list` handles both.
    #[serde(skip)]
    List(Vec<ParamValue>),
}

impl ParamValue {
    /// Check if this parameter is fully literal (statically resolvable at compile
    /// time with an empty context).
    ///
    /// Nested parameter lists (a `warp_affine` matrix, a `reshape` shape) are
    /// hoisted into [`ParamValue::List`] by graph compilation
    /// (`graph::compiled::bind_param`) *before* this classification runs, so a
    /// bare `Literal` never holds nested sub-params and is always fully literal;
    /// a `List` is literal iff every element is (a bound `Slot`/`Expr` element
    /// makes the whole op dynamic, so it re-resolves per row).
    pub fn is_literal(&self) -> bool {
        match self {
            ParamValue::Literal { .. } => true,
            ParamValue::List(items) => items.iter().all(ParamValue::is_literal),
            ParamValue::Expr { .. } | ParamValue::Slot { .. } => false,
        }
    }

    /// Look up this param's bound column in the context.
    fn slot_col<'c, 'a>(&self, ctx: &'c ParamCtx<'a>) -> PolarsResult<&'c ParamCol<'a>> {
        match self {
            ParamValue::Slot { idx } => ctx.col(*idx),
            ParamValue::Expr { col, .. } => Err(polars_err!(ComputeError:
                "Internal error: unbound expression parameter (col: {:?}); \
                 the graph must be compiled before execution",
                col
            )),
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
            _ => self.slot_col(ctx)?.get_i64(row_idx),
        }
    }

    /// Resolve this parameter to a concrete u32 value.
    pub fn resolve_u32(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<u32> {
        let value = self.resolve_i64(row_idx, ctx)?;
        if value < 0 || value > u32::MAX as i64 {
            return Err(polars_err!(ComputeError: "Value {} out of range for u32", value));
        }
        Ok(value as u32)
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
            _ => self.slot_col(ctx)?.get_f64(row_idx),
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
            ParamValue::Expr { .. } | ParamValue::Slot { .. } | ParamValue::List(_) => {
                Err(polars_err!(ComputeError:
                    "This string parameter is structural (it fixes the output \
                     dtype at planning time) and cannot be an expression"))
            }
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
            _ => self.slot_col(ctx)?.get_str(row_idx).map(Some),
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
            _ => self.slot_col(ctx)?.get_bool(row_idx).map(Some),
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
            ParamValue::Expr { .. } | ParamValue::Slot { .. } => {
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
    /// element *count* is structural but the values are not. Axis lists, which
    /// reorder dimensions and therefore change the output rank, must keep using
    /// [`as_int_list`](Self::as_int_list).
    pub fn resolve_usize_list(&self, row_idx: usize, ctx: &ParamCtx) -> PolarsResult<Vec<usize>> {
        let mut owned = None;
        self.param_elements(&mut owned)?
            .iter()
            .map(|p| p.resolve_usize(row_idx, ctx))
            .collect()
    }

    /// Get literal value as a list of integers (for transpose, flip axes).
    pub fn as_int_list(&self) -> PolarsResult<Vec<usize>> {
        match self {
            ParamValue::Literal { value } => {
                let arr = value.as_array().ok_or_else(
                    || polars_err!(ComputeError: "Expected array literal, got {:?}", value),
                )?;

                arr.iter()
                    .map(|v| {
                        v.as_i64()
                            .and_then(|i| {
                                if i < 0 {
                                    None
                                } else {
                                    Some(i as usize)
                                }
                            })
                            .ok_or_else(|| polars_err!(ComputeError: "Expected non-negative integer in array"))
                    })
                    .collect()
            }
            ParamValue::Expr { .. } | ParamValue::Slot { .. } | ParamValue::List(_) => {
                Err(polars_err!(ComputeError: "Axes parameters cannot be expressions"))
            }
        }
    }

    /// Get literal value as a Vec<f64> (for histogram bins).
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            ParamValue::Literal { value } => {
                let arr = value.as_array()?;
                arr.iter().map(|v| v.as_f64()).collect::<Option<Vec<f64>>>()
            }
            ParamValue::Expr { .. } | ParamValue::Slot { .. } | ParamValue::List(_) => None,
        }
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

    /// The effective row index after scalar broadcasting.
    #[inline]
    fn value_index(&self, row_idx: usize) -> usize {
        if self.broadcast {
            0
        } else {
            row_idx
        }
    }

    fn null_err(&self, row_idx: usize) -> PolarsError {
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
    pub fn get_i64(&self, row_idx: usize) -> PolarsResult<i64> {
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
            TypedCol::Other(s) => return s.get(idx)?.try_extract::<i64>(),
        };
        value.ok_or_else(|| self.null_err(row_idx))
    }

    /// Read the value at `row_idx` as f64.
    pub fn get_f64(&self, row_idx: usize) -> PolarsResult<f64> {
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
            TypedCol::Other(s) => return s.get(idx)?.try_extract::<f64>(),
        };
        value.ok_or_else(|| self.null_err(row_idx))
    }

    /// Read the value at `row_idx` as a string, for per-row enum parameters.
    ///
    /// Only a genuine string column is accepted: a numeric column here means
    /// the user routed the wrong expression to an enum parameter, which must
    /// be an error rather than a silent fallback to the default.
    pub fn get_str(&self, row_idx: usize) -> PolarsResult<&'a str> {
        let idx = self.value_index(row_idx);
        match &self.typed {
            TypedCol::Str(ca) => ca.get(idx).ok_or_else(|| self.null_err(row_idx)),
            _ => Err(polars_err!(ComputeError:
                "Parameter column '{}' must be a String column for an enum \
                 parameter, got {}",
                self.series.name(), self.series.dtype()
            )),
        }
    }

    /// Read the value at `row_idx` as a boolean, for per-row flag parameters.
    pub fn get_bool(&self, row_idx: usize) -> PolarsResult<bool> {
        let idx = self.value_index(row_idx);
        match &self.typed {
            TypedCol::Bool(ca) => ca.get(idx).ok_or_else(|| self.null_err(row_idx)),
            _ => Err(polars_err!(ComputeError:
                "Parameter column '{}' must be a Boolean column for a flag \
                 parameter, got {}",
                self.series.name(), self.series.dtype()
            )),
        }
    }

    /// Read the value at `row_idx` as an `AnyValue` (for non-numeric
    /// parameter columns such as contour structs).
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
}

impl<'a> ParamCtx<'a> {
    /// Build a context over every plugin input series.
    ///
    /// Source columns are included (slots never point at them, but absolute
    /// indexing keeps the binding trivial and collision-free).
    pub fn from_inputs(inputs: &'a [Series]) -> Self {
        ParamCtx {
            cols: inputs.iter().map(ParamCol::new).collect(),
            probe: false,
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
        }
    }

    /// Whether this is a plan-time shape probe rather than real execution.
    pub fn is_probe(&self) -> bool {
        self.probe
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

/// Shared accessors for optional and enum-valued operation parameters.
///
/// These implement the **single parameter failure policy** for `resolve_op`:
/// an *absent* optional parameter takes its documented default, while a
/// parameter that is *present but invalid* — unknown enum string, wrong type,
/// out-of-range value, or a per-row expression that fails to resolve — is
/// always an error. Helpers never swallow a resolution error into a default
/// (guarded by `execute::strict_param_tests`).
pub mod get {
    use super::{ParamCtx, ParamValue};
    use polars::prelude::*;
    use std::collections::HashMap;

    type Params = HashMap<String, ParamValue>;

    fn named(name: &str, e: PolarsError) -> PolarsError {
        polars_err!(ComputeError: "parameter '{}': {}", name, e)
    }

    /// Optional boolean. Booleans are structural: only a literal
    /// `true`/`false` is accepted — strings, numbers, and expressions error
    /// instead of silently reading as `false`.
    pub fn opt_bool(params: &Params, name: &str, default: bool) -> PolarsResult<bool> {
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

    /// Optional u32 with a default for absence.
    pub fn opt_u32(
        params: &Params,
        name: &str,
        default: u32,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<u32> {
        params
            .get(name)
            .map(|p| p.resolve_u32(row_idx, ctx).map_err(|e| named(name, e)))
            .transpose()
            .map(|v| v.unwrap_or(default))
    }

    /// Optional u32 for a **structural** parameter that fixes the output
    /// shape/length (e.g. `perceptual_hash(hash_size)`). Literal-only: a bound
    /// expression `Slot` (or any non-literal form) errors, because letting the
    /// value vary per row would desync the plan-time schema from the data.
    pub fn opt_u32_literal(params: &Params, name: &str, default: u32) -> PolarsResult<u32> {
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
    pub fn maybe_usize_literal(params: &Params, name: &str) -> PolarsResult<Option<usize>> {
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
        params: &Params,
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
        params: &Params,
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
        params: &Params,
        name: &str,
        default: u8,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<u8> {
        match params.get(name) {
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
        params: &Params,
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
        params: &Params,
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
        params: &Params,
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
        params: &Params,
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
    /// dtype effect (`apply_mask(invert)`, `rasterize(anti_alias)`). Flags that
    /// *do* change the output shape — `rotate(expand)` — must keep using
    /// [`opt_bool`], which is literal-only.
    pub fn opt_bool_dyn(
        params: &Params,
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
    fn test_int_list() {
        let param = ParamValue::Literal {
            value: serde_json::json!([0, 2, 1]),
        };
        assert_eq!(param.as_int_list().unwrap(), vec![0, 2, 1]);
    }

    #[test]
    fn test_slot_typed_read() {
        let s = Series::new("h".into(), &[10i64, 20, 30]);
        let inputs = vec![s];
        let ctx = ParamCtx::from_inputs(&inputs);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(2, &ctx).unwrap(), 30);
        assert_eq!(param.resolve_u32(1, &ctx).unwrap(), 20);
    }

    #[test]
    fn test_slot_broadcast_scalar() {
        // A one-element series (aggregation result) broadcasts to all rows.
        let s = Series::new("h".into(), &[7i32]);
        let inputs = vec![s];
        let ctx = ParamCtx::from_inputs(&inputs);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(0, &ctx).unwrap(), 7);
        assert_eq!(param.resolve_i64(99, &ctx).unwrap(), 7);
    }

    #[test]
    fn test_slot_null_value_errors() {
        let s = Series::new("h".into(), &[Some(1i64), None]);
        let inputs = vec![s];
        let ctx = ParamCtx::from_inputs(&inputs);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(0, &ctx).unwrap(), 1);
        assert!(param.resolve_i64(1, &ctx).is_err());
    }

    #[test]
    fn test_slot_float_truncates_like_try_extract() {
        let s = Series::new("h".into(), &[3.9f64, -2.7]);
        let inputs = vec![s];
        let ctx = ParamCtx::from_inputs(&inputs);
        let param = ParamValue::Slot { idx: 0 };
        assert_eq!(param.resolve_i64(0, &ctx).unwrap(), 3);
        assert_eq!(param.resolve_i64(1, &ctx).unwrap(), -2);
    }

    #[test]
    fn test_unbound_expr_is_internal_error() {
        let param = ParamValue::Expr {
            col: Some("h".to_string()),
        };
        let err = param.resolve_i64(0, &ParamCtx::empty()).unwrap_err();
        assert!(err.to_string().contains("unbound expression parameter"));
    }

    #[test]
    fn test_structural_literal_resolvers_reject_bound_slots() {
        use super::get;
        use std::collections::HashMap;

        // A structural param that reached the resolver as a bound expression
        // slot (or the wire `Expr` form) must error — letting it vary per row
        // would desync the plan-time schema from the data.
        for bad in [ParamValue::Slot { idx: 0 }, ParamValue::Expr { col: None }] {
            let mut params: HashMap<String, ParamValue> = HashMap::new();
            params.insert("axis".into(), bad.clone());
            let err = get::maybe_usize_literal(&params, "axis").unwrap_err();
            assert!(err.to_string().contains("structural"), "got: {err}");

            let mut params2: HashMap<String, ParamValue> = HashMap::new();
            params2.insert("hash_size".into(), bad);
            let err = get::opt_u32_literal(&params2, "hash_size", 64).unwrap_err();
            assert!(err.to_string().contains("structural"), "got: {err}");
        }

        // A literal still resolves normally, and absence yields the default.
        let mut lit: HashMap<String, ParamValue> = HashMap::new();
        lit.insert(
            "axis".into(),
            ParamValue::Literal {
                value: serde_json::json!(2),
            },
        );
        assert_eq!(get::maybe_usize_literal(&lit, "axis").unwrap(), Some(2));
        assert_eq!(
            get::opt_u32_literal(&HashMap::new(), "hash_size", 64).unwrap(),
            64
        );
    }

    #[test]
    fn test_expr_wire_form_is_exactly_type_and_col() {
        // The `Expr` wire form carries only `col` — vestigial serialization
        // fields (expr_serialized/expr_str) were removed. Pin the shape so they
        // cannot silently return and desync the Python `to_dict` emitter.
        let json = serde_json::to_value(&ParamValue::Expr {
            col: Some("h".to_string()),
        })
        .unwrap();
        let obj = json.as_object().unwrap();
        assert_eq!(obj.get("type").and_then(|v| v.as_str()), Some("expr"));
        assert_eq!(obj.get("col").and_then(|v| v.as_str()), Some("h"));
        assert_eq!(obj.len(), 2, "expr wire form must be exactly {{type, col}}");
    }

    #[test]
    fn test_slot_and_list_are_not_serialized() {
        // Slot and List are compile-time-only forms produced by graph
        // compilation; they never appear in the wire format and must not
        // serialize into something that round-trips as a resolvable param.
        assert!(serde_json::to_string(&ParamValue::Slot { idx: 3 }).is_err());
        assert!(serde_json::to_string(&ParamValue::List(vec![])).is_err());
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
