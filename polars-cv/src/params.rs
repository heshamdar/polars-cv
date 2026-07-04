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
        /// Serialized expression (for complex expressions).
        #[serde(default)]
        expr_serialized: Option<String>,
        /// String representation (fallback).
        #[serde(default)]
        expr_str: Option<String>,
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
}

impl ParamValue {
    /// Check if this is a literal value.
    /// Used for optimization when all parameters are known at planning time.
    pub fn is_literal(&self) -> bool {
        matches!(self, ParamValue::Literal { .. })
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

    /// Resolve this parameter to a concrete string value.
    pub fn resolve_string(&self) -> PolarsResult<&str> {
        match self {
            ParamValue::Literal { value } => value.as_str().ok_or_else(
                || polars_err!(ComputeError: "Expected string literal, got {:?}", value),
            ),
            ParamValue::Expr { .. } | ParamValue::Slot { .. } => {
                Err(polars_err!(ComputeError: "String parameters cannot be expressions"))
            }
        }
    }

    /// Get literal value as a list of ParamValue (for reshape).
    pub fn as_param_list(&self) -> PolarsResult<Vec<ParamValue>> {
        match self {
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
            ParamValue::Expr { .. } | ParamValue::Slot { .. } => {
                Err(polars_err!(ComputeError: "Axes parameters cannot be expressions"))
            }
        }
    }

    /// Get literal value as a Vec<f32> (for normalize preset mean/std).
    pub fn as_f32_vec(&self) -> Option<Vec<f32>> {
        match self {
            ParamValue::Literal { value } => {
                let arr = value.as_array()?;
                arr.iter()
                    .map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Option<Vec<f32>>>()
            }
            ParamValue::Expr { .. } | ParamValue::Slot { .. } => None,
        }
    }

    /// Get literal value as a Vec<f64> (for histogram bins).
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            ParamValue::Literal { value } => {
                let arr = value.as_array()?;
                arr.iter().map(|v| v.as_f64()).collect::<Option<Vec<f64>>>()
            }
            ParamValue::Expr { .. } | ParamValue::Slot { .. } => None,
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
            TypedCol::Other(s) => return s.get(idx)?.try_extract::<f64>(),
        };
        value.ok_or_else(|| self.null_err(row_idx))
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
}

impl<'a> ParamCtx<'a> {
    /// Build a context over every plugin input series.
    ///
    /// Source columns are included (slots never point at them, but absolute
    /// indexing keeps the binding trivial and collision-free).
    pub fn from_inputs(inputs: &'a [Series]) -> Self {
        ParamCtx {
            cols: inputs.iter().map(ParamCol::new).collect(),
        }
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

    /// Optional usize where absence is meaningful (e.g. a reduction `axis`:
    /// absent means "global"). Present-but-invalid still errors.
    pub fn maybe_usize(
        params: &Params,
        name: &str,
        row_idx: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<Option<usize>> {
        params
            .get(name)
            .map(|p| p.resolve_usize(row_idx, ctx).map_err(|e| named(name, e)))
            .transpose()
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

    /// Required enum-valued parameter, parsed against a canonical
    /// `NAMED`-style table (plus parser-only aliases). Unknown values error
    /// with the canonical names listed.
    pub fn req_enum<T: Copy>(
        params: &Params,
        name: &str,
        canonical: &[(&str, T)],
        aliases: &[(&str, T)],
    ) -> PolarsResult<T> {
        let param = params
            .get(name)
            .ok_or_else(|| polars_err!(ComputeError: "Missing required parameter: {}", name))?;
        let s = param.resolve_string().map_err(|e| named(name, e))?;
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
    ) -> PolarsResult<T> {
        if params.contains_key(name) {
            req_enum(params, name, canonical, aliases)
        } else {
            Ok(default)
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
            expr_serialized: None,
            expr_str: None,
        };
        let err = param.resolve_i64(0, &ParamCtx::empty()).unwrap_err();
        assert!(err.to_string().contains("unbound expression parameter"));
    }

    #[test]
    fn test_slot_is_not_serialized() {
        // Slot is a compile-time-only form; serializing it must not produce
        // something that round-trips as a resolvable param.
        let param = ParamValue::Slot { idx: 3 };
        assert!(serde_json::to_string(&param).is_err());
    }
}
