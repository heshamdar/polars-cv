//! Op-time parameter map.
//!
//! Plan-time parameters can be either literals (resolved once when the op is
//! constructed) or expression-bound (resolved per row by the bridge layer
//! before the factory is called). This module owns only the literal half;
//! per-row resolution against Polars `Series` lives in the `polars-cv`
//! bridge so this crate stays Polars-free.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A literal parameter value, JSON-shaped for ergonomic op-side parsing.
pub type LiteralValue = serde_json::Value;

/// Map of parameter name → literal value.
///
/// Insertion order is preserved so error messages and debug dumps are stable.
pub type ParamMap = IndexMap<String, LiteralValue>;

/// Errors raised when an op's factory rejects its parameter map.
#[derive(Debug, Error)]
pub enum ParamError {
    #[error("missing required parameter `{name}` for op `{op}`")]
    Missing {
        op: &'static str,
        name: &'static str,
    },

    #[error("parameter `{name}` for op `{op}`: expected {expected}, got {got}")]
    WrongType {
        op: &'static str,
        name: &'static str,
        expected: &'static str,
        got: String,
    },

    #[error("parameter `{name}` for op `{op}` out of range: {message}")]
    OutOfRange {
        op: &'static str,
        name: &'static str,
        message: String,
    },
}

/// Pull a required `i64` from the param map.
pub fn require_i64(
    params: &ParamMap,
    op: &'static str,
    name: &'static str,
) -> Result<i64, ParamError> {
    let v = params
        .get(name)
        .ok_or(ParamError::Missing { op, name })?;
    v.as_i64().ok_or_else(|| ParamError::WrongType {
        op,
        name,
        expected: "int",
        got: type_name_of(v),
    })
}

/// Pull a required `f64` from the param map.
pub fn require_f64(
    params: &ParamMap,
    op: &'static str,
    name: &'static str,
) -> Result<f64, ParamError> {
    let v = params
        .get(name)
        .ok_or(ParamError::Missing { op, name })?;
    v.as_f64().ok_or_else(|| ParamError::WrongType {
        op,
        name,
        expected: "float",
        got: type_name_of(v),
    })
}

/// Pull a required `String` from the param map.
pub fn require_str<'a>(
    params: &'a ParamMap,
    op: &'static str,
    name: &'static str,
) -> Result<&'a str, ParamError> {
    let v = params
        .get(name)
        .ok_or(ParamError::Missing { op, name })?;
    v.as_str().ok_or_else(|| ParamError::WrongType {
        op,
        name,
        expected: "string",
        got: type_name_of(v),
    })
}

/// Pull an optional `bool` with a default.
pub fn opt_bool(params: &ParamMap, name: &str, default: bool) -> bool {
    params
        .get(name)
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

fn type_name_of(v: &LiteralValue) -> String {
    match v {
        LiteralValue::Null => "null".into(),
        LiteralValue::Bool(_) => "bool".into(),
        LiteralValue::Number(_) => "number".into(),
        LiteralValue::String(_) => "string".into(),
        LiteralValue::Array(_) => "array".into(),
        LiteralValue::Object(_) => "object".into(),
    }
}

/// Convenience for tests / construction in this crate.
#[cfg(any(test, doctest))]
pub fn empty_params() -> ParamMap {
    IndexMap::new()
}

/// Wire-format wrapper distinguishing literal vs expression-bound params.
///
/// The bridge layer in `polars-cv` materializes [`ParamValue::Expr`] entries
/// to literals before invoking an op's factory; `pcv-graph` itself only ever
/// sees [`ParamValue::Literal`]. Kept here so the IR and wire format can
/// reference a single canonical type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ParamValue {
    #[serde(rename = "literal")]
    Literal { value: LiteralValue },

    #[serde(rename = "expr")]
    Expr {
        #[serde(default)]
        col: Option<String>,
        #[serde(default)]
        expr_str: Option<String>,
    },
}

impl ParamValue {
    pub fn lit(v: impl Into<LiteralValue>) -> Self {
        ParamValue::Literal { value: v.into() }
    }

    pub fn col(name: impl Into<String>) -> Self {
        ParamValue::Expr {
            col: Some(name.into()),
            expr_str: None,
        }
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, ParamValue::Literal { .. })
    }

    pub fn column_name(&self) -> Option<&str> {
        match self {
            ParamValue::Expr { col, .. } => col.as_deref(),
            ParamValue::Literal { .. } => None,
        }
    }
}
