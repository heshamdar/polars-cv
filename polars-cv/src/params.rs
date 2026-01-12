//! Parameter value types for expression resolution.
//!
//! This module handles the resolution of parameter values that can be either
//! literals (known at planning time) or expressions (resolved per-row).

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
}

impl ParamValue {
    /// Check if this is a literal value.
    /// Used for optimization when all parameters are known at planning time.
    #[allow(dead_code)]
    pub fn is_literal(&self) -> bool {
        matches!(self, ParamValue::Literal { .. })
    }

    /// Get the column name if this is an expression reference.
    /// Used for tracking which columns need to be passed to the plugin.
    #[allow(dead_code)]
    pub fn column_name(&self) -> Option<&str> {
        match self {
            ParamValue::Expr { col, .. } => col.as_deref(),
            ParamValue::Literal { .. } => None,
        }
    }

    /// Resolve this parameter to a concrete i64 value.
    pub fn resolve_i64(
        &self,
        row_idx: usize,
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<i64> {
        match self {
            ParamValue::Literal { value } => value.as_i64().ok_or_else(
                || polars_err!(ComputeError: "Expected integer literal, got {:?}", value),
            ),
            ParamValue::Expr { col, .. } => {
                let col_name = col.as_deref().ok_or_else(
                    || polars_err!(ComputeError: "Expression parameter missing column name"),
                )?;

                let series = expr_columns.get(col_name).ok_or_else(|| {
                    polars_err!(ComputeError: "Column '{}' not found in expression inputs", col_name)
                })?;

                // Get the value at this row, with scalar broadcasting support.
                // When an expression is an aggregation (like .max()), Polars returns a
                // single-element Series. We broadcast this scalar to all rows, matching
                // Polars' contextual broadcasting behavior.
                let idx = if series.len() == 1 { 0 } else { row_idx };
                let value = series.get(idx)?;
                value.try_extract::<i64>()
            }
        }
    }

    /// Resolve this parameter to a concrete u32 value.
    pub fn resolve_u32(
        &self,
        row_idx: usize,
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<u32> {
        let value = self.resolve_i64(row_idx, expr_columns)?;
        if value < 0 || value > u32::MAX as i64 {
            return Err(polars_err!(ComputeError: "Value {} out of range for u32", value));
        }
        Ok(value as u32)
    }

    /// Resolve this parameter to a concrete usize value.
    pub fn resolve_usize(
        &self,
        row_idx: usize,
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<usize> {
        let value = self.resolve_i64(row_idx, expr_columns)?;
        if value < 0 {
            return Err(polars_err!(ComputeError: "Value {} cannot be negative", value));
        }
        Ok(value as usize)
    }

    /// Resolve this parameter to a concrete f64 value.
    pub fn resolve_f64(
        &self,
        row_idx: usize,
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<f64> {
        match self {
            ParamValue::Literal { value } => value.as_f64().ok_or_else(
                || polars_err!(ComputeError: "Expected float literal, got {:?}", value),
            ),
            ParamValue::Expr { col, .. } => {
                let col_name = col.as_deref().ok_or_else(
                    || polars_err!(ComputeError: "Expression parameter missing column name"),
                )?;

                let series = expr_columns.get(col_name).ok_or_else(|| {
                    polars_err!(ComputeError: "Column '{}' not found in expression inputs", col_name)
                })?;

                // Scalar broadcasting: use index 0 for aggregation results (length 1)
                let idx = if series.len() == 1 { 0 } else { row_idx };
                let value = series.get(idx)?;
                value.try_extract::<f64>()
            }
        }
    }

    /// Resolve this parameter to a concrete f32 value.
    pub fn resolve_f32(
        &self,
        row_idx: usize,
        expr_columns: &HashMap<String, &Series>,
    ) -> PolarsResult<f32> {
        self.resolve_f64(row_idx, expr_columns).map(|v| v as f32)
    }

    /// Resolve this parameter to a concrete string value.
    pub fn resolve_string(&self) -> PolarsResult<String> {
        match self {
            ParamValue::Literal { value } => value.as_str().map(|s| s.to_string()).ok_or_else(
                || polars_err!(ComputeError: "Expected string literal, got {:?}", value),
            ),
            ParamValue::Expr { .. } => {
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
            ParamValue::Expr { .. } => {
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
                            .map(|i| i as usize)
                            .ok_or_else(|| polars_err!(ComputeError: "Expected integer in array"))
                    })
                    .collect()
            }
            ParamValue::Expr { .. } => {
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
            ParamValue::Expr { .. } => None,
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
        assert_eq!(param.resolve_i64(0, &HashMap::new()).unwrap(), 42);
    }

    #[test]
    fn test_literal_f64() {
        let param = ParamValue::Literal {
            value: serde_json::json!(1.5),
        };
        assert_eq!(param.resolve_f64(0, &HashMap::new()).unwrap(), 1.5);
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
}
