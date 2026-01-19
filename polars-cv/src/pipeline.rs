//! Pipeline specification and parsing.
//!
//! This module defines the PipelineSpec structure that represents
//! a serialized vision pipeline from Python.

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::params::ParamValue;

/// Source format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpec {
    /// The format of the input data.
    pub format: String,
    /// Data type for "raw" format.
    #[serde(default)]
    pub dtype: Option<String>,
    /// Width for contour rasterization.
    #[serde(default)]
    pub width: Option<crate::params::ParamValue>,
    /// Height for contour rasterization.
    #[serde(default)]
    pub height: Option<crate::params::ParamValue>,
    /// Fill value for contour interior (default 255).
    #[serde(default = "default_fill_value")]
    pub fill_value: u8,
    /// Background value for contour exterior (default 0).
    #[serde(default)]
    pub background: u8,
    /// Serialized shape pipeline for dimension inference.
    #[serde(default)]
    pub shape_pipeline: Option<serde_json::Value>,
    /// Whether to require contiguous data for list/array sources.
    /// If true and data is jagged, an error is raised.
    #[serde(default)]
    pub require_contiguous: bool,
}

fn default_fill_value() -> u8 {
    255
}

/// Sink format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkSpec {
    /// The format of the output data.
    pub format: String,
    /// JPEG quality (for jpeg format).
    #[serde(default = "default_quality")]
    pub quality: u8,
    /// Output shape (for array format).
    #[serde(default)]
    pub shape: Option<Vec<usize>>,
}

fn default_quality() -> u8 {
    85
}

/// Shape hints for pipeline planning.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShapeHints {
    #[serde(default)]
    pub height: Option<ParamValue>,
    #[serde(default)]
    pub width: Option<ParamValue>,
    #[serde(default)]
    pub channels: Option<ParamValue>,
    #[serde(default)]
    pub batch: Option<ParamValue>,
}

/// A single operation in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpSpec {
    /// Operation name.
    pub op: String,
    /// Operation parameters (flattened into the struct).
    #[serde(flatten)]
    pub params: HashMap<String, ParamValue>,
}

impl OpSpec {
    /// Check if all parameters in this op are literals (no expressions).
    ///
    /// Used for per-node precompilation optimization: when all params are
    /// literals, the ViewDto can be resolved once and reused for all rows.
    pub fn is_all_literal(&self) -> bool {
        self.params.values().all(|p| p.is_literal())
    }
}

/// Complete pipeline specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpec {
    /// Input source specification.
    pub source: SourceSpec,
    /// Optional shape hints.
    #[serde(default)]
    pub shape_hints: Option<ShapeHints>,
    /// List of operations to apply.
    pub ops: Vec<OpSpec>,
    /// Output sink specification.
    pub sink: SinkSpec,
}

impl PipelineSpec {
    /// Determine the output Polars dtype based on the sink format.
    /// Used for dynamic output type inference based on sink format.
    #[allow(dead_code)]
    pub fn output_dtype(&self) -> PolarsResult<DataType> {
        match self.sink.format.as_str() {
            "numpy" | "torch" | "blob" | "png" | "jpeg" | "webp" => Ok(DataType::Binary),
            "list" => Ok(DataType::List(Box::new(DataType::List(Box::new(
                DataType::List(Box::new(DataType::UInt8)),
            ))))),
            "array" => {
                // For array format, we need the shape to determine the type
                let shape =
                    self.sink.shape.as_ref().ok_or_else(
                        || polars_err!(ComputeError: "Array sink format requires shape"),
                    )?;

                // Build nested Array type from innermost to outermost
                // e.g., shape [H, W, C] -> Array[H, Array[W, Array[C, UInt8]]]
                let mut dtype = DataType::UInt8;
                for &dim in shape.iter().rev() {
                    dtype = DataType::Array(Box::new(dtype), dim);
                }
                Ok(dtype)
            }
            other => Err(polars_err!(ComputeError: "Unknown sink format: {}", other)),
        }
    }

    /// Check if all parameters in this pipeline are literals (no expressions).
    /// Used for optimization when we can pre-compute all parameters.
    #[allow(dead_code)]
    pub fn is_all_literals(&self) -> bool {
        // Check shape hints
        if let Some(hints) = &self.shape_hints {
            if hints.height.as_ref().is_some_and(|p| !p.is_literal())
                || hints.width.as_ref().is_some_and(|p| !p.is_literal())
                || hints.channels.as_ref().is_some_and(|p| !p.is_literal())
                || hints.batch.as_ref().is_some_and(|p| !p.is_literal())
            {
                return false;
            }
        }

        // Check ops
        self.ops
            .iter()
            .all(|op| op.params.values().all(|p| p.is_literal()))
    }

    /// Get the source format.
    pub fn source_format(&self) -> &str {
        &self.source.format
    }

    /// Get the sink format.
    pub fn sink_format(&self) -> &str {
        &self.sink.format
    }

    /// Validate the pipeline for known issues.
    ///
    /// Returns a list of warnings for operations that may have unexpected behavior.
    ///
    /// Note: As of the dtype promotion system, operations like normalize, scale,
    /// clamp, and relu now accept any numeric input dtype and automatically
    /// handle type promotion. No warnings are generated for dtype requirements.
    ///
    /// This method is intended to be exposed to Python for user feedback,
    /// hence the allow(dead_code) until Python bindings are added.
    ///
    /// Future warnings could include:
    /// - Potential precision loss (e.g., f64 -> f32 pipeline)
    /// - Performance hints (e.g., redundant casts)
    /// - Shape compatibility issues
    #[allow(dead_code)]
    pub fn validate_warnings(&self) -> Vec<String> {
        // With the dtype promotion system, operations automatically handle
        // type casting internally. No dtype-related warnings are needed.
        Vec::new()
    }

    /// Check if the pipeline uses any operations that are known to have restrictions.
    ///
    /// Returns Ok(()) if the pipeline is valid, or an error describing the issue.
    #[allow(dead_code)]
    pub fn validate(&self) -> PolarsResult<()> {
        // Validate contour source parameters
        if self.source.format == "contour" {
            let has_explicit_dims = self.source.width.is_some() || self.source.height.is_some();
            let has_shape_pipeline = self.source.shape_pipeline.is_some();

            if has_explicit_dims && has_shape_pipeline {
                return Err(
                    polars_err!(ComputeError: "Contour source: cannot specify both 'shape' pipeline and explicit dimensions"),
                );
            }

            if !has_explicit_dims && !has_shape_pipeline {
                return Err(
                    polars_err!(ComputeError: "Contour source requires either explicit width/height or a shape pipeline"),
                );
            }

            if has_explicit_dims && (self.source.width.is_none() || self.source.height.is_none()) {
                return Err(
                    polars_err!(ComputeError: "Contour source: both 'width' and 'height' must be specified together"),
                );
            }
        }

        // Check for unsupported operation sequences
        for op in &self.ops {
            // Validate that all required parameters are present
            match op.op.as_str() {
                "resize" => {
                    if !op.params.contains_key("height") || !op.params.contains_key("width") {
                        return Err(
                            polars_err!(ComputeError: "resize operation requires 'height' and 'width' parameters"),
                        );
                    }
                }
                "crop" => {
                    if !op.params.contains_key("top") || !op.params.contains_key("left") {
                        return Err(
                            polars_err!(ComputeError: "crop operation requires 'top' and 'left' parameters"),
                        );
                    }
                }
                "normalize" => {
                    if !op.params.contains_key("method") {
                        return Err(
                            polars_err!(ComputeError: "normalize operation requires 'method' parameter"),
                        );
                    }
                }
                "blur" => {
                    if !op.params.contains_key("sigma") {
                        return Err(
                            polars_err!(ComputeError: "blur operation requires 'sigma' parameter"),
                        );
                    }
                }
                "threshold" => {
                    if !op.params.contains_key("value") {
                        return Err(
                            polars_err!(ComputeError: "threshold operation requires 'value' parameter"),
                        );
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_pipeline() {
        let json = r#"{
            "source": {"format": "image_bytes"},
            "ops": [
                {"op": "resize", "height": {"type": "literal", "value": 224}, "width": {"type": "literal", "value": 224}, "filter": {"type": "literal", "value": "lanczos3"}},
                {"op": "grayscale"}
            ],
            "sink": {"format": "numpy"}
        }"#;

        let pipeline: PipelineSpec = serde_json::from_str(json).unwrap();
        assert_eq!(pipeline.source.format, "image_bytes");
        assert_eq!(pipeline.ops.len(), 2);
        assert_eq!(pipeline.sink.format, "numpy");
        assert!(pipeline.is_all_literals());
    }

    #[test]
    fn test_parse_pipeline_with_expressions() {
        let json = r#"{
            "source": {"format": "image_bytes"},
            "ops": [
                {"op": "resize", "height": {"type": "expr", "col": "target_h"}, "width": {"type": "literal", "value": 224}, "filter": {"type": "literal", "value": "lanczos3"}}
            ],
            "sink": {"format": "numpy"}
        }"#;

        let pipeline: PipelineSpec = serde_json::from_str(json).unwrap();
        assert!(!pipeline.is_all_literals());
    }

    #[test]
    fn test_output_dtype_binary() {
        let json = r#"{
            "source": {"format": "image_bytes"},
            "ops": [],
            "sink": {"format": "numpy"}
        }"#;

        let pipeline: PipelineSpec = serde_json::from_str(json).unwrap();
        assert_eq!(pipeline.output_dtype().unwrap(), DataType::Binary);
    }

    #[test]
    fn test_output_dtype_array() {
        let json = r#"{
            "source": {"format": "image_bytes"},
            "ops": [],
            "sink": {"format": "array", "shape": [224, 224, 3]}
        }"#;

        let pipeline: PipelineSpec = serde_json::from_str(json).unwrap();
        let dtype = pipeline.output_dtype().unwrap();

        // Should be Array[224, Array[224, Array[3, UInt8]]]
        match dtype {
            DataType::Array(inner, 224) => match *inner {
                DataType::Array(inner2, 224) => match *inner2 {
                    DataType::Array(inner3, 3) => {
                        assert_eq!(*inner3, DataType::UInt8);
                    }
                    _ => panic!("Expected Array[3, UInt8]"),
                },
                _ => panic!("Expected Array[224, ...]"),
            },
            _ => panic!("Expected Array[224, ...]"),
        }
    }
}
