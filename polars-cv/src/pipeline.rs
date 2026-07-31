//! Serde types for the pipeline graph wire format.
//!
//! These are the per-node specification types deserialized from the graph
//! JSON produced by the Python planner: `SourceSpec` (how a node's input is
//! decoded), `SinkSpec` (how an output is encoded) and `OpSpec` (one
//! operation with its parameters). They are consumed by `graph::types`
//! (`GraphNode`/`OutputSpec`) and the executor.

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
    /// Fill value for contour interior (default 255). Per-row capable, matching
    /// the identical parameter on the `rasterize` op.
    #[serde(default)]
    pub fill_value: Option<crate::params::ParamValue>,
    /// Background value for contour exterior (default 0). Per-row capable,
    /// matching the identical parameter on the `rasterize` op.
    #[serde(default)]
    pub background: Option<crate::params::ParamValue>,
    /// Serialized shape pipeline for dimension inference.
    #[serde(default)]
    pub shape_pipeline: Option<serde_json::Value>,
    /// Cloud-storage credentials/options for `file_path` sources
    /// (string key/value pairs matching `cloud::CloudOptions::from_map`).
    #[serde(default)]
    pub cloud_options: Option<HashMap<String, String>>,
    /// Explicit decode-scale assertion for image sources: the pipeline only
    /// needs at least this many pixels on the image's long side. JPEG decode
    /// uses IDCT scaling (1/8, 1/4, 1/2) to skip work; other formats decode
    /// at full size.
    #[serde(default)]
    pub decode_max_size: Option<u32>,
    /// Whether to require contiguous data for list/array sources.
    /// If true and data is jagged, an error is raised.
    #[serde(default)]
    pub require_contiguous: bool,
    /// Error handling for source decoding: "raise" (default) or "null".
    #[serde(default = "default_on_error")]
    pub on_error: String,
}

fn default_on_error() -> String {
    "raise".to_string()
}

impl SourceSpec {
    /// Resolve a contour source's `(fill_value, background)` at `row_idx`.
    ///
    /// Both default when absent (255 / 0) and both may be per-row expressions,
    /// mirroring `resolve_rasterize_style` for the `rasterize` op so the two
    /// spellings of the same operation cannot diverge.
    pub fn resolve_fill(
        &self,
        row_idx: usize,
        ctx: &crate::params::ParamCtx,
    ) -> PolarsResult<(u8, u8)> {
        use crate::params::get::opt_u8_value;
        Ok((
            opt_u8_value(self.fill_value.as_ref(), "fill_value", 255, row_idx, ctx)?,
            opt_u8_value(self.background.as_ref(), "background", 0, row_idx, ctx)?,
        ))
    }
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
    /// Output element dtype for the `numpy`/`torch` sink. Currently only
    /// `"f16"` is meaningful: the buffer (which the engine has no f16 dtype for)
    /// is downcast from float to half at encode time. `None` keeps the buffer's
    /// native dtype. Accepts the wire key `dtype` (the user-facing sink kwarg).
    #[serde(default, alias = "dtype")]
    pub out_dtype: Option<String>,
}

fn default_quality() -> u8 {
    85
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_op_spec_with_literals() {
        let json = r#"{"op": "resize", "height": {"type": "literal", "value": 224}, "width": {"type": "literal", "value": 224}}"#;
        let op: OpSpec = serde_json::from_str(json).unwrap();
        assert_eq!(op.op, "resize");
        assert!(op.is_all_literal());
    }

    #[test]
    fn test_parse_op_spec_with_expression() {
        let json = r#"{"op": "resize", "height": {"type": "expr", "col": "target_h"}, "width": {"type": "literal", "value": 224}}"#;
        let op: OpSpec = serde_json::from_str(json).unwrap();
        assert!(!op.is_all_literal());
    }

    #[test]
    fn test_source_spec_defaults() {
        let source: SourceSpec = serde_json::from_str(r#"{"format": "image_bytes"}"#).unwrap();
        assert_eq!(source.format, "image_bytes");
        // Absent fill/background resolve to their documented defaults.
        let (fill, background) = source
            .resolve_fill(0, &crate::params::ParamCtx::empty())
            .unwrap();
        assert_eq!(fill, 255);
        assert_eq!(background, 0);
        assert_eq!(source.on_error, "raise");
        assert!(source.decode_max_size.is_none());
        assert!(source.cloud_options.is_none());
    }

    #[test]
    fn test_sink_spec_defaults() {
        let sink: SinkSpec = serde_json::from_str(r#"{"format": "jpeg"}"#).unwrap();
        assert_eq!(sink.format, "jpeg");
        assert_eq!(sink.quality, 85);
        assert!(sink.shape.is_none());
    }
}
