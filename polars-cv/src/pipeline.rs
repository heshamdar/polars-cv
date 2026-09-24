//! Serde types for the pipeline graph wire format.
//!
//! These are the per-node specification types deserialized from the graph
//! JSON produced by the Python planner: `SourceSpec` (how a node's input is
//! decoded) and `OpSpec` (one operation with its parameters). Sinks are typed
//! per format in `crate::formats::sink`. They are consumed by `graph::types`
//! (`GraphNode`/`OutputSpec`) and the executor.

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::execute::LEGACY_OPS;
use crate::ops::TypedOp;
use crate::params::ParamValue;

/// Source format specification.
///
/// `deny_unknown_fields` closes this end of the wire format, as `GraphNode`
/// and the typed sinks do for theirs. It is needed *per struct*: serde's attribute
/// does not descend into nested types, so closing `GraphNode` left everything
/// it holds — this included — accepting anything Python sent. That mattered
/// most here, because `allowed_roots` is the path sandbox: a misspelled key
/// deserialized to `None`, i.e. no sandbox at all, silently.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Contour source only: the graph node whose buffer fixes the canvas.
    #[serde(default)]
    pub shape_node: Option<String>,
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
    /// Locations this source's path column may read from. Empty/absent means
    /// unrestricted, which is the default; see `crate::fetch::PathPolicy`.
    #[serde(default)]
    pub allowed_roots: Option<Vec<String>>,
}

fn default_on_error() -> String {
    "raise".to_string()
}

impl SourceSpec {
    /// Resolve a contour source's `(fill_value, background)` at `row_idx`.
    ///
    /// Both default when absent (255 / 0) and both may be per-row expressions,
    /// as the `rasterize` op's `Param<u8>` fields are (`ops::geometry`), so the
    /// two spellings of the same operation cannot diverge.
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

/// A single operation in the pipeline.
///
/// The wire form is one object: `{"op": <name>, <field>: <value>, ...}`. Its
/// deserializer is the migration's one dispatcher (typed-op plan P2–P6): a
/// name in [`TypedOp::NAMES`] deserializes *strictly* into its typed struct,
/// a name in [`LEGACY_OPS`] takes the untyped legacy path, and any other name
/// is an error. There is no fallback between the two, so a typed op's
/// misspelled field is rejected rather than read as a legacy param.
#[derive(Debug, Clone)]
pub enum OpSpec {
    /// An op in the typed catalogue (`crate::ops`).
    Typed(TypedOp),
    /// An op not yet migrated: a name plus an untyped param map.
    Legacy(LegacyOpSpec),
}

/// A not-yet-typed operation: its name and untyped parameters.
///
/// Built only by [`OpSpec`]'s deserializer, which reads each parameter through
/// `ParamValue::from_wire`, so it has no `Deserialize` of its own (its
/// `#[serde(flatten)]` map could not refuse an unknown key; resolution's
/// read-tracking does that for a legacy op).
#[derive(Debug, Clone, Serialize)]
pub struct LegacyOpSpec {
    /// Operation name.
    pub op: String,
    /// Operation parameters (flattened into the struct).
    #[serde(flatten)]
    pub params: HashMap<String, ParamValue>,
}

impl OpSpec {
    /// The op's wire name.
    #[cfg(test)]
    pub fn name(&self) -> &str {
        match self {
            OpSpec::Typed(op) => op.name(),
            OpSpec::Legacy(spec) => &spec.op,
        }
    }

    /// Whether no parameter is per-row, so the op resolves once per graph.
    pub fn is_static(&self) -> bool {
        match self {
            OpSpec::Typed(op) => op.is_static(),
            OpSpec::Legacy(spec) => spec.params.values().all(|p| p.is_literal()),
        }
    }
}

impl Serialize for OpSpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            OpSpec::Typed(op) => {
                let mut value = op.fields_json();
                value
                    .as_object_mut()
                    .expect("an op struct serializes to a JSON object")
                    .insert("op".into(), op.name().into());
                value.serialize(s)
            }
            OpSpec::Legacy(spec) => spec.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for OpSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        // Both name lists are sorted (pinned by `typed_names_are_sorted_and_unique`
        // and `known_ops_sorted_and_unique`), so the lookups are binary searches.
        // One parse into a map; each path then consumes it without a second
        // copy (this runs several times per builder append, via the planning
        // FFIs, so a clone of the tree is measurable).
        let mut fields = serde_json::Map::<String, serde_json::Value>::deserialize(d)?;
        let name = match fields.remove("op") {
            Some(serde_json::Value::String(name)) => name,
            _ => return Err(D::Error::custom("an operation needs a string \"op\" name")),
        };
        if TypedOp::NAMES.binary_search(&name.as_str()).is_ok() {
            return TypedOp::from_fields(&name, serde_json::Value::Object(fields))
                .expect("a name in NAMES is registered")
                .map(OpSpec::Typed)
                .map_err(|e| D::Error::custom(format!("operation '{name}': {e}")));
        }
        if LEGACY_OPS.binary_search(&name.as_str()).is_ok() {
            let params = fields
                .into_iter()
                .map(|(key, value)| {
                    ParamValue::from_wire(&value)
                        .map(|p| (key, p))
                        .map_err(D::Error::custom)
                })
                .collect::<Result<_, _>>()?;
            return Ok(OpSpec::Legacy(LegacyOpSpec { op: name, params }));
        }
        Err(D::Error::custom(format!("Unknown operation: '{name}'")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_op_spec_with_literals() {
        let json = r#"{"op": "resize", "height": 224, "width": 224, "filter": "bilinear"}"#;
        let op: OpSpec = serde_json::from_str(json).unwrap();
        assert_eq!(op.name(), "resize");
        assert!(op.is_static());
    }

    #[test]
    fn test_parse_op_spec_with_expression() {
        let json =
            r#"{"op": "resize", "height": {"$slot": 1}, "width": 224, "filter": "bilinear"}"#;
        let op: OpSpec = serde_json::from_str(json).unwrap();
        assert!(!op.is_static());
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
}
