//! Serde types for the pipeline graph wire format.
//!
//! These are the per-node specification types deserialized from the graph
//! JSON produced by the Python planner: `OpSpec`, one operation with its
//! parameters. Sources and sinks are typed per format in `crate::formats`. They are consumed by `graph::types`
//! (`GraphNode`/`OutputSpec`) and the executor.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::execute::LEGACY_OPS;
use crate::ops::TypedOp;
use crate::params::ParamValue;

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
}
