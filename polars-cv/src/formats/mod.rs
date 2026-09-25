//! Typed sources and sinks: one struct per format.
//!
//! A source or sink crosses the wire as `{"format": <name>, <field>: <value>,
//! ...}`. Each format is a struct deriving [`Op`](polars_cv_macros::Op) — the
//! same derive the op catalogue uses — carrying exactly the fields that format
//! reads, and one line in a [`formats!`] registry. From that:
//!
//! - **deserialization** rejects an unknown format, and a field the chosen
//!   format does not read — naming the formats it *does* apply to, computed
//!   from the definitions;
//! - **the catalogue** ([`io_catalog_json`], committed as
//!   `tests/golden/io_catalog.json`) is what `scripts/gen_ops.py` generates
//!   Python's `SourceFormat`/`SinkFormat` from.
//!
//! This replaces the per-format parameter-applicability tables Python used to
//! keep by hand (typed-op plan P4).

pub mod sink;
pub mod sink_dtype;
pub mod source;

use serde::Serialize;

use crate::ops::OpDesc;

/// Reject a key the format `name` does not read, saying where it does apply.
///
/// `all` is every format's description, `own` the chosen one's field names.
fn check_applies(
    kind: &str,
    name: &str,
    fields: &serde_json::Map<String, serde_json::Value>,
    all: &[OpDesc],
) -> Result<(), String> {
    let own = all
        .iter()
        .find(|d| d.name == name)
        .expect("a registered format has a description");
    for key in fields.keys() {
        if own.fields.iter().any(|f| f.name == key) {
            continue;
        }
        let applies: Vec<&str> = all
            .iter()
            .filter(|d| d.fields.iter().any(|f| f.name == key))
            .map(|d| d.name)
            .collect();
        return Err(if applies.is_empty() {
            let mut known: Vec<&str> = all
                .iter()
                .flat_map(|d| d.fields.iter().map(|f| f.name))
                .collect();
            known.sort_unstable();
            known.dedup();
            format!(
                "'{key}' is not a {kind} parameter (known: {})",
                known.join(", ")
            )
        } else {
            format!(
                "'{key}' does not apply to the '{name}' {kind} (it applies to: {})",
                applies.join(", ")
            )
        });
    }
    Ok(())
}

/// Register the formats of one end of the pipeline: the enum, its wire names,
/// its (de)serialization and its catalogue, from one line per format.
macro_rules! formats {
    (
        $(#[$meta:meta])*
        $enum:ident ($kind:literal) {
            $($wire:literal => $variant:ident($ty:ty) $sample:tt),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq)]
        pub enum $enum {
            $($variant($ty)),+
        }

        impl $enum {
            /// Every format's wire name, sorted.
            pub const NAMES: &'static [&'static str] = &[$($wire),+];

            /// The format's wire name.
            pub fn name(&self) -> &'static str {
                match self {
                    $($enum::$variant(_) => $wire),+
                }
            }

            /// Every format's description, in `NAMES` order.
            pub fn catalog() -> Vec<crate::ops::OpDesc> {
                vec![$(crate::ops::op_desc::<$ty>($wire)),+]
            }

            /// See [`crate::ops::OpFields::visit_slots`].
            #[allow(dead_code)]
            pub fn visit_slots(&self, f: &mut dyn FnMut(&'static str, usize)) {
                use crate::ops::OpFields;
                match self {
                    $($enum::$variant(spec) => spec.visit_slots(f),)+
                }
            }

            fn from_fields(
                name: &str,
                fields: serde_json::Map<String, serde_json::Value>,
            ) -> Result<Self, String> {
                if !Self::NAMES.contains(&name) {
                    return Err(format!(
                        "unknown {} format '{}', expected one of {:?}",
                        $kind, name, Self::NAMES
                    ));
                }
                $crate::formats::check_applies($kind, name, &fields, &Self::catalog())?;
                let fields = serde_json::Value::Object(fields);
                match name {
                    $($wire => serde_path_to_error::deserialize::<_, $ty>(fields)
                        .map($enum::$variant)
                        .map_err(|e| crate::ops::path_error(&e)),)+
                    _ => unreachable!("checked against NAMES above"),
                }
            }

            /// One valid instance of every format, in `NAMES` order.
            #[cfg(test)]
            pub fn samples() -> Vec<Self> {
                vec![$(
                    Self::from_fields($wire, match serde_json::json!($sample) {
                        serde_json::Value::Object(m) => m,
                        _ => unreachable!("a sample is an object"),
                    })
                    .unwrap_or_else(|e| panic!("sample for '{}': {e}", $wire)),
                )+]
            }
        }

        impl serde::Serialize for $enum {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                let mut value = match self {
                    $($enum::$variant(spec) => serde_json::to_value(spec),)+
                }
                .map_err(serde::ser::Error::custom)?;
                value
                    .as_object_mut()
                    .expect("a format struct serializes to a JSON object")
                    .insert("format".into(), self.name().into());
                value.serialize(s)
            }
        }

        impl<'de> serde::Deserialize<'de> for $enum {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                use serde::de::Error;
                let mut fields =
                    serde_json::Map::<String, serde_json::Value>::deserialize(d)?;
                let name = match fields.remove("format") {
                    Some(serde_json::Value::String(name)) => name,
                    _ => {
                        return Err(D::Error::custom(concat!(
                            "a ", $kind, " needs a string \"format\" name"
                        )))
                    }
                };
                Self::from_fields(&name, fields)
                    .map_err(|e| D::Error::custom(format!("{} '{name}': {e}", $kind)))
            }
        }
    };
}
pub(crate) use formats;

/// Both ends' catalogues, as committed in `tests/golden/io_catalog.json`.
#[derive(Serialize)]
struct IoCatalog {
    sources: Vec<OpDesc>,
    sinks: Vec<OpDesc>,
}

/// The source/sink catalogue as committed in `tests/golden/io_catalog.json`.
pub fn io_catalog_json() -> String {
    let catalog = IoCatalog {
        sources: source::Source::catalog(),
        sinks: sink::Sink::catalog(),
    };
    let mut text = serde_json::to_string_pretty(&catalog).expect("the catalogue serializes");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed catalogue is what `scripts/gen_ops.py` generates
    /// `SourceFormat`/`SinkFormat` from. Regenerate with `POLARS_CV_BLESS=1
    /// cargo test -p polars-cv io_catalog_matches`.
    #[test]
    fn io_catalog_matches_the_committed_file() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/io_catalog.json");
        let current = io_catalog_json();
        if std::env::var_os("POLARS_CV_BLESS").is_some() {
            std::fs::write(path, &current).unwrap();
        }
        let committed = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            committed == current,
            "tests/golden/io_catalog.json is stale; regenerate with \
             POLARS_CV_BLESS=1 cargo test -p polars-cv io_catalog_matches, then \
             python scripts/gen_ops.py"
        );
    }
}
