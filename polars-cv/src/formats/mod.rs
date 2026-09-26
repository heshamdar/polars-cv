//! Typed sources and sinks: one family per end, defined as the ops are.
//!
//! A source or sink crosses the wire as `{"format": <name>, <field>: <value>,
//! ...}`. Each end is an enum deriving [`Ops`](polars_cv_macros::Ops) — the
//! derive the op families use — whose variants are the formats, each carrying
//! exactly the fields that format reads, with its doc comment, declared
//! defaults and a sample. From that:
//!
//! - **deserialization** rejects an unknown format, and a field the chosen
//!   format does not read — naming the formats it *does* apply to, computed
//!   from the definitions ([`Format`]);
//! - **the catalogue** ([`io_catalog_json`], committed as
//!   `tests/golden/io_catalog.json`) is what `scripts/gen_ops.py` generates
//!   Python's `SourceFormat`/`SinkFormat` from.
//!
//! Neither end has a per-row value (a contour canvas is the `rasterize`
//! op's), so the families have no mode: each is its own wire form.

pub mod sink;
pub mod sink_dtype;
pub mod source;

use serde::Serialize;
use view_buffer::mode::{OpDesc, WireOps};

/// One end of the pipeline: the formats of a [`WireOps`] family, tagged by a
/// `"format"` key on the wire.
pub trait Format: WireOps + 'static {
    /// "source" or "sink", for messages.
    const KIND: &'static str;

    /// Every format's description, built once.
    fn formats() -> &'static [OpDesc];

    /// The format's wire name.
    fn name(&self) -> &'static str {
        self.wire_name().expect("every format has a wire name")
    }
}

/// The wire object of `format`: its fields plus the `"format"` tag.
fn to_wire<F: Format>(format: &F) -> serde_json::Value {
    let mut fields = format.wire_fields().expect("every format has wire fields");
    fields.insert("format".into(), format.name().into());
    serde_json::Value::Object(fields)
}

/// Parse a tagged wire object as a format of `F`.
///
/// A key the chosen format does not read is refused naming the formats it
/// applies to (or, when none does, every known key) — more than the derive's
/// own unknown-field error can say, because it knows the other formats.
fn from_wire<F: Format>(
    mut fields: serde_json::Map<String, serde_json::Value>,
) -> Result<F, String> {
    let kind = F::KIND;
    let name = match fields.remove("format") {
        Some(serde_json::Value::String(name)) => name,
        _ => return Err(format!("a {kind} needs a string \"format\" name")),
    };
    let all = F::formats();
    let Some(own) = all.iter().find(|d| d.name == name) else {
        let names: Vec<&str> = all.iter().map(|d| d.name).collect();
        return Err(format!(
            "unknown {kind} format '{name}', expected one of {names:?}"
        ));
    };
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
                "{kind} '{name}': '{key}' is not a {kind} parameter (known: {})",
                known.join(", ")
            )
        } else {
            format!(
                "{kind} '{name}': '{key}' does not apply to the '{name}' {kind} \
                 (it applies to: {})",
                applies.join(", ")
            )
        });
    }
    F::from_wire(&name, serde_json::Value::Object(fields))
        .expect("a catalogued format parses")
        .map_err(|e| format!("{kind} '{name}': {e}"))
}

/// `Serialize`/`Deserialize` for a [`Format`] family, through [`to_wire`] and
/// [`from_wire`].
macro_rules! tagged_serde {
    ($ty:ty) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(&$crate::formats::to_wire(self), s)
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let fields: serde_json::Map<String, serde_json::Value> =
                    serde::Deserialize::deserialize(d)?;
                $crate::formats::from_wire(fields).map_err(serde::de::Error::custom)
            }
        }
    };
}
pub(crate) use tagged_serde;

/// Both ends' catalogues, as committed in `tests/golden/io_catalog.json`.
#[derive(Serialize)]
struct IoCatalog {
    /// `Pipeline.source()`'s docstring body.
    source_doc: &'static str,
    /// The format `source()` reads when none is named.
    default_source: &'static str,
    sources: Vec<OpDesc>,
    sinks: Vec<OpDesc>,
}

/// The source/sink catalogue as committed in `tests/golden/io_catalog.json`.
pub fn io_catalog_json() -> String {
    let catalog = IoCatalog {
        source_doc: source::Source::FAMILY_DOC,
        default_source: source::Source::DEFAULT_FORMAT,
        sources: <source::Source as Format>::formats().to_vec(),
        sinks: <sink::Sink as Format>::formats().to_vec(),
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
