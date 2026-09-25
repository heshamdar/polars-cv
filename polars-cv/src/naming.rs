//! This crate's own enum vocabularies, and the registry that surfaces them.
//!
//! Most user-facing enums belong to view-buffer and live in
//! [`view_buffer::naming::REGISTRY`]. A few describe things the engine has no
//! concept of — how a graph handles a failing row, what a null parameter means,
//! how a path read reports an unreadable file — so they are declared here.
//!
//! They are declared *the same way*, through the exported `named_variants!` and
//! `registry!` macros, and [`PLUGIN_REGISTRY`] is chained onto the engine's by
//! [`enum_catalog_json`]. That chaining is the whole point: registering an enum
//! is what generates its Python class (`enum_catalog` → `scripts/gen_ops.py`).
//!
//! Adding an enum here is therefore the same act as surfacing it. Do not
//! hand-write a Python class for one instead.

use view_buffer::naming::registry;

registry!(
    PLUGIN_REGISTRY:
    crate::graph::RowErrorPolicy,
    crate::params::NullParamPolicy,
    crate::fetch::FetchErrorPolicy,
    crate::formats::sink_dtype::SinkDType,
    crate::passes::LogicalPass,
);

/// One registered enum, as the enum catalogue describes it.
#[derive(serde::Serialize)]
struct EnumDesc {
    name: &'static str,
    doc: &'static str,
    variants: Vec<&'static str>,
}

/// Every registered enum — the engine's and this crate's — with its doc and
/// variant spellings, sorted by name, as committed in
/// `tests/golden/enum_catalog.json`. `scripts/gen_ops.py` generates the
/// Python enum classes from it.
pub fn enum_catalog_json() -> String {
    let mut enums: Vec<EnumDesc> = view_buffer::naming::REGISTRY
        .iter()
        .chain(PLUGIN_REGISTRY)
        .map(|(name, variants, doc)| EnumDesc {
            name,
            doc,
            variants: variants(),
        })
        .collect();
    enums.sort_by_key(|e| e.name);
    let mut text = serde_json::to_string_pretty(&enums).expect("the catalogue serializes");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    /// The committed enum catalogue is what `scripts/gen_ops.py` generates the
    /// Python enums from. Regenerate with `POLARS_CV_BLESS=1 cargo test -p
    /// polars-cv catalog_matches`.
    #[test]
    fn enum_catalog_matches_the_committed_file() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/enum_catalog.json"
        );
        let current = super::enum_catalog_json();
        if std::env::var_os("POLARS_CV_BLESS").is_some() {
            std::fs::write(path, &current).unwrap();
        }
        let committed = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            committed == current,
            "tests/golden/enum_catalog.json is stale; regenerate with \
             POLARS_CV_BLESS=1 cargo test -p polars-cv catalog_matches, then \
             python scripts/gen_ops.py"
        );
    }

    use super::PLUGIN_REGISTRY;

    /// Names within an enum must be unique, and an enum must have some.
    ///
    /// The engine's registry has the same test over its own entries. Neither
    /// covers the other, because neither can see the other's list — so this is
    /// a deliberate second copy of a *test*, not of a fact.
    #[test]
    fn plugin_enums_have_unique_names() {
        assert!(
            !PLUGIN_REGISTRY.is_empty(),
            "PLUGIN_REGISTRY is empty; this test would check nothing"
        );
        for (enum_name, variants, _) in PLUGIN_REGISTRY {
            let names = variants();
            assert!(!names.is_empty(), "{enum_name}: no variants");
            for (i, name) in names.iter().enumerate() {
                assert!(!name.is_empty(), "{enum_name}: empty name");
                assert!(
                    names.iter().skip(i + 1).all(|n| n != name),
                    "{enum_name}: duplicate name '{name}'"
                );
            }
        }
    }

    /// No plugin enum may shadow an engine enum.
    ///
    /// The enum catalogue would otherwise carry two classes of one name, and
    /// the generated module would silently keep only the later one.
    #[test]
    fn plugin_enums_do_not_shadow_engine_enums() {
        let clashes: Vec<_> = PLUGIN_REGISTRY
            .iter()
            .map(|(n, _, _)| *n)
            .filter(|n| view_buffer::naming::REGISTRY.iter().any(|(e, _, _)| e == n))
            .collect();
        assert!(
            clashes.is_empty(),
            "these plugin enums share a name with an engine enum, so \
             the catalogue would carry two classes of one name: {clashes:?}"
        );
    }

    /// Every `named_variants!` enum in **this** crate, found by scanning `src/`.
    ///
    /// The twin of view-buffer's `declared_enums`, and it exists because that
    /// one cannot see this crate: it walks `CARGO_MANIFEST_DIR/src`, which is
    /// view-buffer's. Exporting the macro gave this crate the ability to
    /// declare vocabularies without giving it the check that makes declaring
    /// one the same act as being checked — so a `named_variants!` table here
    /// that nobody added to `registry!` was invisible to the enum catalogue its
    /// Python class is generated from.
    fn declared_enums() -> std::collections::BTreeSet<String> {
        fn walk(dir: &std::path::Path, out: &mut std::collections::BTreeSet<String>) {
            for entry in std::fs::read_dir(dir).expect("src/ is readable") {
                let path = entry.expect("readable dir entry").path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let source = std::fs::read_to_string(&path).expect("readable .rs file");
                    for line in source.lines() {
                        let trimmed = line.trim_start();
                        if trimmed.starts_with("//") || trimmed.starts_with("macro_rules!") {
                            continue;
                        }
                        let Some(rest) = line.split_once("named_variants!(") else {
                            continue;
                        };
                        let ty: String = rest
                            .1
                            .chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect();
                        if !ty.is_empty() {
                            out.insert(ty);
                        }
                    }
                }
            }
        }
        let mut found = std::collections::BTreeSet::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut found,
        );
        found
    }

    /// A `NAMED` table in this crate that is not registered must fail here.
    ///
    /// Watched failing: a `ReviewProbePolicy` added to `fetch.rs` and left out
    /// of `registry!` passed all 128 tests before this existed.
    #[test]
    fn every_plugin_named_enum_is_registered() {
        let declared = declared_enums();
        assert!(
            !declared.is_empty(),
            "the scan found no `named_variants!` invocations in polars-cv/src, \
             which means it is broken rather than that there are none — this \
             crate declares RowErrorPolicy, NullParamPolicy and FetchErrorPolicy"
        );

        let registered: std::collections::BTreeSet<String> = PLUGIN_REGISTRY
            .iter()
            .map(|(n, _, _)| n.to_string())
            .collect();

        let unregistered: Vec<_> = declared.difference(&registered).cloned().collect();
        assert!(
            unregistered.is_empty(),
            "these enums declare a named_variants! table in polars-cv/src but \
             are not in PLUGIN_REGISTRY: {unregistered:?}. Add each to the \
             registry! invocation in src/naming.rs — that is what puts it in \
             the enum catalogue and so generates its Python class."
        );

        let missing: Vec<_> = registered.difference(&declared).cloned().collect();
        assert!(
            missing.is_empty(),
            "these names are in PLUGIN_REGISTRY but no named_variants! \
             invocation was found for them: {missing:?}. Either the enum moved \
             (update the registry) or the scan above has rotted."
        );
    }
}
