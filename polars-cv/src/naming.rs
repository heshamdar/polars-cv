//! This crate's own enum vocabularies, and the registry that surfaces them.
//!
//! Most user-facing enums belong to view-buffer and live in
//! [`view_buffer::naming::REGISTRY`]. A few describe things the engine has no
//! concept of — how a graph handles a failing row, what a null parameter means,
//! how a path read reports an unreadable file — so they are declared here.
//!
//! They are declared *the same way*, through the exported `named_variants!` and
//! `registry!` macros, and [`PLUGIN_REGISTRY`] is chained onto the engine's by
//! `enum_variants`/`enum_names` in `lib.rs`. That chaining is the whole point:
//! registering an enum is what surfaces it to Python **and** what makes
//! `test_every_rust_enum_is_parity_checked` demand a Python mirror for it. The
//! alternative — a hand-written match arm per enum in the FFI — is what
//! `BinaryOp` used to need, and it came with a by-name exemption from that very
//! test.
//!
//! Adding an enum here is therefore the same act as getting it checked. Do not
//! add an arm to `enum_variants` instead.

use view_buffer::naming::registry;

registry!(
    PLUGIN_REGISTRY:
    crate::graph::RowErrorPolicy,
    crate::params::NullParamPolicy,
    crate::fetch::FetchErrorPolicy,
);

/// Look up a plugin-owned enum's variant names.
pub(crate) fn registered_variants(name: &str) -> Option<Vec<&'static str>> {
    PLUGIN_REGISTRY
        .iter()
        .find_map(|(n, f)| (*n == name).then(f))
}

/// The names of every plugin-owned enum.
pub(crate) fn registered_names() -> Vec<&'static str> {
    PLUGIN_REGISTRY.iter().map(|(n, _)| *n).collect()
}

#[cfg(test)]
mod tests {
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
        for (enum_name, variants) in PLUGIN_REGISTRY {
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
    /// `enum_variants` searches the engine registry first, so a duplicated name
    /// here would be silently unreachable — the plugin's variants would never be
    /// the ones Python saw, and the parity test would compare the Python mirror
    /// against the wrong Rust enum.
    #[test]
    fn plugin_enums_do_not_shadow_engine_enums() {
        let engine = view_buffer::naming::registered_names();
        let clashes: Vec<_> = PLUGIN_REGISTRY
            .iter()
            .map(|(n, _)| *n)
            .filter(|n| engine.contains(n))
            .collect();
        assert!(
            clashes.is_empty(),
            "these plugin enums share a name with an engine enum, so \
             `enum_variants` would answer with the engine's: {clashes:?}"
        );
    }
}
