//! Canonical name tables for user-facing enums.
//!
//! Every string-valued operation parameter maps to a Rust enum defined in
//! this crate. The `named_variants!` macro declares the **single authority**
//! for that enum's Python-facing names: a `NAMED` table consumed by both the
//! polars-cv parameter parser (accepting these names) and the `enum_variants`
//! FFI (surfacing them to Python parity tests). Parser and surfaced names
//! therefore cannot drift.
//!
//! The macro also emits a hidden exhaustive `match` over the listed variants,
//! so adding an enum variant without extending its `NAMED` table is a compile
//! error — the table can never silently under-list.

/// Declare `pub const NAMED: &[(&'static str, Self)]` for a fieldless enum.
///
/// ```ignore
/// named_variants!(BorderMode {
///     "replicate" => Replicate,
///     "zero" => Zero,
///     "reflect" => Reflect,
/// });
/// ```
macro_rules! named_variants {
    ($ty:ident { $($name:literal => $variant:ident),+ $(,)? }) => {
        impl $ty {
            /// Canonical Python-facing name of every variant.
            ///
            /// Single authority for parameter parsing and the `enum_variants`
            /// FFI — see `view_buffer::naming`.
            pub const NAMED: &'static [(&'static str, $ty)] = &[
                $(($name, $ty::$variant)),+
            ];
        }
        // Exhaustiveness guard: a new variant fails to compile here until it
        // is added to the NAMED table above.
        const _: fn($ty) = |v: $ty| match v { $($ty::$variant => ()),+ };
    };
}

pub(crate) use named_variants;

/// Look up the enum value for `name` in a `NAMED`-style table.
pub fn lookup<T: Copy>(table: &[(&str, T)], name: &str) -> Option<T> {
    table.iter().find_map(|(n, v)| (*n == name).then_some(*v))
}

/// The canonical names of a `NAMED`-style table (for error messages).
pub fn names<'a, T>(table: &'a [(&'a str, T)]) -> Vec<&'a str> {
    table.iter().map(|(n, _)| *n).collect()
}

#[cfg(test)]
mod tests {
    /// Names in every NAMED table must be unique — checked centrally here for
    /// each table registered below.
    #[test]
    fn named_tables_have_unique_names() {
        fn assert_unique<T>(table: &[(&str, T)], enum_name: &str) {
            for (i, (name, _)) in table.iter().enumerate() {
                assert!(!name.is_empty(), "{enum_name}: empty name");
                assert!(
                    table.iter().skip(i + 1).all(|(n, _)| n != name),
                    "{enum_name}: duplicate name '{name}'"
                );
            }
        }
        assert_unique(crate::ops::image::FilterType::NAMED, "FilterType");
        assert_unique(crate::ops::dto::PadMode::NAMED, "PadMode");
        assert_unique(crate::ops::dto::PadPosition::NAMED, "PadPosition");
        assert_unique(crate::ops::phash::HashAlgorithm::NAMED, "HashAlgorithm");
        assert_unique(
            crate::ops::histogram::HistogramOutput::NAMED,
            "HistogramOutput",
        );
        assert_unique(
            crate::ops::histogram::HistogramClosed::NAMED,
            "HistogramClosed",
        );
        assert_unique(crate::geometry::ops::ExtractMode::NAMED, "ExtractMode");
        assert_unique(crate::geometry::ops::ApproxMethod::NAMED, "ApproxMethod");
        assert_unique(
            crate::ops::affine::InterpolationType::NAMED,
            "InterpolationType",
        );
        assert_unique(crate::ops::filter::BorderMode::NAMED, "BorderMode");
        assert_unique(crate::ops::color::ColorSpace::NAMED, "ColorSpace");
        assert_unique(crate::core::dtype::DType::NAMED, "DType");
        assert_unique(crate::ops::Domain::NAMED, "Domain");
    }
}
