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

        impl $crate::naming::NamedEnum for $ty {
            const ENUM_NAME: &'static str = stringify!($ty);
            fn variant_names() -> Vec<&'static str> {
                $crate::naming::names(Self::NAMED)
            }
        }
        // Exhaustiveness guard: a new variant fails to compile here until it
        // is added to the NAMED table above.
        const _: fn($ty) = |v: $ty| match v { $($ty::$variant => ()),+ };
    };
}

pub(crate) use named_variants;

/// An enum with a canonical `NAMED` table, implemented by `named_variants!`.
///
/// Exists so [`REGISTRY`] can list *types* and read their names generically,
/// rather than repeating each enum's name and accessor by hand.
pub trait NamedEnum {
    /// The enum's own name, as Python knows it.
    const ENUM_NAME: &'static str;
    /// Its variant names, in declaration order.
    fn variant_names() -> Vec<&'static str>;
}

/// Register every enum whose names cross the FFI.
///
/// One line per enum, and that line is the whole registration. The entries are
/// read by this module's uniqueness test *and*, through
/// [`registered_variants`]/[`registered_names`], by the plugin's
/// `enum_variants`/`enum_names` FFI — so adding an enum here is what makes
/// Python able to query it and what gets its names checked for duplicates.
/// There is no second list to update.
///
/// Both lists this replaced had already drifted: the uniqueness test omitted
/// `LabelReduction` and `LabelRegionMode`, and Python's parity tests named
/// neither, so a divergence in either would have shipped unnoticed.
///
/// Do not add a hand-written arm to `enum_variants` for a new enum. The one
/// entry that is not here — `BinaryOp` — is absent because its table lives in
/// the plugin crate and this crate cannot reference it, not as a precedent.
macro_rules! registry {
    ($($ty:path),+ $(,)?) => {
        /// Every enum surfaced across the FFI: `(name, variant names)`.
        pub const REGISTRY: &[(&str, fn() -> Vec<&'static str>)] = &[
            $((
                <$ty as NamedEnum>::ENUM_NAME,
                <$ty as NamedEnum>::variant_names as fn() -> Vec<&'static str>,
            )),+
        ];
    };
}

registry!(
    crate::core::dtype::DType,
    crate::ops::Domain,
    crate::ops::color::ColorSpace,
    crate::ops::image::FilterType,
    crate::ops::filter::BorderMode,
    crate::ops::pad::PadMode,
    crate::ops::pad::PadPosition,
    crate::ops::phash::HashAlgorithm,
    crate::ops::histogram::HistogramOutput,
    crate::ops::histogram::HistogramClosed,
    crate::ops::affine::InterpolationType,
    crate::geometry::ops::ExtractMode,
    crate::geometry::ops::ApproxMethod,
    crate::geometry::label::LabelReduction,
    crate::geometry::label::LabelRegionMode,
    crate::ops::NormalizeMethod,
);

// `NormalizeMethod::Preset` carries payload, so it has no value table and
// cannot use `named_variants!`. It is registered by hand off its `NAMES` list
// (which has its own exhaustiveness guard) rather than left out — being
// unregistered is what stops an enum from being parity-checked at all.
impl NamedEnum for crate::ops::NormalizeMethod {
    const ENUM_NAME: &'static str = "NormalizeMethod";
    fn variant_names() -> Vec<&'static str> {
        Self::NAMES.to_vec()
    }
}

/// Look up a registered enum's variant names.
pub fn registered_variants(name: &str) -> Option<Vec<&'static str>> {
    REGISTRY.iter().find_map(|(n, f)| (*n == name).then(f))
}

/// The names of every registered enum.
pub fn registered_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|(n, _)| *n).collect()
}

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
    use super::REGISTRY;

    /// Names within an enum must be unique, for *every registered enum*.
    ///
    /// This used to be a hand-written list of thirteen `assert_unique` calls,
    /// which had already drifted from reality: `LabelReduction` and
    /// `LabelRegionMode` both had `NAMED` tables and neither was listed.
    /// Iterating the registry means registering an enum is what gets it
    /// checked.
    #[test]
    fn registered_enums_have_unique_names() {
        for (enum_name, variants) in REGISTRY {
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

    /// The registry must not be trivially empty or half-populated — a
    /// regression that would make the test above vacuous.
    #[test]
    fn registry_covers_the_known_enums() {
        let names = super::registered_names();
        for expected in [
            "DType",
            "Domain",
            "ColorSpace",
            "FilterType",
            "BorderMode",
            "PadMode",
            "PadPosition",
            "HashAlgorithm",
            "HistogramOutput",
            "HistogramClosed",
            "InterpolationType",
            "ExtractMode",
            "ApproxMethod",
            "LabelReduction",
            "LabelRegionMode",
            "NormalizeMethod",
        ] {
            assert!(names.contains(&expected), "{expected} is not registered");
        }
    }
}
