//! Canonical name tables for user-facing enums.
//!
//! Every string-valued operation parameter maps to a Rust enum defined in
//! this crate. The `named_variants!` macro declares the **single authority**
//! for that enum's Python-facing names: a `NAMED` table consumed by both the
//! polars-cv parameter parser (accepting these names) and the enum catalogue
//! the Python enums are generated from. Parser and Python names therefore
//! cannot drift.
//!
//! The macro also emits a hidden exhaustive `match` over the listed variants,
//! so adding an enum variant without extending its `NAMED` table is a compile
//! error — the table can never silently under-list.

/// Declare `pub const NAMED: &[(&'static str, Self)]` for a fieldless enum.
///
/// ```ignore
/// named_variants!(BorderMode: "Border-handling mode for 2D convolution (``convolve2d``).\n\n- REPLICATE: Replicate the nearest edge pixel.\n- ZERO: Treat out-of-bounds pixels as zero.\n- REFLECT: Reflect pixels around the edge (dcba|abcd|dcba)." {
///     "replicate" => Replicate,
///     "zero" => Zero,
///     "reflect" => Reflect,
/// });
/// ```
///
/// A variant may carry additional accepted spellings after the canonical one:
///
/// ```ignore
/// named_variants!(Winding: "Winding direction of a contour ring (``.contour.ensure_winding``).\n\nLong spellings included: the plugin has always accepted\n``\"clockwise\"``/``\"counterclockwise\"`` alongside the short forms." {
///     "ccw" | "counterclockwise" => CounterClockwise,
///     "cw" | "clockwise" => Clockwise,
/// });
/// ```
///
/// Aliases join `NAMED` — so they are accepted by the parser *and* surfaced
/// in the generated Python enum. They do not appear separately in the exhaustiveness guard, because
/// each variant is still named exactly once there.
///
/// Exported from the crate so the **plugin** crate can declare its own
/// vocabularies through the same mechanism. Some enums genuinely belong there
/// and cannot move here — `RowErrorPolicy` describes graph execution and
/// `NullParamPolicy` describes per-row parameter resolution, neither of which
/// this crate has a concept of — and the alternative to lending them the macro
/// a hand-written Python class per enum, which is exactly the second list this
/// module exists to abolish.
#[macro_export]
macro_rules! named_variants {
    ($ty:ident $(: $doc:literal)? { $($name:literal $(| $alias:literal)* => $variant:ident),+ $(,)? }) => {
        impl $ty {
            /// Canonical Python-facing name of every variant.
            ///
            /// Single authority for parameter parsing and the generated Python
            /// enum — see `view_buffer::naming`.
            pub const NAMED: &'static [(&'static str, $ty)] = &[
                $(($name, $ty::$variant) $(, ($alias, $ty::$variant))*),+
            ];
        }

        impl $crate::naming::NamedEnum for $ty {
            const ENUM_NAME: &'static str = stringify!($ty);
            const DOC: &'static str = $crate::naming::first_or_empty(&[$($doc)?]);
            fn variant_names() -> Vec<&'static str> {
                $crate::naming::names(Self::NAMED)
            }
        }
        impl $crate::naming::WireScalar for $ty {
            const KIND: $crate::naming::WireKind = $crate::naming::WireKind::Name;
            const PY_TYPE: &'static str = stringify!($ty);
            fn from_wire(value: $crate::naming::WireValue<'_>) -> Result<Self, String> {
                $crate::naming::named_from_wire(stringify!($ty), Self::NAMED, value)
            }
            fn to_wire(self) -> $crate::naming::WireValue<'static> {
                $crate::naming::named_to_wire(Self::NAMED, self)
            }
            fn spellings() -> Vec<&'static str> {
                $crate::naming::names(Self::NAMED)
            }
            fn probe_value() -> Self {
                Self::NAMED[0].1
            }
        }
        // Exhaustiveness guard: a new variant fails to compile here until it
        // is added to the NAMED table above.
        const _: fn($ty) = |v: $ty| match v { $($ty::$variant => ()),+ };
    };
}

// `#[macro_export]` places the macro at the crate root; re-exporting it here
// keeps `crate::naming::named_variants!(...)` working for this crate's own
// call sites, so the two crates spell the invocation the same way.
pub use crate::named_variants;

/// An enum with a canonical `NAMED` table, implemented by `named_variants!`.
///
/// Exists so [`REGISTRY`] can list *types* and read their names generically,
/// rather than repeating each enum's name and accessor by hand.
pub trait NamedEnum {
    /// The enum's own name, as Python knows it.
    const ENUM_NAME: &'static str;
    /// Its description, the generated Python enum's docstring (empty when the
    /// invocation gives none).
    const DOC: &'static str;
    /// Its variant names, in declaration order.
    fn variant_names() -> Vec<&'static str>;
}

/// The doc a `named_variants!` invocation gave, or `""`.
pub const fn first_or_empty(docs: &[&'static str]) -> &'static str {
    match docs {
        [doc, ..] => doc,
        [] => "",
    }
}

/// Register every enum whose names cross the FFI.
///
/// One line per enum, and that line is the whole registration. The entries are
/// read by this module's uniqueness test *and* by the plugin's enum catalogue
/// (`enum_catalog_json`), which the Python enums are generated from — so
/// adding an enum here is what gives Python its class and what gets its names
/// checked for duplicates. There is no second list to update.
///
/// An enum that genuinely belongs to the plugin (`RowErrorPolicy` and friends)
/// declares itself with the exported [`named_variants!`] and lands in that
/// crate's own `registry!`, which the catalogue chains onto this one.
///
/// Exported alongside `named_variants!` for that purpose. The generated const
/// is named by the caller, so the two registries can coexist without one
/// shadowing the other.
#[macro_export]
macro_rules! registry {
    ($name:ident: $($ty:path),+ $(,)?) => {
        /// Every enum surfaced across the FFI: `(name, variant names, doc)`.
        pub const $name: &[(&str, fn() -> Vec<&'static str>, &str)] = &[
            $((
                <$ty as $crate::naming::NamedEnum>::ENUM_NAME,
                <$ty as $crate::naming::NamedEnum>::variant_names
                    as fn() -> Vec<&'static str>,
                <$ty as $crate::naming::NamedEnum>::DOC,
            )),+
        ];
    };
}

pub use crate::registry;

registry!(
    REGISTRY:
    crate::core::dtype::DType,
    crate::geometry::contour::Winding,
    crate::ops::binary::BinaryOp,
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
    crate::geometry::ops::ScaleOrigin,
    crate::geometry::ops::ExtractMode,
    crate::geometry::ops::ApproxMethod,
    crate::geometry::label::LabelReduction,
    crate::geometry::label::LabelRegionMode,
    crate::ops::NormalizeMethod,
);

/// Look up the enum value for `name` in a `NAMED`-style table.
pub fn lookup<T: Copy>(table: &[(&str, T)], name: &str) -> Option<T> {
    table.iter().find_map(|(n, v)| (*n == name).then_some(*v))
}

/// The canonical names of a `NAMED`-style table (for error messages).
pub fn names<'a, T>(table: &'a [(&'a str, T)]) -> Vec<&'a str> {
    table.iter().map(|(n, _)| *n).collect()
}

/// A scalar parameter value as it appears on the wire or in a per-row column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WireValue<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(&'a str),
}

/// Which kind of per-row column a [`WireScalar`] reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireKind {
    Int,
    Float,
    Bool,
    /// A named enum, read from a string column through its `NAMED` table.
    Name,
}

/// A value an op parameter may hold: how it is spelled on the wire and in a
/// per-row column, and how the op catalogue names its type.
///
/// Implemented here for the primitive types and, by [`named_variants!`], for
/// every named enum — so an enum's spellings on the wire are its `NAMED`
/// table and nothing else. (It lives in this crate, beside `NamedEnum`,
/// because the plugin cannot give a blanket impl over this crate's enums
/// alongside impls for primitives.)
pub trait WireScalar: Sized + Copy + PartialEq + core::fmt::Debug {
    /// The per-row column kind.
    const KIND: WireKind;
    /// The Python type: `int`, `float`, `bool`, or the enum's name.
    const PY_TYPE: &'static str;
    /// Parse a wire or column value; the error names what was expected.
    fn from_wire(value: WireValue<'_>) -> Result<Self, String>;
    /// The canonical wire value.
    fn to_wire(self) -> WireValue<'static>;
    /// Every accepted spelling, for a named enum; empty otherwise.
    fn spellings() -> Vec<&'static str>;
    /// A valid value to stand in for a per-row one under a plan-time probe.
    fn probe_value() -> Self;
}

fn describe_wire(value: WireValue<'_>) -> String {
    match value {
        WireValue::Int(i) => format!("the integer {i}"),
        WireValue::Float(f) => format!("the float {f}"),
        WireValue::Bool(b) => format!("the boolean {b}"),
        WireValue::Str(s) => format!("the string {s:?}"),
    }
}

macro_rules! wire_int {
    ($($t:ty),+) => {$(
        impl WireScalar for $t {
            const KIND: WireKind = WireKind::Int;
            const PY_TYPE: &'static str = "int";
            fn from_wire(value: WireValue<'_>) -> Result<Self, String> {
                match value {
                    WireValue::Int(i) => <$t>::try_from(i).map_err(|_| {
                        if i < 0 {
                            format!("{i} cannot be negative (expected a {} up to {})",
                                stringify!($t), <$t>::MAX)
                        } else {
                            format!("{i} is out of range for {} ({}..={})",
                                stringify!($t), <$t>::MIN, <$t>::MAX)
                        }
                    }),
                    other => Err(format!("expected an integer, got {}", describe_wire(other))),
                }
            }
            fn to_wire(self) -> WireValue<'static> {
                WireValue::Int(i64::from(self))
            }
            fn spellings() -> Vec<&'static str> {
                Vec::new()
            }
            fn probe_value() -> Self {
                0
            }
        }
    )+};
}
wire_int!(u8, u32, i32, i64);

macro_rules! wire_float {
    ($($t:ty),+) => {$(
        impl WireScalar for $t {
            const KIND: WireKind = WireKind::Float;
            const PY_TYPE: &'static str = "float";
            fn from_wire(value: WireValue<'_>) -> Result<Self, String> {
                match value {
                    // An integer is a float the caller did not write a `.0` on.
                    WireValue::Int(i) => Ok(i as $t),
                    WireValue::Float(f) => Ok(f as $t),
                    other => Err(format!("expected a number, got {}", describe_wire(other))),
                }
            }
            fn to_wire(self) -> WireValue<'static> {
                WireValue::Float(f64::from(self))
            }
            fn spellings() -> Vec<&'static str> {
                Vec::new()
            }
            fn probe_value() -> Self {
                0.0
            }
        }
    )+};
}
wire_float!(f32, f64);

impl WireScalar for bool {
    const KIND: WireKind = WireKind::Bool;
    const PY_TYPE: &'static str = "bool";
    fn from_wire(value: WireValue<'_>) -> Result<Self, String> {
        match value {
            WireValue::Bool(b) => Ok(b),
            other => Err(format!("expected a boolean, got {}", describe_wire(other))),
        }
    }
    fn to_wire(self) -> WireValue<'static> {
        WireValue::Bool(self)
    }
    fn spellings() -> Vec<&'static str> {
        Vec::new()
    }
    fn probe_value() -> Self {
        false
    }
}

/// [`WireScalar::from_wire`] for a named enum: a string in its `NAMED` table.
pub fn named_from_wire<T: Copy>(
    type_name: &str,
    table: &[(&'static str, T)],
    value: WireValue<'_>,
) -> Result<T, String> {
    match value {
        WireValue::Str(s) => lookup(table, s).ok_or_else(|| {
            format!(
                "unknown {type_name} {s:?}, expected one of {:?}",
                names(table)
            )
        }),
        other => Err(format!(
            "expected a {type_name} name (one of {:?}), got {}",
            names(table),
            describe_wire(other)
        )),
    }
}

/// [`WireScalar::to_wire`] for a named enum: the variant's canonical name.
pub fn named_to_wire<T: Copy + PartialEq>(table: &[(&'static str, T)], v: T) -> WireValue<'static> {
    let name = table
        .iter()
        .find(|(_, candidate)| *candidate == v)
        .map(|(name, _)| *name)
        .expect("named_variants! lists every variant");
    WireValue::Str(name)
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
        for (enum_name, variants, _) in REGISTRY {
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

    /// Every `named_variants!` enum in this crate, found by scanning `src/`.
    ///
    /// Deliberately not a hand-written list: a list is the thing this test
    /// exists to eliminate. The invocation is unambiguous (`named_variants!(`
    /// followed by the type name), so the scan needs no Rust parsing — and the
    /// set comparison below fails loudly if it ever matches nothing, which is
    /// the failure mode a source scan has to be protected from.
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
                        // Skip doc comments (the macro's own usage example)
                        // and the `macro_rules!` definition itself.
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

    /// Declaring a `NAMED` table and registering it are the same act.
    ///
    /// This is the hole the registry would otherwise leave open. `REGISTRY` is
    /// what gets an enum's names checked for duplicates above and what puts it
    /// in the plugin's enum catalogue, which the generated Python enums are
    /// built from. So an enum with a `named_variants!` table that is *not* registered
    /// is invisible to every one of those: Python would have no class for it,
    /// or a hand-written one free to disagree with it. Not registering must therefore fail here,
    /// not silently opt the enum out.
    ///
    /// The reverse direction matters as much: it is what stops this test from
    /// passing vacuously if the scan stops matching. A registry name with no
    /// `named_variants!` invocation behind it means either the scan rotted or
    /// the enum was registered without the table that makes it authoritative.
    #[test]
    fn every_named_enum_is_registered() {
        let declared = declared_enums();
        let registered: std::collections::BTreeSet<String> = super::REGISTRY
            .iter()
            .map(|(n, _, _)| n.to_string())
            .collect();

        let unregistered: Vec<&String> = declared.difference(&registered).collect();
        assert!(
            unregistered.is_empty(),
            "these enums declare a named_variants! table but are not in \
             REGISTRY: {unregistered:?}. Add each to the registry! invocation \
             in this module — that one line is what surfaces it to Python and \
             what generates its Python class. Leaving it out does not make it \
             private, it makes it unchecked."
        );

        let unbacked: Vec<&String> = registered.difference(&declared).collect();
        assert!(
            unbacked.is_empty(),
            "these names are in REGISTRY but no named_variants! invocation was \
             found for them: {unbacked:?}. Either the source scan in \
             declared_enums() has stopped matching -- in which case the check \
             above is passing vacuously -- or the enum was registered without \
             the NAMED table that makes it authoritative."
        );
    }
}
