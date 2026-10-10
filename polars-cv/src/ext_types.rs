//! polars-cv's Arrow extension types — the Rust half.
//!
//! An extension type tags a column with what it is (`polars_cv.point`, …) on
//! top of the plain struct that stores it. [`ExtType`] is the single list of
//! them: its name, and its storage read from the authority that already owns
//! that layout (`geom_schema`, [`crate::output::numpy_output_dtype`]).
//!
//! ## Only outputs are tagged here
//!
//! The plugin never *receives* an extension dtype. Every Python route into the
//! plugin goes through `polars_cv._plugin.call`, which hands each argument over
//! as `.ext.storage()`; so no input-side code has to know tags exist, and a
//! tagged column is validated by exactly the parser that validates the plain
//! struct. What remains for Rust is building tagged *outputs*
//! ([`ExtType::tag`], [`ExtType::dtype`]), which needs no registry: the
//! instance is constructed here, and the host rebuilds it by name on the way
//! out (`polars_cv.extension_types`, registered at `import polars_cv`).
//!
//! That is also why nothing here registers with polars-core's extension
//! registry. The spike that preceded this module did, from the `#[pymodule]`
//! init — which polars' own `dlopen` of the library never runs — and so
//! depended on import order and on Python and polars loading the same file.
//!
//! ## Parity
//!
//! [`extension_types`](crate::extension_types) publishes `ExtType::ALL` over
//! FFI, and `test_python_types_match_the_rust_declaration` holds the Python
//! classes to it in both directions (names, order, storage).
//!
//! Polars documents extension types as unstable; this module is the only Rust
//! code that touches the API.

use std::any::Any;
use std::borrow::Cow;
use std::hash::BuildHasher;

use polars::datatypes::extension::{ExtensionTypeImpl, ExtensionTypeInstance};
use polars::prelude::*;
use polars_utils::aliases::PlFixedStateQuality;

/// A polars-cv extension type.
///
/// Adding a variant is a compile error in every `match` below until it names
/// its extension name and storage, and [`ExtType::ALL`] is checked against the
/// variants by `all_lists_every_variant_once`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExtType {
    /// A strided N-D array carried as the numpy/torch sink struct.
    NdArray,
    /// A 2-D `{x, y}` point.
    Point,
    /// A polygon `{exterior, holes, is_closed}`.
    Contour,
    /// An axis-aligned `{x, y, width, height}` box.
    BBox,
}

impl ExtType {
    /// Every type, in the order Python's `EXTENSION_TYPES` must list them.
    pub(crate) const ALL: [ExtType; 4] = [
        ExtType::NdArray,
        ExtType::Point,
        ExtType::Contour,
        ExtType::BBox,
    ];

    /// The name written to `ARROW:extension:name`.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            ExtType::NdArray => "polars_cv.ndarray",
            ExtType::Point => "polars_cv.point",
            ExtType::Contour => "polars_cv.contour",
            ExtType::BBox => "polars_cv.bbox",
        }
    }

    /// The storage dtype, from the authority that owns each layout.
    pub(crate) fn storage(self) -> DataType {
        match self {
            ExtType::NdArray => crate::output::numpy_output_dtype(),
            ExtType::Point => crate::geom_schema::point_struct_dtype(),
            ExtType::Contour => DataType::Struct(crate::geom_schema::contour_fields()),
            ExtType::BBox => crate::geom_schema::bbox_struct_dtype(),
        }
    }

    /// Short form shown in a DataFrame header (matches Python's `DISPLAY`).
    const fn display(self) -> &'static str {
        match self {
            ExtType::NdArray => "ndarray",
            ExtType::Point => "point",
            ExtType::Contour => "contour",
            ExtType::BBox => "bbox",
        }
    }

    /// The tagged dtype: this type over its storage.
    pub(crate) fn dtype(self) -> DataType {
        DataType::Extension(self.instance(), Box::new(self.storage()))
    }

    /// Tag `series` with this type — a zero-copy relabel.
    ///
    /// Errors, rather than tagging anyway, when the series is not exactly this
    /// type's storage: a tag over the wrong layout is the one thing a tag must
    /// never be, and every caller builds the storage from the same authority,
    /// so a mismatch is a bug in the caller.
    pub(crate) fn tag(self, series: Series) -> PolarsResult<Series> {
        let storage = self.storage();
        polars_ensure!(
            series.dtype() == &storage,
            SchemaMismatch: "cannot tag a {} column as `{}`: expected storage {:?}",
            series.dtype(), self.name(), storage
        );
        Ok(series.into_extension(self.instance()))
    }

    fn instance(self) -> ExtensionTypeInstance {
        ExtensionTypeInstance(Box::new(CvExtension(self)))
    }
}

/// The one [`ExtensionTypeImpl`] behind every [`ExtType`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct CvExtension(ExtType);

impl ExtensionTypeImpl for CvExtension {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.0.name())
    }

    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        None
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> {
        Box::new(self.clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        (other as &dyn Any)
            .downcast_ref::<Self>()
            .is_some_and(|o| self == o)
    }

    fn dyn_hash(&self) -> u64 {
        PlFixedStateQuality::default().hash_one(self.0.name())
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.0.display())
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{:?}", self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The variants of `ExtType`, parsed from this file's enum definition.
    ///
    /// Rust cannot enumerate an enum's variants without a derive or a second
    /// hand-written list, and a second list is what `ALL` already is. Same shape
    /// as `acknowledged_kinds` in `graph/sink_kind.rs`: the parse asserts it
    /// found a plausible set rather than silently matching nothing.
    fn declared_variants() -> Vec<String> {
        let src = include_str!("ext_types.rs");
        let body = src
            .split("pub(crate) enum ExtType {")
            .nth(1)
            .expect("the ExtType definition moved; this scan reads nothing")
            .split("\n}")
            .next()
            .expect("ExtType has no closing brace");
        let variants: Vec<String> = body
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && l.ends_with(','))
            .map(|l| l.trim_end_matches(',').to_string())
            .collect();
        assert!(
            variants.len() >= 4,
            "parsed {} ExtType variants; the scan is out of date",
            variants.len()
        );
        variants
    }

    #[test]
    fn all_lists_every_variant_once() {
        let listed: Vec<String> = ExtType::ALL.iter().map(|t| format!("{t:?}")).collect();
        assert_eq!(
            listed,
            declared_variants(),
            "ExtType::ALL must list every variant, once, in declaration order"
        );
    }

    #[test]
    fn names_share_the_polars_cv_prefix() {
        for t in ExtType::ALL {
            assert!(t.name().starts_with("polars_cv."), "{}", t.name());
        }
    }

    #[test]
    fn tag_relabels_canonical_storage() {
        let x = Series::new(PlSmallStr::from_static("x"), [1.0f64, 3.0]);
        let y = Series::new(PlSmallStr::from_static("y"), [2.0f64, 4.0]);
        let s = StructChunked::from_series(PlSmallStr::from_static("p"), 2, [x, y].iter())
            .unwrap()
            .into_series();

        let tagged = ExtType::Point.tag(s.clone()).unwrap();

        assert_eq!(tagged.dtype(), &ExtType::Point.dtype());
        assert_eq!(tagged.ext().unwrap().storage(), &s);
    }

    #[test]
    fn tag_refuses_other_storage() {
        let x = Series::new(PlSmallStr::from_static("x"), [1.0f32]);
        let y = Series::new(PlSmallStr::from_static("y"), [2.0f32]);
        let s = StructChunked::from_series(PlSmallStr::from_static("p"), 1, [x, y].iter())
            .unwrap()
            .into_series();

        let err = ExtType::Point.tag(s).unwrap_err();
        assert!(err.to_string().contains("polars_cv.point"), "{err}");
    }

    #[test]
    fn a_tagged_dtype_equals_only_its_own_type() {
        assert_eq!(ExtType::Point.dtype(), ExtType::Point.dtype());
        assert_ne!(ExtType::Point.dtype(), ExtType::BBox.dtype());
    }
}
