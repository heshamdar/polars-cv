//! SPIKE (throwaway): an Arrow extension type for `polars_cv.contour`.
//!
//! Part of the design-review extension-type spike, rounding out the geometry
//! family (point, contour, bbox). A tagged column lets a consumer identify a
//! contour (`{exterior, holes, is_closed}`) by its type tag rather than by the
//! structural "looks-like" field matching `parse_contour` does today.
//!
//! Isolated and additive: not wired into the `.contour` namespace or
//! `geom_schema` parity guards. Storage stays the plain `contour_fields()`
//! struct (one authority). Delete after the migrate-or-drop decision.

use std::any::Any;
use std::borrow::Cow;
use std::hash::BuildHasher;
use std::sync::Arc;

use polars::datatypes::extension::{
    register_extension_type, ExtensionTypeFactory, ExtensionTypeImpl, ExtensionTypeInstance,
};
use polars::prelude::*;
use polars_utils::aliases::PlFixedStateQuality;
use pyo3_polars::derive::polars_expr;

pub(crate) const CONTOUR_EXT_NAME: &str = "polars_cv.contour";

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Contour;

impl ExtensionTypeImpl for Contour {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(CONTOUR_EXT_NAME)
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
        PlFixedStateQuality::default().hash_one(CONTOUR_EXT_NAME)
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed("contour")
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Borrowed("Contour")
    }
}

pub(crate) struct ContourFactory;

impl ExtensionTypeFactory for ContourFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        _storage: &DataType,
        _metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        Box::new(Contour)
    }
}

/// Register `polars_cv.contour` on the plugin's copy of polars-core.
pub(crate) fn register() -> PolarsResult<()> {
    register_extension_type(CONTOUR_EXT_NAME, Some(Arc::new(ContourFactory)))
}

fn contour_instance() -> ExtensionTypeInstance {
    ExtensionTypeInstance(Box::new(Contour))
}

fn contour_ext_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let DataType::Extension(typ, _) = field.dtype() else {
        polars_bail!(
            SchemaMismatch: "expected a `polars_cv.contour` column, got: {}", field.dtype()
        );
    };
    if (&*typ.0 as &dyn Any).downcast_ref::<Contour>().is_none() {
        polars_bail!(
            SchemaMismatch: "expected a `polars_cv.contour` column, got extension: {}", typ.name()
        );
    }
    Ok(field.clone())
}

/// Identity op over a `polars_cv.contour` column: recover storage via
/// `.ext()?.storage()` and re-wrap, proving boundary survival through the shared
/// path (no kwargs — nothing to deny_unknown_fields).
#[polars_expr(output_type_func=contour_ext_output)]
fn contour_ext_identity(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let storage = s.ext()?.storage().clone();
    Ok(storage.into_extension(contour_instance()))
}
