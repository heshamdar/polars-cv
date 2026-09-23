//! SPIKE (throwaway): an Arrow extension type for `polars_cv.bbox`.
//!
//! Part of the design-review extension-type spike, rounding out the geometry
//! family (point, contour, bbox). A tagged column lets a consumer tell a bbox
//! from any other `{x, y, width, height}` struct by its type tag rather than by
//! field-name inspection.
//!
//! Isolated and additive: not wired into the `.bbox` namespace or `geom_schema`
//! parity guards. Storage is checked against
//! [`crate::geom_schema::bbox_struct_dtype`] at schema resolution (see
//! [`crate::ext_check`]). Delete after the migrate-or-drop decision.

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

pub(crate) const BBOX_EXT_NAME: &str = "polars_cv.bbox";

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct BBox;

impl ExtensionTypeImpl for BBox {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(BBOX_EXT_NAME)
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
        PlFixedStateQuality::default().hash_one(BBOX_EXT_NAME)
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed("bbox")
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Borrowed("BBox")
    }
}

pub(crate) struct BBoxFactory;

impl ExtensionTypeFactory for BBoxFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        _storage: &DataType,
        _metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        Box::new(BBox)
    }
}

/// Register `polars_cv.bbox` on the plugin's copy of polars-core.
pub(crate) fn register() -> PolarsResult<()> {
    register_extension_type(BBOX_EXT_NAME, Some(Arc::new(BBoxFactory)))
}

fn bbox_instance() -> ExtensionTypeInstance {
    ExtensionTypeInstance(Box::new(BBox))
}

fn bbox_ext_output(input_fields: &[Field]) -> PolarsResult<Field> {
    crate::ext_check::expect_ext::<BBox>(
        &input_fields[0],
        BBOX_EXT_NAME,
        &crate::geom_schema::bbox_struct_dtype(),
    )
}

/// Identity op over a `polars_cv.bbox` column: recover storage via
/// `.ext()?.storage()` and re-wrap, proving boundary survival through the shared
/// path (no kwargs — nothing to deny_unknown_fields).
#[polars_expr(output_type_func=bbox_ext_output)]
fn bbox_ext_identity(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let storage = s.ext()?.storage().clone();
    Ok(storage.into_extension(bbox_instance()))
}
