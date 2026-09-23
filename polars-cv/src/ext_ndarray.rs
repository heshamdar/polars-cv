//! SPIKE (throwaway): an Arrow extension type for `polars_cv.ndarray`.
//!
//! The second type in the design-review spike, proving the same lazy-registered
//! extension mechanism spans a different subsystem than geometry: the numpy/torch
//! sink struct (`{data, dtype, shape, strides, offset}`). A tagged column lets a
//! consumer identify a tensor by its type tag instead of sniffing struct fields.
//!
//! Isolated and additive: this does NOT change the production `.sink("numpy")`
//! path — its output stays a plain, untagged struct. Storage is checked against
//! [`crate::output::numpy_output_dtype`] at schema resolution (see
//! [`crate::ext_check`]), so the 5-field layout keeps one authority. Delete
//! after the migrate-or-drop decision.

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

/// The extension name written to `ARROW:extension:name`. Must match the Python
/// `NdArray.name` so both copies of polars-core agree on the tag.
pub(crate) const NDARRAY_EXT_NAME: &str = "polars_cv.ndarray";

/// The concrete extension type: a strided N-D array carried as the numpy sink
/// struct. No metadata — the layout is fully described by the storage struct.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct NdArray;

impl ExtensionTypeImpl for NdArray {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(NDARRAY_EXT_NAME)
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
        PlFixedStateQuality::default().hash_one(NDARRAY_EXT_NAME)
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed("ndarray")
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Borrowed("NdArray")
    }
}

/// Factory for [`NDARRAY_EXT_NAME`].
pub(crate) struct NdArrayFactory;

impl ExtensionTypeFactory for NdArrayFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        _storage: &DataType,
        _metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        Box::new(NdArray)
    }
}

/// Register `polars_cv.ndarray` on the plugin's copy of polars-core. Called from
/// the `#[pymodule]` init hook alongside the point registration.
pub(crate) fn register() -> PolarsResult<()> {
    register_extension_type(NDARRAY_EXT_NAME, Some(Arc::new(NdArrayFactory)))
}

/// A fresh [`ExtensionTypeInstance`] wrapping [`NdArray`], for `into_extension`.
fn ndarray_instance() -> ExtensionTypeInstance {
    ExtensionTypeInstance(Box::new(NdArray))
}

/// Echo the ndarray extension field so the output keeps its `polars_cv.ndarray`
/// tag, and reject a non-ndarray input or wrong storage at schema resolution.
fn ndarray_ext_output(input_fields: &[Field]) -> PolarsResult<Field> {
    crate::ext_check::expect_ext::<NdArray>(
        &input_fields[0],
        NDARRAY_EXT_NAME,
        &crate::output::numpy_output_dtype(),
    )
}

/// Identity op over a `polars_cv.ndarray` column: recover the struct storage via
/// `.ext()?.storage()` and re-wrap with `into_extension`. Proves a non-point
/// type survives the plugin boundary through the same code path (and that the
/// tag is preserved on output). No kwargs — nothing to deny_unknown_fields.
#[polars_expr(output_type_func=ndarray_ext_output)]
fn ndarray_ext_identity(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let storage = s.ext()?.storage().clone();
    Ok(storage.into_extension(ndarray_instance()))
}
