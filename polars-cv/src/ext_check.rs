//! SPIKE (throwaway): the schema-resolution check every spike extension op shares.
//!
//! A tag alone is not enough: the plugin-side factories build their type from
//! the name and cannot fail, so a column tagged `polars_cv.point` over the wrong
//! storage (e.g. a Parquet file written elsewhere) would otherwise pass schema
//! resolution and fail mid-execution on a child-column dtype. This check compares
//! the storage against the canonical layout from its single authority
//! (`geom_schema` / `output::numpy_output_dtype`) and rejects a mismatch up front.
//! Delete with the rest of the spike.

use std::any::Any;

use polars::datatypes::extension::ExtensionTypeImpl;
use polars::prelude::*;

/// Accept `field` only if it carries extension `T` (named `name`) over exactly
/// `storage`; return it unchanged so the op's output keeps the tag.
pub(crate) fn expect_ext<T: ExtensionTypeImpl + 'static>(
    field: &Field,
    name: &str,
    storage: &DataType,
) -> PolarsResult<Field> {
    let DataType::Extension(typ, actual) = field.dtype() else {
        polars_bail!(SchemaMismatch: "expected a `{name}` column, got: {}", field.dtype());
    };
    if (&*typ.0 as &dyn Any).downcast_ref::<T>().is_none() {
        polars_bail!(SchemaMismatch: "expected a `{name}` column, got extension: {}", typ.name());
    }
    if actual.as_ref() != storage {
        polars_bail!(
            SchemaMismatch: "`{name}` column has storage {actual:?}, expected storage {storage:?}"
        );
    }
    Ok(field.clone())
}
