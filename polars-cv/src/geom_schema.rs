//! The one declaration of the `{x, y}` point struct the geometry surfaces publish.
//!
//! `contour.rs`, `point.rs` and `geometry/schemas.py` each used to spell this
//! out — eight separate `Field::new("x", Float64), Field::new("y", Float64)`
//! constructions with nothing relating them. Dtypes, op names and enum
//! variants all have a named authority and a bidirectional parity guard; this
//! surface had neither, so a rename here could land in one module and not the
//! others, and the mismatch would only show as a confusing dtype error at
//! query time.
//!
//! Rust reads [`point_fields`]; Python holds `POINT_SCHEMA` to it through the
//! `point_schema` FFI, the way `enum_variants` surfaces the naming registry —
//! a runtime accessor plus a parity test, rather than a generated file, so
//! `polars_cv.geometry` still imports with no compiled extension present.
//!
//! A bbox is deliberately *not* built from these fields. It begins `x, y` by
//! coincidence of meaning, not by sharing this schema, and chaining them would
//! silently drag a point rename into the bbox wire format.

use polars::prelude::*;

/// The field names of an `{x, y}` point, in wire order.
///
/// Separate from [`point_fields`] so the FFI can publish the names without
/// constructing polars types, and so a rename has exactly one site.
pub(crate) const POINT_FIELD_NAMES: [&str; 2] = ["x", "y"];

/// The `{x, y}` fields a point-valued result publishes.
///
/// **This is the declaration.** Every other point struct in the plugin is
/// built from it — see the module docs for why a second one is a bug.
pub(crate) fn point_fields() -> Vec<Field> {
    vec![
        Field::new(PlSmallStr::from_static("x"), DataType::Float64),
        Field::new(PlSmallStr::from_static("y"), DataType::Float64),
    ]
}

/// The `{x, y}` struct dtype a point-valued result publishes.
pub(crate) fn point_struct_dtype() -> DataType {
    DataType::Struct(point_fields())
}

/// One point as a struct value matching [`point_struct_dtype`].
pub(crate) fn point_anyvalue(x: f64, y: f64) -> AnyValue<'static> {
    AnyValue::StructOwned(Box::new((
        vec![AnyValue::Float64(x), AnyValue::Float64(y)],
        point_fields(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_and_the_fields_agree() {
        // The FFI publishes `POINT_FIELD_NAMES` while Rust callers build
        // `point_fields()`. Nothing else compares the two, so a rename applied
        // to one alone would let Python check a schema Rust does not produce.
        let fields = point_fields();
        let built: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(built, POINT_FIELD_NAMES.to_vec());
    }

    #[test]
    fn a_point_value_matches_the_published_dtype() {
        let DataType::Struct(fields) = point_struct_dtype() else {
            panic!("point_struct_dtype must be a struct");
        };
        let AnyValue::StructOwned(payload) = point_anyvalue(1.0, 2.0) else {
            panic!("point_anyvalue must be an owned struct");
        };
        assert_eq!(payload.1, fields);
        assert_eq!(payload.0.len(), fields.len());
    }
}
