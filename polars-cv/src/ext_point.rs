//! SPIKE (throwaway): an Arrow extension type for `polars_cv.point`.
//!
//! Feasibility probe for the Polars-plugin design review. It answers whether a
//! geometry column can carry a *semantic type tag* (`polars_cv.point`) across
//! the plugin boundary instead of being an anonymous `{x, y}` struct that every
//! consumer has to identify by field-name inspection.
//!
//! What it proves:
//!   1. the extension API is reachable at the pinned polars 0.54 / pyo3-polars
//!      0.27 stack (no version bump);
//!   2. a tagged column survives a round trip and arrives *inside* a plugin
//!      function still tagged (not decayed to storage);
//!   3. passing a non-point column to a point op fails at schema resolution.
//!
//! It is deliberately NOT wired into the `.point` namespace, `geom_schema`'s
//! parity guards, or the FFI schema publishing. Storage is checked against
//! [`crate::geom_schema::point_struct_dtype`] at schema resolution (see
//! [`crate::ext_check`]), so the `{x, y}` layout keeps a single authority.
//! Delete after the migrate-or-drop decision.

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
use serde::Deserialize;

/// The extension name written to `ARROW:extension:name`. Must match the Python
/// `PointXY.name` so both copies of polars-core agree on the tag.
pub(crate) const POINT_EXT_NAME: &str = "polars_cv.point";

/// The concrete extension type: a 2-D `{x, y}` point. Carries no metadata — the
/// shape is fully described by the storage struct.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PointXY;

impl ExtensionTypeImpl for PointXY {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(POINT_EXT_NAME)
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
        // No fields to distinguish instances; hash the name, as the generic
        // extension type does.
        PlFixedStateQuality::default().hash_one(POINT_EXT_NAME)
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed("point[xy]")
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Borrowed("PointXY")
    }
}

/// Factory for [`POINT_EXT_NAME`]. One name, one shape here — a production
/// version would branch on `storage` to pick 2-D vs 3-D (per the plugin docs).
pub(crate) struct PointFactory;

impl ExtensionTypeFactory for PointFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        _storage: &DataType,
        _metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        Box::new(PointXY)
    }
}

/// Register `polars_cv.point` on the plugin's copy of polars-core. Called from
/// the `#[pymodule]` init hook, mirroring the host-side `pl.register_extension_type`.
pub(crate) fn register() -> PolarsResult<()> {
    register_extension_type(POINT_EXT_NAME, Some(Arc::new(PointFactory)))
}

/// A fresh [`ExtensionTypeInstance`] wrapping [`PointXY`], for `into_extension`.
fn point_instance() -> ExtensionTypeInstance {
    ExtensionTypeInstance(Box::new(PointXY))
}

/// Echo the point extension field so the output keeps its `polars_cv.point`
/// tag, and reject a non-point input or wrong storage *at schema resolution* —
/// the improved failure mode the spike is testing for (vs. a query-time
/// struct-parse error).
fn point_ext_output(input_fields: &[Field]) -> PolarsResult<Field> {
    crate::ext_check::expect_ext::<PointXY>(
        &input_fields[0],
        POINT_EXT_NAME,
        &crate::geom_schema::point_struct_dtype(),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TranslateKwargs {
    dx: f64,
    dy: f64,
}

/// Translate a `polars_cv.point` column by `(dx, dy)`, preserving the tag.
///
/// Recovers the `{x, y}` storage via `.ext()?.storage()`, does the math on the
/// child columns, then re-wraps with a `PointXY` instance via `into_extension`.
/// If the input arrived decayed to plain storage (registration missing), `ext()`
/// errors — which is exactly the signal that the two copies did not agree.
#[polars_expr(output_type_func=point_ext_output)]
fn point_ext_translate(inputs: &[Series], kwargs: TranslateKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let storage = s.ext()?.storage();
    let sc = storage.struct_()?;

    let x_src = sc.field_by_name("x")?;
    let y_src = sc.field_by_name("y")?;

    let x_new = Float64Chunked::from_iter_options(
        PlSmallStr::from_static("x"),
        x_src.f64()?.iter().map(|o| o.map(|v| v + kwargs.dx)),
    )
    .into_series();
    let y_new = Float64Chunked::from_iter_options(
        PlSmallStr::from_static("y"),
        y_src.f64()?.iter().map(|o| o.map(|v| v + kwargs.dy)),
    )
    .into_series();

    let out = StructChunked::from_series(s.name().clone(), storage.len(), [x_new, y_new].iter())?
        .into_series();
    Ok(out.into_extension(point_instance()))
}
