//! Per-row parameter resolution for the geometry namespaces.
//!
//! The `.contour` / `.point` / `.bbox` accessors are standalone
//! `#[polars_expr]` functions rather than `vb_graph` graph nodes, but they use
//! the same per-row mechanism as every typed op: a kwarg is a
//! [`Param<T>`](crate::ops::Param) — the literal value, or `{"$slot": n}`
//! naming the plugin input that holds it per row — and an optional data
//! operand (`point.rotate`'s `origin`, `correspond`'s `order`) is a
//! [`ColumnRef`]. Each function's arguments are its typed definition
//! (`geom_fns`), parsed strictly by name; the generated Python accessor appends
//! each expression as an input and writes its position into its field, so
//! nothing is looked up by name.
//!
//! Reading is delegated to [`crate::params::ParamCol`], so these namespaces
//! inherit the same dtype coverage, scalar broadcasting (a length-1 series from
//! an aggregation applies to every row) and [`NullParamPolicy`] the graph
//! engine uses. The policy arrives as an `on_null` kwarg (set from Python by
//! `_GeomNamespace.on_null`) and is applied by [`GeomParams::row`], which each
//! row loop wraps its parameter resolution in.

use crate::ops::ParamExt as _;
use polars::prelude::*;
use serde::Deserialize;
use view_buffer::mode::WireOps;
use view_buffer::naming::WireScalar;

use crate::ops::{ColumnRef, Literal, Param};
use crate::params::{NullParamPolicy, ParamCtx};

/// A geometry plugin call's kwargs: the function's own wire fields, and the
/// null policy (`_GeomNamespace.on_null`) every call carries.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeomKwargs {
    args: serde_json::Map<String, serde_json::Value>,
    on_null: Literal<NullParamPolicy>,
}

/// Per-row resolver over one plugin call's inputs.
pub struct GeomParams<'a> {
    inputs: &'a [Series],
    ctx: ParamCtx<'a>,
}

impl<'a> GeomParams<'a> {
    /// Parse the call's arguments as the function `name` of family `F` —
    /// strictly: an unknown or missing field, a wrong type and a per-row value
    /// for a structural field are refused, naming the field — and wrap its
    /// inputs.
    ///
    /// Checks the arguments' slots against the inputs up front, because both
    /// ways they can disagree fail badly otherwise: a slot past the end would
    /// panic on a raw index, and an input no slot claims means an operand or
    /// parameter was dropped between the builder and here — a quietly wrong
    /// result rather than an error. The slots come from the definition's
    /// derived visitor, so there is no list of them to keep.
    pub fn parse<F: WireOps>(
        inputs: &'a [Series],
        kwargs: GeomKwargs,
        name: &str,
    ) -> PolarsResult<(F, Self)> {
        let op = F::from_wire(name, serde_json::Value::Object(kwargs.args))
            .ok_or_else(|| polars_err!(ComputeError: "'{}' is not a function of its family", name))?
            .map_err(|e| polars_err!(ComputeError: "{}: {}", name, e))?;
        let mut claimed: Vec<usize> = Vec::new();
        let mut bad: Option<(&'static str, usize)> = None;
        op.visit_slots(&mut |field, slot| {
            if slot == 0 || slot >= inputs.len() {
                bad.get_or_insert((field, slot));
            }
            claimed.push(slot);
        });
        if let Some((field, slot)) = bad {
            polars_bail!(ComputeError:
                "'{}' reads input {} but the call has {} inputs; the expression \
                 was built by an incompatible version",
                field, slot, inputs.len()
            );
        }
        // Index 0 is the namespace's own column; every other input must be
        // claimed exactly once.
        claimed.sort_unstable();
        if claimed != (1..inputs.len()).collect::<Vec<_>>() {
            polars_bail!(ComputeError:
                "call has {} inputs but its arguments read {:?}; every operand \
                 and per-row parameter must be passed exactly once",
                inputs.len(), claimed
            );
        }
        let params = GeomParams {
            inputs,
            ctx: ParamCtx::with_null_policy(inputs, kwargs.on_null.get()),
        };
        Ok((op, params))
    }

    /// Resolve one row's parameters, applying the call's [`NullParamPolicy`].
    ///
    /// Returns `Ok(None)` when `f` failed *because* a per-row parameter was null
    /// and the policy asks for a null result — the caller pushes its null for
    /// that row, exactly as it already does for null input data. Every other
    /// error propagates unchanged.
    ///
    /// Wrapping resolution in this one helper is what keeps the policy a shared
    /// mechanism: no geometry function re-implements null handling.
    pub fn row<T>(&self, f: impl FnOnce() -> PolarsResult<T>) -> PolarsResult<Option<T>> {
        self.ctx.clear_null();
        match f() {
            Ok(value) => Ok(Some(value)),
            Err(_) if self.ctx.took_null() => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// A parameter's value at `row`.
    pub fn value<T: WireScalar>(&self, param: &Param<T>, row: usize) -> PolarsResult<T> {
        param.resolve(row, &self.ctx)
    }

    /// A data operand's input series.
    pub fn column(&self, column: &ColumnRef) -> &'a Series {
        &self.inputs[column.0]
    }

    /// An optional data operand's input series, when the caller gave one.
    pub fn optional_column(&self, column: &Option<ColumnRef>) -> Option<&'a Series> {
        column.as_ref().map(|c| self.column(c))
    }
}

/// The error for a function whose arguments parsed as another function of
/// its family — impossible, since [`GeomParams::parse`] parses by the
/// function's own name, but refused rather than panicked on.
pub fn parsed_as_another(name: &str) -> PolarsError {
    polars_err!(ComputeError: "internal: '{}' parsed as another function", name)
}

/// Validate a resolved parameter that must lie within an inclusive range.
///
/// Per-row parameters cannot be range-checked once per batch, so the check
/// moves into the row loop and names the offending row.
pub fn check_range(name: &str, value: f64, lo: f64, hi: f64, row: usize) -> PolarsResult<()> {
    if !(lo..=hi).contains(&value) {
        polars_bail!(ComputeError:
            "{} must be in [{}, {}], got {} at row {}", name, lo, hi, value, row
        );
    }
    Ok(())
}
