//! Per-row parameter resolution for the geometry namespaces.
//!
//! The `.contour` / `.point` / `.bbox` accessors are standalone
//! `#[polars_expr]` functions rather than `vb_graph` graph nodes, but they use
//! the same per-row mechanism as every typed op: a kwarg is a
//! [`Param<T>`](crate::ops::Param) — the literal value, or `{"$slot": n}`
//! naming the plugin input that holds it per row — and an optional data
//! operand (`point.rotate`'s `origin`, `correspond`'s `order`) is a
//! [`ColumnRef`]. Python's `_ArgBinder` appends each expression as an input and
//! writes its position into the kwarg itself, so nothing is looked up by name.
//!
//! Reading is delegated to [`crate::params::ParamCol`], so these namespaces
//! inherit the same dtype coverage, scalar broadcasting (a length-1 series from
//! an aggregation applies to every row) and [`NullParamPolicy`] the graph
//! engine uses. The policy arrives as an `on_null` kwarg (set from Python by
//! `_GeomNullPolicy.on_null`) and is applied by [`GeomParams::row`], which each
//! row loop wraps its parameter resolution in.

#[allow(unused_imports)]
use crate::ops::ParamExt as _;
use polars::prelude::*;
use view_buffer::naming::WireScalar;

use crate::ops::{ColumnRef, Literal, OpFields, Param};
use crate::params::{NullParamPolicy, ParamCtx};

/// Per-row resolver over one plugin call's inputs.
pub struct GeomParams<'a> {
    inputs: &'a [Series],
    ctx: ParamCtx<'a>,
}

impl<'a> GeomParams<'a> {
    /// Wrap a call's inputs and the kwargs that address them.
    ///
    /// Checks the kwargs' slots against the inputs up front, because both ways
    /// they can disagree fail badly otherwise: a slot past the end would panic
    /// on a raw index, and an input no slot claims means an operand or
    /// parameter was dropped between the builder and here — a quietly wrong
    /// result rather than an error. The slots come from the kwargs' derived
    /// visitor, so there is no list of them to keep.
    pub fn new<K: OpFields>(
        inputs: &'a [Series],
        kwargs: &K,
        on_null: Option<Literal<NullParamPolicy>>,
    ) -> PolarsResult<Self> {
        let mut claimed: Vec<usize> = Vec::new();
        let mut bad: Option<(&'static str, usize)> = None;
        kwargs.visit_slots(&mut |name, slot| {
            if slot == 0 || slot >= inputs.len() {
                bad.get_or_insert((name, slot));
            }
            claimed.push(slot);
        });
        if let Some((name, slot)) = bad {
            polars_bail!(ComputeError:
                "'{}' reads input {} but the call has {} inputs; the expression \
                 was built by an incompatible version",
                name, slot, inputs.len()
            );
        }
        // Index 0 is the namespace's own column; every other input must be
        // claimed exactly once.
        claimed.sort_unstable();
        if claimed != (1..inputs.len()).collect::<Vec<_>>() {
            polars_bail!(ComputeError:
                "call has {} inputs but its kwargs read {:?}; every operand and \
                 per-row parameter must be passed exactly once (see `_ArgBinder`)",
                inputs.len(), claimed
            );
        }
        Ok(GeomParams {
            inputs,
            ctx: ParamCtx::with_null_policy(inputs, on_null.map(|p| p.get()).unwrap_or_default()),
        })
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

    /// A parameter's value at `row`, or `default` when the caller gave none.
    pub fn get<T: WireScalar>(
        &self,
        param: &Option<Param<T>>,
        default: T,
        row: usize,
    ) -> PolarsResult<T> {
        match param {
            Some(p) => p.resolve(row, &self.ctx),
            None => Ok(default),
        }
    }

    /// A required parameter's value at `row`.
    pub fn required<T: WireScalar>(
        &self,
        param: &Option<Param<T>>,
        name: &str,
        row: usize,
    ) -> PolarsResult<T> {
        match param {
            Some(p) => p.resolve(row, &self.ctx),
            None => polars_bail!(ComputeError: "{} is required", name),
        }
    }

    /// The input series of a data operand, when the caller gave one.
    pub fn column(&self, column: &Option<ColumnRef>) -> Option<&'a Series> {
        column.map(|c| &self.inputs[c.0])
    }

    /// The input series of a required data operand.
    pub fn required_column(
        &self,
        column: &Option<ColumnRef>,
        name: &str,
    ) -> PolarsResult<&'a Series> {
        self.column(column)
            .ok_or_else(|| polars_err!(ComputeError: "missing required input '{}'", name))
    }
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
