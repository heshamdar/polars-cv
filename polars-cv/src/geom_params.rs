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

use crate::graph::RowErrorPolicy;
use crate::ops::{ColumnRef, Literal, Param};
use crate::params::{NullParamPolicy, ParamCtx};

/// A geometry plugin call's kwargs: the function's own wire fields, and the
/// null and error policies (`_GeomNamespace.on_null` / `.on_error`) every
/// call carries.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeomKwargs {
    args: serde_json::Map<String, serde_json::Value>,
    on_null: Literal<NullParamPolicy>,
    on_error: Literal<RowErrorPolicy>,
}

/// Per-row resolver over one plugin call's inputs.
pub struct GeomParams<'a> {
    inputs: &'a [Series],
    policy: NullParamPolicy,
    on_error: RowErrorPolicy,
    ctx: ParamCtx<'a>,
}

/// What a [`GeomParams`] is made from, shareable across threads.
///
/// A `GeomParams` carries its row's null-parameter flag in a `Cell`, so it
/// belongs to one thread; a call split over the pool
/// ([`run_split`](crate::row_split::run_split)) hands each row range this
/// instead, and the range makes its own with [`params`](Self::params).
#[derive(Clone, Copy)]
pub struct SharedParams<'a> {
    inputs: &'a [Series],
    policy: NullParamPolicy,
    on_error: RowErrorPolicy,
}

impl<'a> SharedParams<'a> {
    /// A `GeomParams` of its own, for one thread's rows.
    pub fn params(&self) -> GeomParams<'a> {
        GeomParams {
            inputs: self.inputs,
            policy: self.policy,
            on_error: self.on_error,
            ctx: ParamCtx::with_null_policy(self.inputs, self.policy),
        }
    }
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
        let on_error = kwargs.on_error.get();
        if on_error == RowErrorPolicy::NullWithMessage {
            polars_bail!(ComputeError:
                "{}: on_error='null_with_message' needs a struct output to carry \
                 the message, which a geometry function does not have; use 'null'",
                name
            );
        }
        let shared = SharedParams {
            inputs,
            policy: kwargs.on_null.get(),
            on_error,
        };
        Ok((op, shared.params()))
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
    /// Run `row(params, i)` for every row `0..len`, split over the plugin's
    /// thread pool ([`run_split`](crate::row_split::run_split)), and return
    /// the rows in order.
    ///
    /// **The one row loop of the geometry functions.** Each row range
    /// resolves parameters through its own `GeomParams` (the null flag is per
    /// thread), and each row runs under [`row`](Self::row): `Ok(None)` is a
    /// null row (a null input), and so is a null per-row parameter under
    /// `on_null="null"`. The first failing range's error is the earliest
    /// failing row's, as a sequential loop would report.
    pub fn map_rows<T: Send>(
        &self,
        len: usize,
        row: impl Fn(&GeomParams, usize) -> PolarsResult<Option<T>> + Sync,
    ) -> PolarsResult<Vec<Option<T>>> {
        let shared = self.shared();
        let parts = crate::row_split::run_split(len, |_, range| {
            let params = shared.params();
            range
                .map(|i| params.row(|| row(&params, i)).map(Option::flatten))
                .collect::<PolarsResult<Vec<_>>>()
        });
        let mut rows = Vec::with_capacity(len);
        for part in parts {
            rows.extend(part?);
        }
        Ok(rows)
    }

    /// What another thread's rows need to make their own `GeomParams`.
    pub fn shared(&self) -> SharedParams<'a> {
        SharedParams {
            inputs: self.inputs,
            policy: self.policy,
            on_error: self.on_error,
        }
    }

    pub fn row<T>(&self, f: impl FnOnce() -> PolarsResult<T>) -> PolarsResult<Option<T>> {
        self.ctx.clear_null();
        match f() {
            Ok(value) => Ok(Some(value)),
            Err(_) if self.ctx.took_null() => Ok(None),
            // A row's data refused under `on_error="null"`: that row is null.
            // A `SchemaMismatch` is about the column — its arity or layout —
            // which no row could get past, so it raises under every policy
            // rather than nulling the whole column ([`column_error`]).
            Err(e)
                if self.on_error == RowErrorPolicy::Null
                    && !matches!(e, PolarsError::SchemaMismatch(_)) =>
            {
                Ok(None)
            }
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

    /// A data operand's row `row` as a value, after scalar broadcasting (a
    /// one-row operand is every row's), through [`ParamCol::at`] — for an
    /// operand no geometry reader reads (`correspond`'s `order`).
    ///
    /// [`ParamCol::at`]: crate::params::ParamCol::at
    pub fn operand_value(&self, column: &ColumnRef, row: usize) -> PolarsResult<AnyValue<'a>> {
        let (series, idx) = self.ctx.col(column.0)?.at(row);
        series.get(idx)
    }

    /// An optional data operand's input series, when the caller gave one.
    pub fn optional_column(&self, column: &Option<ColumnRef>) -> Option<&'a Series> {
        column.as_ref().map(|c| self.column(c))
    }
}

/// An error about an operand **column** — an arity a function cannot take, a
/// layout no reader understands — rather than one row's data. It raises under
/// every `on_error` policy ([`GeomParams::row`]): no row of such a column could
/// succeed, so nulling it would turn a broken query into an all-null result.
pub fn column_error(msg: impl std::fmt::Display) -> PolarsError {
    PolarsError::SchemaMismatch(msg.to_string().into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom_fns::PointFn;
    use view_buffer::mode::Wire;

    fn params<'a>(inputs: &'a [Series], on_error: &str) -> PolarsResult<GeomParams<'a>> {
        let kwargs: GeomKwargs = serde_json::from_value(serde_json::json!({
            "args": {},
            "on_null": "raise",
            "on_error": on_error,
        }))
        .expect("the kwargs envelope parses");
        GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, "point_x").map(|(_, p)| p)
    }

    fn column() -> [Series; 1] {
        [Series::new("p".into(), [1.0f64])]
    }

    /// Under `"null"` a row's own error is that row's null; under `"raise"`
    /// it fails the call.
    #[test]
    fn a_row_error_is_null_only_under_null() {
        let inputs = column();
        let failing = || -> PolarsResult<()> { polars_bail!(ComputeError: "bad row") };
        assert!(params(&inputs, "raise").unwrap().row(failing).is_err());
        assert_eq!(params(&inputs, "null").unwrap().row(failing).unwrap(), None);
    }

    /// An error about the column is no row's to null: it raises under both.
    #[test]
    fn a_column_error_raises_under_every_policy() {
        let inputs = column();
        for policy in ["raise", "null"] {
            let result = params(&inputs, policy)
                .unwrap()
                .row(|| -> PolarsResult<()> { Err(column_error("a contour set")) });
            assert!(
                matches!(result, Err(PolarsError::SchemaMismatch(_))),
                "{policy}"
            );
        }
    }

    /// An accessor's output has no struct to carry `_error` in.
    #[test]
    fn null_with_message_is_refused() {
        let inputs = column();
        let err = params(&inputs, "null_with_message").err().expect("refused");
        assert!(err.to_string().contains("null_with_message"), "{err}");
    }
}
