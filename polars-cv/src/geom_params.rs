//! Per-row parameter resolution for the geometry namespaces.
//!
//! The `.contour` / `.point` / `.bbox` accessors are standalone
//! `#[polars_expr]` functions rather than `vb_graph` graph nodes, so they have
//! none of [`crate::params::ParamValue`]'s literal-vs-expression machinery.
//! Their per-row channel is the plugin's *input series*: Python appends an
//! expression-valued parameter as an extra argument and records its index in
//! the `input_slots` kwarg.
//!
//! Position alone is not enough to identify those inputs, because several of
//! these functions already read *optional* data operands positionally
//! (`point_rotate`'s `origin`, `bbox_match_detections`' `scores`) — an appended
//! parameter would be indistinguishable from an omitted operand. Looking every
//! variable input up by name removes the ambiguity.
//!
//! Reading is delegated to [`crate::params::ParamCol`], so these namespaces
//! inherit the same dtype coverage, scalar broadcasting (a length-1 series from
//! an aggregation applies to every row) and null-as-error policy the graph
//! engine already uses.

use std::collections::HashMap;

use polars::prelude::*;

use crate::params::ParamCtx;

/// Named indices into a plugin's `inputs` slice.
///
/// Empty for a call with no expression parameters and no optional operands,
/// which reproduces the original all-literal behaviour.
pub type InputSlots = HashMap<String, usize>;

/// Per-row resolver over one plugin call's inputs.
///
/// Built once before the row loop. A lookup inside the loop is a small
/// string-keyed map probe followed by an indexed column read — these functions
/// are dominated by contour parsing, so resolving by name rather than hoisting
/// literals out of the loop is not a measurable cost.
pub struct GeomParams<'a> {
    ctx: ParamCtx<'a>,
    slots: &'a InputSlots,
}

impl<'a> GeomParams<'a> {
    /// Wrap a call's inputs and its `input_slots` map.
    ///
    /// Validates the map against the inputs up front, because both ways it can
    /// be wrong fail silently or violently otherwise: an index past the end
    /// would panic on a raw `inputs[idx]`, and a map that does not account for
    /// every extra input means an operand or parameter was dropped somewhere
    /// between the builder and here — which would compute a quietly wrong
    /// result rather than fail.
    pub fn new(inputs: &'a [Series], slots: &'a InputSlots) -> PolarsResult<Self> {
        for (name, &idx) in slots {
            if idx == 0 || idx >= inputs.len() {
                polars_bail!(ComputeError:
                    "input slot '{}' points at index {} but the call has {} inputs; \
                     the expression was built by an incompatible version",
                    name, idx, inputs.len()
                );
            }
        }
        // Index 0 is the namespace's own column; every other input must be
        // claimed by exactly one name.
        if slots.len() + 1 != inputs.len() {
            polars_bail!(ComputeError:
                "call has {} inputs but 'input_slots' names {}; every operand and \
                 per-row parameter must be registered (see `_ArgBinder`)",
                inputs.len(), slots.len()
            );
        }
        Ok(GeomParams {
            ctx: ParamCtx::from_inputs(inputs),
            slots,
        })
    }

    /// The input series index bound to `name`, if any.
    pub fn slot(&self, name: &str) -> Option<usize> {
        self.slots.get(name).copied()
    }

    /// Resolve a float parameter: the bound input at `row`, else the literal
    /// kwarg, else `default`.
    pub fn f64(
        &self,
        name: &str,
        literal: Option<f64>,
        default: f64,
        row: usize,
    ) -> PolarsResult<f64> {
        match self.slot(name) {
            Some(idx) => self.ctx.col(idx)?.get_f64(row),
            None => Ok(literal.unwrap_or(default)),
        }
    }

    /// Resolve a required float parameter: the bound input at `row`, else the
    /// literal kwarg, erroring when the caller supplied neither.
    pub fn required_f64(&self, name: &str, literal: Option<f64>, row: usize) -> PolarsResult<f64> {
        match self.slot(name) {
            Some(idx) => self.ctx.col(idx)?.get_f64(row),
            None => literal.ok_or_else(|| polars_err!(ComputeError: "{} is required", name)),
        }
    }

    /// Resolve a string parameter from a bound input, else the literal kwarg.
    ///
    /// For enum-valued parameters with no shape or dtype effect. A bound input
    /// must be a genuine String column; the callers pass the result to the
    /// same `parse` used for the literal form, so an unknown value produces
    /// the same error either way.
    pub fn str_opt(
        &self,
        name: &str,
        literal: Option<&'a str>,
        row: usize,
    ) -> PolarsResult<Option<&'a str>> {
        match self.slot(name) {
            Some(idx) => self.ctx.col(idx)?.get_str(row).map(Some),
            None => Ok(literal),
        }
    }

    /// Resolve a boolean parameter from a bound input, else the literal kwarg.
    ///
    /// A bound input must be a genuine Boolean column, matching how the graph
    /// engine's `get::opt_bool_dyn` treats flags: silently accepting a numeric
    /// column would turn a mis-routed expression into a wrong result rather
    /// than an error.
    pub fn bool(&self, name: &str, literal: bool, row: usize) -> PolarsResult<bool> {
        match self.slot(name) {
            Some(idx) => self.ctx.col(idx)?.get_bool(row),
            None => Ok(literal),
        }
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
