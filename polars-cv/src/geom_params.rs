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
/// Built once before the row loop; every lookup inside the loop is an indexed
/// read.
pub struct GeomParams<'a> {
    ctx: ParamCtx<'a>,
    slots: &'a InputSlots,
}

impl<'a> GeomParams<'a> {
    /// Wrap a call's inputs and its `input_slots` map.
    pub fn new(inputs: &'a [Series], slots: &'a InputSlots) -> Self {
        GeomParams {
            ctx: ParamCtx::from_inputs(inputs),
            slots,
        }
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
