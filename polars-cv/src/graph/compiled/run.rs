//! Running a compiled graph over one call's columns: the row loop, each
//! node's steps (fused engine segments and their per-layout plan cache), and
//! the shape a node's buffer carries between steps (`Carried`).

use polars::prelude::*;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::graph::node_output::NodeOutput;
use view_buffer::ops::Domain;
use view_buffer::{ViewBuffer, ViewDto, ViewExpr};

use crate::formats::source::Source;
use crate::params::ParamCtx;

use crate::graph::row::{run_step, RowOutputs};
use crate::graph::step::GraphStep;

use crate::graph::sink::{
    build_series_from_spec, encode_node_output, null_row_result_for_spec, RowParts,
};
use crate::graph::source::{decode_source_row, RowFetch};
use crate::graph::types::{OutputSpec, OutputValue, RowErrorPolicy, RowResult};

use super::compile::{
    resolved_output_specs, CompiledGraph, NodePlan, OpResolver, PlannedDecode, PlannedInput,
};

/// Per-call execution state: everything derived from the actual input series.
pub(super) struct ExecState<'a> {
    pub(super) inputs: &'a [Series],
    /// How many rows this call (one batch under streaming) holds.
    pub(super) rows: usize,
    /// Output specs with `"auto"` dtype/ndim resolved from the input column
    /// type, sorted by alias.
    pub(super) resolved_outputs: Vec<(String, OutputSpec)>,
    /// Each path-reading (`file_path`, or `"auto"` over a `String` column)
    /// node's fetch window over this call's column, aligned with `plan`.
    /// Per-path errors surface at their row so the usual error policies
    /// apply.
    pub(super) fetchers: Vec<Option<crate::fetch::Fetcher<'a>>>,
    /// The concrete source each `"auto"` source node reads this batch's column
    /// as ([`Source::route`]), aligned with `plan`. The column dtype is
    /// constant across rows, so this is taken once per batch; a routing error
    /// is stored and surfaced at its row so the usual error policies apply.
    /// `None` for a concrete source.
    pub(super) routed_sources: Vec<Option<Result<Source, String>>>,
    /// Position in `plan` of each resolved output's node (aligned with
    /// `resolved_outputs`).
    pub(super) output_nodes: Vec<Option<usize>>,
}

impl CompiledGraph {
    /// Execute the graph on input series.
    ///
    /// Returns:
    /// - Binary/typed column if single output ("_output" only)
    /// - Struct column with named fields if multiple outputs
    pub fn execute(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            return Err(polars_err!(ComputeError : "No input columns provided"));
        };
        // Slots and column bindings are compile-time data but their bounds
        // depend on this call's inputs: check once here instead of per row.
        if inputs.len() < self.min_inputs {
            return Err(polars_err!(ComputeError:
                "graph reads {} input columns but the call supplied {}",
                self.min_inputs, inputs.len()
            ));
        }
        // A contour column no row can be read as is the column's error, under
        // every policy: `on_error("null")` would otherwise null each row.
        for np in &self.plan {
            if let Some(PlannedDecode {
                source: Source::Contour { .. },
                column: col,
                ..
            }) = np.input.decode()
            {
                if inputs[*col].dtype() != &DataType::Null {
                    crate::geometry::columns::ContourColumn::new(&inputs[*col]).check_layout()?;
                }
            }
        }
        let resolved_outputs = resolved_output_specs(
            &self.graph,
            &inputs.iter().map(|s| s.dtype().clone()).collect::<Vec<_>>(),
        )?;
        let output_nodes = resolved_outputs
            .iter()
            .map(|(_, spec)| self.node_index.get(&spec.node).copied())
            .collect();
        let state = ExecState {
            inputs,
            rows: len,
            resolved_outputs,
            fetchers: self.fetchers(inputs),
            routed_sources: self.route_auto_sources(inputs),
            output_nodes,
        };

        // Rows are independent: the call's rows run in contiguous ranges on
        // this thread and on whatever pool threads are idle
        // (`row_split::split`, shared with the geometry accessors).
        let plan_cache = PlanCache::new(&self.plan);
        let first_failure = AtomicUsize::new(usize::MAX);
        let run_range = |range_idx: usize, rows: Range<usize>| -> RangeOutcome {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.execute_rows(&state, rows, range_idx, &plan_cache, &first_failure)
            }))
            .map_err(|payload| {
                polars_err!(ComputeError : "Pipeline batch failed: {}",
                    panic_message(payload.as_ref()))
            })?
            .map_err(|msg| polars_err!(ComputeError : "Pipeline execution failed: {}", msg))
        };
        // Held until the columns are built: filling them splits the same way.
        let split = crate::row_split::split();
        let outcomes = split.run(len, run_range);

        // Concatenate in row order. Under `on_error="raise"` the first failing
        // range holds the earliest failing row, so its error is the one a
        // sequential run would have reported. Each output keeps its ranges'
        // parts, which the column builder converts in order: concatenating
        // them first was a serial, call-sized copy of every row result.
        let mut results: Vec<RowParts> = (0..state.resolved_outputs.len())
            .map(|_| Vec::new())
            .collect();
        let mut error_messages: Vec<Option<String>> = Vec::new();
        for outcome in outcomes {
            let (range_results, range_messages) = outcome?;
            for (parts, part) in results.iter_mut().zip(range_results) {
                parts.push(part);
            }
            error_messages.extend(range_messages);
        }

        let with_message = self.graph.on_error == RowErrorPolicy::NullWithMessage;
        if self.graph.is_single_output() && !with_message {
            let (_, spec) = &state.resolved_outputs[0];
            // Take ownership of the row results so the encoder can move each
            // row's bytes/buffers into Arrow instead of copying them.
            let data = results.swap_remove(0);
            build_series_from_spec(inputs[0].name().clone(), spec, data, Some(&split))
        } else {
            let mut fields: Vec<Series> = Vec::with_capacity(state.resolved_outputs.len() + 1);
            for ((alias, spec), data) in state.resolved_outputs.iter().zip(results) {
                let field_series =
                    build_series_from_spec(PlSmallStr::from_str(alias), spec, data, Some(&split))?;
                fields.push(field_series);
            }
            if with_message {
                fields.push(Series::new(
                    PlSmallStr::from_static("_error"),
                    error_messages,
                ));
            }
            let output_name = inputs[0].name().clone();
            StructChunked::from_series(output_name, len, fields.iter()).map(|sc| sc.into_series())
        }
    }

    /// The per-row loop over one row range: decode sources, run ops, encode
    /// outputs.
    ///
    /// Applies the graph's [`RowErrorPolicy`] to per-row errors and returns
    /// this range's rows (one vector per resolved output) plus one
    /// error-message slot per row when the policy is `NullWithMessage` (an
    /// empty Vec otherwise). Errors are `String` so the surrounding
    /// `catch_unwind`/`PolarsError` wrapping stays in one place.
    ///
    /// Under `Raise` a failing range records its index in `first_failure`,
    /// and a range after it stops early: its rows would be discarded, since
    /// the earlier error is the one reported.
    pub(super) fn execute_rows(
        &self,
        state: &ExecState<'_>,
        rows: Range<usize>,
        range_idx: usize,
        plan_cache: &PlanCache,
        first_failure: &AtomicUsize,
    ) -> Result<RangeRows, String> {
        let policy = self.graph.on_error;
        let with_message = policy == RowErrorPolicy::NullWithMessage;
        let mut results: Vec<Vec<RowResult>> = (0..state.resolved_outputs.len())
            .map(|_| Vec::with_capacity(rows.len()))
            .collect();
        let mut error_messages: Vec<Option<String>> = if with_message {
            Vec::with_capacity(rows.len())
        } else {
            Vec::new()
        };
        // Per range, not per call: `ParamCtx` carries this thread's
        // null-parameter flag in a `Cell`.
        let ctx = ParamCtx::with_null_policy(state.inputs, self.graph.on_null_param);
        // Allocated once and reused across rows/nodes to avoid per-row churn.
        let mut node_outputs = RowOutputs::new(self.plan.len());
        let mut dto_scratch: Vec<Cow<'_, GraphStep>> = Vec::new();
        let start = rows.start;
        for row_idx in rows {
            if first_failure.load(Ordering::Relaxed) < range_idx {
                break;
            }
            node_outputs.clear();
            // Panics are caught per row, so they reach the row policy like any
            // other row error. view-buffer reports some data-dependent failures
            // (e.g. operands that cannot broadcast) by panicking. Caught only
            // once per call, one such row failed the whole batch even under
            // `on_error="null"`, and under streaming how much of the query it
            // took down depended on the morsel size. The per-row state
            // is rebuilt from scratch every row (`node_outputs` is reset, and the
            // null policies truncate `results` back to this row), so nothing a
            // panicking row half-wrote survives it.
            let row_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.execute_one_row(
                    state,
                    &ctx,
                    row_idx,
                    &mut node_outputs,
                    &mut dto_scratch,
                    plan_cache,
                    &mut results,
                )
            }))
            .unwrap_or_else(|payload| {
                // Labelled so a user (and the no-panic sweep in
                // `tests/test_engine_no_panics.py`) can tell an engine bug
                // from an input the engine rejected with a proper error.
                Err(format!(
                    "internal error: the engine panicked: {}",
                    panic_message(payload.as_ref())
                ))
            });
            match row_result {
                Ok(()) => {
                    if with_message {
                        error_messages.push(None);
                    }
                }
                Err(msg) => match policy {
                    RowErrorPolicy::Raise => {
                        first_failure.fetch_min(range_idx, Ordering::Relaxed);
                        // The row, named here only (`row_ref`).
                        return Err(crate::row_ref::at_row(msg, row_idx, state.rows));
                    }
                    RowErrorPolicy::Null | RowErrorPolicy::NullWithMessage => {
                        // All-or-nothing per row: drop anything this row may
                        // have pushed before failing, then null every output.
                        for ((_, spec), out) in
                            state.resolved_outputs.iter().zip(results.iter_mut())
                        {
                            out.truncate(row_idx - start);
                            out.push(null_row_result_for_spec(spec).map_err(|e| e.to_string())?);
                        }
                        if with_message {
                            error_messages.push(Some(msg));
                        }
                    }
                },
            }
        }
        Ok((results, error_messages))
    }

    /// Execute every node and encode every output for one row.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_one_row<'g>(
        &'g self,
        state: &ExecState<'_>,
        ctx: &ParamCtx<'_>,
        row_idx: usize,
        node_outputs: &mut RowOutputs,
        dto_scratch: &mut Vec<Cow<'g, GraphStep>>,
        plan_cache: &PlanCache,
        results: &mut [Vec<RowResult>],
    ) -> Result<(), String> {
        #[cfg(test)]
        {
            // A held thread's rows wait here, before they are recorded, so
            // a held call stays running without counting as a row thread.
            let held = self.held.lock().unwrap();
            let me = std::thread::current().id();
            if held.contains(&me) {
                self.held_rows.fetch_add(1, Ordering::SeqCst);
                drop(self.released.wait_while(held, |h| h.contains(&me)).unwrap());
                self.held_rows.fetch_sub(1, Ordering::SeqCst);
            } else {
                drop(held);
            }
            self.rendezvous.visit();
        }
        self.run_row_nodes(state, ctx, row_idx, node_outputs, dto_scratch, plan_cache)?;
        for (((alias, spec), node), rows) in state
            .resolved_outputs
            .iter()
            .zip(&state.output_nodes)
            .zip(results.iter_mut())
        {
            if let Some(output) = node.and_then(|i| node_outputs.output(i)) {
                validate_output_schema(alias, spec, output)?;
                match encode_node_output(output, spec) {
                    Ok(encoded) => {
                        let row_result = match encoded {
                            OutputValue::Binary(bytes) => RowResult::Binary(Some(bytes)),
                            OutputValue::Scalar(val) => RowResult::Scalar(Some(val)),
                            OutputValue::Vector(vals) => RowResult::Vector(Some((*vals).clone())),
                            OutputValue::Contours(contours) => {
                                RowResult::Contours(Some((*contours).clone()))
                            }
                            OutputValue::TypedList(buf) => RowResult::TypedList(Some(buf)),
                            OutputValue::TypedArray(buf) => RowResult::TypedArray(Some(buf)),
                            OutputValue::NumpyStruct(buf) => RowResult::NumpyStruct(Some(buf)),
                            OutputValue::Long(row) => RowResult::Long(Some(row)),
                            OutputValue::HistogramBuckets(buckets) => {
                                RowResult::HistogramBuckets(Some(buckets))
                            }
                        };
                        rows.push(row_result);
                    }
                    Err(e) => {
                        return Err(format!("Encode error for '{alias}': {e}"));
                    }
                }
            } else {
                let null_result = null_row_result_for_spec(spec).map_err(|e| e.to_string())?;
                rows.push(null_result);
            }
        }
        Ok(())
    }

    /// Run every node of the graph for one row, leaving each node's output in
    /// `node_outputs` (no output encoding).
    pub(super) fn run_row_nodes<'g>(
        &'g self,
        state: &ExecState<'_>,
        ctx: &ParamCtx<'_>,
        row_idx: usize,
        node_outputs: &mut RowOutputs,
        dto_scratch: &mut Vec<Cow<'g, GraphStep>>,
        plan_cache: &PlanCache,
    ) -> Result<(), String> {
        let inputs = state.inputs;
        // Labelled so a null per-row parameter or a null operand can skip
        // straight to the next node: leaving this node without an output is
        // exactly how a null *input* already propagates.
        'nodes: for (idx, np) in self.plan.iter().enumerate() {
            let node_input: Option<NodeOutput> = match &np.input {
                PlannedInput::Decode(decode) => {
                    let PlannedDecode {
                        column,
                        source,
                        source_null,
                        ..
                    } = decode.as_ref();
                    // An `"auto"` source reads as the concrete source it was
                    // routed to once per batch (`route_auto_sources`).
                    let decode_result = match state.routed_sources[idx].as_ref() {
                        Some(Ok(routed)) => Ok(routed),
                        Some(Err(e)) => Err(e.clone()),
                        None => Ok(source),
                    }
                    .and_then(|source| {
                        // Bounds are validated once per call in `execute()`.
                        decode_source_row(
                            &np.id,
                            source,
                            &inputs[*column],
                            row_idx,
                            RowFetch {
                                fetcher: state.fetchers[idx].as_ref(),
                            },
                        )
                    });
                    match decode_result {
                        Ok(output) => output,
                        Err(_e) if *source_null => None,
                        Err(e) => return Err(e),
                    }
                }
                PlannedInput::Node { upstream, take } => node_outputs.input(*upstream, *take),
            };
            let Some(input) = node_input else {
                continue;
            };
            // Static ops are borrowed from the compiled graph; dynamic ops are
            // re-resolved for this row through typed slot reads. The scratch
            // Vec is reused so steady-state rows allocate nothing here.
            dto_scratch.clear();
            // The shape each step reads, for a node whose `this` parameters
            // read it: the input's, then each resolved step's output over it
            // (`None` once it is not known before the step runs).
            let mut carried = np.reads_shape.then(|| Carried::of(&input));
            for step in &np.steps {
                match &step.resolver {
                    OpResolver::Static(resolved) => dto_scratch.push(Cow::Borrowed(resolved)),
                    OpResolver::Dynamic(spec) => {
                        ctx.clear_null();
                        let shape = carried.as_ref().and_then(|c| c.shape.as_deref());
                        match spec.resolve_on(row_idx, ctx, shape) {
                            Ok(resolved) => dto_scratch.push(Cow::Owned(resolved)),
                            // Under `on_null_param="null"` a null parameter is
                            // not a failure — this node just has no output for
                            // this row.
                            Err(_) if ctx.took_null() => continue 'nodes,
                            // The op named once (`ResolveError::named`).
                            Err(e) => {
                                return Err(format!("Op resolution error: {}", e.named(step.name)))
                            }
                        }
                    }
                }
                if let Some(c) = carried.as_mut() {
                    let resolved = dto_scratch.last().expect("the step was pushed above");
                    *c = c.after(resolved, node_outputs, &step.operands);
                }
            }
            let mut current_output = input;
            // Engine-tier optimization toggles for this graph (Tier-2).
            // `OptConfig` is `Copy`, so the closure captures a value and does
            // not borrow `self`.
            let opt_cfg = self.graph.opt;
            let node_cache = &plan_cache.slots[idx];
            let flush_buffer_ops = |output: NodeOutput,
                                    pending: &mut PendingSegment<'_>|
             -> Result<NodeOutput, String> {
                let slot = match pending.start {
                    Some(start) if pending.cacheable => Some(&node_cache[start]),
                    _ => None,
                };
                let result = run_segment(output, &pending.ops, slot, &opt_cfg);
                pending.clear();
                let (output, _planned) = result?;
                #[cfg(test)]
                if _planned {
                    self.plan_builds.fetch_add(1, Ordering::Relaxed);
                }
                Ok(output)
            };
            let mut pending_buffer_ops = PendingSegment::default();
            for (step_idx, (resolved, compiled)) in dto_scratch.iter().zip(&np.steps).enumerate() {
                match resolved.as_ref() {
                    // Fusable single-buffer engine ops accumulate and run as
                    // one ViewExpr chain at the next flush.
                    GraphStep::Buffer(dto) => {
                        let is_static = matches!(resolved, Cow::Borrowed(_));
                        pending_buffer_ops.push(step_idx, dto, is_static);
                    }
                    step => {
                        current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                        let Some(operands) = node_outputs.operands(&compiled.operands) else {
                            continue 'nodes;
                        };
                        current_output = run_step(
                            step,
                            compiled.name,
                            current_output,
                            &operands,
                            ctx,
                            row_idx,
                            node_outputs.regions(),
                        )?;
                    }
                }
            }
            current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
            node_outputs.set(idx, current_output);
        }
        Ok(())
    }

    /// Route each `"auto"` source node to its concrete source once per batch
    /// ([`Source::route`]).
    ///
    /// The route depends only on the bound input column's dtype, which is
    /// constant across rows, so taking it here (rather than per row) avoids
    /// repeated work — including the O(n) magic-byte scan for `Binary` columns.
    /// Errors are stored per node and re-surfaced at their row so the batch's
    /// row-error policy still applies. Aligned with `plan`; concrete sources
    /// are `None`.
    pub(super) fn route_auto_sources(
        &self,
        inputs: &[Series],
    ) -> Vec<Option<Result<Source, String>>> {
        self.plan
            .iter()
            .map(|np| {
                let decode = np.input.decode()?;
                decode.source.route(inputs.get(decode.column)?)
            })
            .collect()
    }

    /// The fetch window of every path-reading node over this call's column.
    ///
    /// Per call (per morsel) and derived only from this call's path values —
    /// nothing here is cached on the compiled graph. Building one makes no
    /// request; rows fetch ahead as they run (`crate::fetch`). Wrong-dtype
    /// columns get none, so the row loop reports them with its usual error
    /// message. `"auto"` nodes are included too: an auto source over a
    /// `String` column resolves to `file_path`, and the `series.str()` check
    /// below skips auto nodes bound to any other column type. Aligned with
    /// `plan`.
    pub(super) fn fetchers<'a>(
        &'a self,
        inputs: &'a [Series],
    ) -> Vec<Option<crate::fetch::Fetcher<'a>>> {
        self.plan
            .iter()
            .map(|np| {
                let Some(PlannedDecode {
                    column,
                    source: Source::FilePath { .. } | Source::Auto { .. },
                    cloud_options,
                    path_policy,
                    ..
                }) = np.input.decode()
                else {
                    return None;
                };
                let ca = inputs.get(*column)?.str().ok()?;
                Some(crate::fetch::Fetcher::new(
                    ca,
                    cloud_options.as_ref(),
                    path_policy,
                ))
            })
            .collect()
    }
}

/// A run of consecutive single-buffer engine ops awaiting one fused
/// `ViewExpr` execution.
#[derive(Default)]
pub(super) struct PendingSegment<'s> {
    /// The ops, borrowed from the row's resolved steps; cloned only when the
    /// segment has to be planned. Inline, since one is built per node per row.
    pub(super) ops: smallvec::SmallVec<[&'s ViewDto; 8]>,
    /// Index of the segment's first op in the node's op list: its cache slot.
    pub(super) start: Option<usize>,
    /// Every op is static (all-literal), so its plan is the same for every
    /// row with the same source layout.
    pub(super) cacheable: bool,
}

impl<'s> PendingSegment<'s> {
    pub(super) fn push(&mut self, step_idx: usize, op: &'s ViewDto, is_static: bool) {
        if self.ops.is_empty() {
            self.start = Some(step_idx);
            self.cacheable = true;
        }
        self.cacheable &= is_static;
        self.ops.push(op);
    }

    pub(super) fn clear(&mut self) {
        self.ops.clear();
        self.start = None;
        self.cacheable = false;
    }
}

/// The planned steps of a static op segment, valid for sources of exactly
/// this dtype, shape and strides.
///
/// Those are the only facts about the source that planning reads:
/// `ViewExpr::new_source` records them, and `build_plan` consults the source
/// only for contiguity, which they determine. Data and offset are never read,
/// so replaying the steps on another buffer with the same three facts yields
/// the plan planning it afresh would.
pub(super) struct CachedPlan {
    pub(super) dtype: view_buffer::DType,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<isize>,
    /// Shared, so a hit hands the list out without copying it.
    pub(super) steps: Arc<[view_buffer::execution::PlanStep]>,
}

/// Distinct source layouts remembered per segment. A column whose rows keep
/// changing shape stops being cached once this many are held, so a call's
/// cache stays small; such segments plan per row, as an uncached segment does.
pub(super) const PLAN_CACHE_LAYOUTS: usize = 16;

/// The plans of every static buffer-op segment for one call, shared by the
/// row ranges that call runs in parallel: a segment is planned once per
/// distinct source layout per call, whichever thread meets it first.
/// Per call, so nothing data-derived outlives it.
pub(super) struct PlanCache {
    /// Per node, per segment start: the layouts planned so far.
    pub(super) slots: Vec<Vec<RwLock<Vec<CachedPlan>>>>,
}

impl PlanCache {
    pub(super) fn new(plan: &[NodePlan]) -> Self {
        PlanCache {
            slots: plan
                .iter()
                .map(|np| np.steps.iter().map(|_| RwLock::default()).collect())
                .collect(),
        }
    }
}

/// The steps planned for `source`'s layout, if `plans` holds them.
pub(super) fn cached_steps(
    plans: &[CachedPlan],
    source: &ViewBuffer,
) -> Option<Arc<[view_buffer::execution::PlanStep]>> {
    plans
        .iter()
        .find(|p| {
            p.dtype == source.dtype()
                && p.shape == source.shape()
                && p.strides == source.strides_bytes()
        })
        .map(|p| p.steps.clone())
}

/// Plan (or replay the cached plan of) one op segment and execute it.
/// Returns the output and whether the segment had to be planned.
pub(super) fn run_segment(
    output: NodeOutput,
    ops: &[&ViewDto],
    cache: Option<&RwLock<Vec<CachedPlan>>>,
    cfg: &view_buffer::OptConfig,
) -> Result<(NodeOutput, bool), String> {
    if ops.is_empty() {
        return Ok((output, false));
    }
    // Taken by value, so a buffer nothing else holds reaches the segment as
    // its sole owner and a fused kernel can write it in place. Cloning it out
    // of a still-live `output` held a second reference for the whole run,
    // and in place was never possible.
    let source = match output {
        NodeOutput::Buffer(buf) => Arc::try_unwrap(buf).unwrap_or_else(|shared| (*shared).clone()),
        other => {
            return Err(format!(
                "Expected Buffer for pending ops, got {:?}",
                other.domain()
            ))
        }
    };
    let execute = |source: ViewBuffer, steps: &[view_buffer::execution::PlanStep]| {
        NodeOutput::from_buffer(view_buffer::execution::ExecutionPlan::execute_steps(
            source, steps,
        ))
    };
    let plan = |source: ViewBuffer| -> Result<Arc<[view_buffer::execution::PlanStep]>, String> {
        let mut expr = ViewExpr::new_source(source);
        for op in ops {
            // The validated entry point: an op that cannot run on the
            // shape reaching it is this row's error, not a kernel panic.
            expr = expr
                .try_apply_op((*op).clone())
                .map_err(|e| format!("{}: {e}", op.as_op().name()))?;
        }
        Ok(expr.plan_with(cfg).steps.into())
    };
    let cached = |plans: &[CachedPlan]| cached_steps(plans, &source);

    let Some(cache) = cache else {
        let steps = plan(source.clone())?;
        return Ok((execute(source, &steps), true));
    };
    if let Some(steps) = cached(&cache.read().unwrap()) {
        return Ok((execute(source, &steps), false));
    }
    // Planned under the write lock: a range that misses on the same layout
    // meanwhile waits here and then finds this plan, rather than planning
    // its own. Only planning is serialised; execution happens after release.
    let mut plans = cache.write().unwrap();
    if let Some(steps) = cached(&plans) {
        drop(plans);
        return Ok((execute(source, &steps), false));
    }
    let steps = plan(source.clone())?;
    if plans.len() < PLAN_CACHE_LAYOUTS {
        plans.push(CachedPlan {
            dtype: source.dtype(),
            shape: source.shape().to_vec(),
            strides: source.strides_bytes().to_vec(),
            steps: Arc::clone(&steps),
        });
    }
    drop(plans);
    Ok((execute(source, &steps), true))
}

/// One row range's rows (one vector per resolved output) and error messages.
pub(super) type RangeRows = (Vec<Vec<RowResult>>, Vec<Option<String>>);
/// A row range's result, with its failure already in `PolarsError` form.
pub(super) type RangeOutcome = PolarsResult<RangeRows>;

/// The message a caught panic carried.
pub(super) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// What a node's next step is given, as its `this` parameters read it: the
/// domain, and for a buffer its shape when known before the step runs.
pub(super) struct Carried {
    pub(super) domain: Domain,
    pub(super) shape: Option<Vec<usize>>,
}

impl Carried {
    /// The node's input.
    pub(super) fn of(input: &NodeOutput) -> Self {
        Carried {
            domain: input.domain(),
            shape: input.as_buffer().map(|b| b.shape().to_vec()),
        }
    }

    /// What `step` outputs, by its domain rule and its shape rule over its
    /// input and the buffers of the nodes it reads. An input that is not a
    /// buffer has no sizes to give (`rasterize` takes its own, or its
    /// operand's); a size the rule cannot state from known ones, or a buffer
    /// input of unknown size, leaves the shape unknown.
    pub(super) fn after(&self, step: &GraphStep, outputs: &RowOutputs, operands: &[usize]) -> Self {
        let domain = step.output_domain(self.domain);
        let unknown = Carried {
            domain,
            shape: None,
        };
        if domain != Domain::Buffer {
            return unknown;
        }
        let input: Vec<view_buffer::ops::Dim> = match (self.domain, &self.shape) {
            (Domain::Buffer, Some(shape)) => shape
                .iter()
                .map(|&n| view_buffer::ops::Dim::Known(n))
                .collect(),
            (Domain::Buffer, None) => return unknown,
            _ => Vec::new(),
        };
        let mut shapes: Vec<Vec<view_buffer::ops::Dim>> = vec![input];
        let Some(read) = outputs.operands(operands) else {
            return unknown;
        };
        for operand in read {
            let Some(buffer) = operand.as_buffer() else {
                return unknown;
            };
            shapes.push(view_buffer::ops::shape_rule::known_dims(buffer.shape()));
        }
        let refs: Vec<&[view_buffer::ops::Dim]> = shapes.iter().map(Vec::as_slice).collect();
        let shape = step
            .shape()
            .dims(&refs)
            .and_then(|dims| dims.iter().map(|d| d.known()).collect::<Option<Vec<_>>>())
            .filter(|dims| !dims.is_empty());
        Carried { domain, shape }
    }
}

/// Validate that a buffer output's produced schema matches the plan.
///
/// This is the runtime `plan == data` guard: the schema the Python planner
/// inferred from the ops' view-buffer contracts (dtype, rank, per-dim shape)
/// must equal what execution actually produced. A divergence is a contract bug
/// (a rule that lies about its transform, or a source that guessed), not a data
/// condition — so it hard-errors.
///
/// Only known facts are checked (`"auto"`/`None` = genuinely unknown, skipped).
/// Restricted to buffer-domain outputs without a re-encoding: vector/scalar/
/// contour outputs ride a raw buffer whose rank differs from the logical one
/// (e.g. histogram buckets = `[n, fields]` behind a rank-1 vector), so their raw
/// buffer shape is intentionally not the planned schema.
fn validate_output_schema(
    alias: &str,
    spec: &OutputSpec,
    output: &NodeOutput,
) -> Result<(), String> {
    let Some(buf) = output.as_buffer() else {
        return Ok(());
    };

    // Dtype: the produced dtype must be one the plan allows (the exact one
    // when known, a float when only that is known).
    let expected = spec.expected_dtype;
    let actual = buf.dtype();
    if !expected.candidates().is_empty() && !expected.candidates().contains(&actual) {
        return Err(format!(
            "Output '{alias}': planned dtype {} but execution produced {actual:?}. \
             This indicates a mismatch between the planner's view-buffer contract and \
             the Rust implementation.",
            expected.as_str()
        ));
    }

    // Rank + per-dim shape, for plain buffer outputs only.
    if spec.expected_domain == Domain::Buffer && !spec.histogram_buckets {
        let actual_shape = buf.shape();
        if let Some(expected_ndim) = spec.expected_ndim {
            if actual_shape.len() != expected_ndim {
                return Err(format!(
                    "Output '{alias}': planned rank {expected_ndim} but execution \
                     produced a {}-D buffer (shape {actual_shape:?}). The planner's \
                     rank contract disagrees with the Rust implementation.",
                    actual_shape.len()
                ));
            }
        }
        if let Some(expected_shape) = spec.expected_shape.as_ref() {
            // A user's `assert_shape` is checked where it was written, so a
            // divergence here is always an op's contract disagreeing with its
            // implementation.
            if actual_shape != expected_shape.as_slice() {
                return Err(format!(
                    "Output '{alias}': planned shape {expected_shape:?} but execution \
                     produced {actual_shape:?}. The planner's shape contract disagrees \
                     with the Rust implementation."
                ));
            }
        }
    }
    Ok(())
}
