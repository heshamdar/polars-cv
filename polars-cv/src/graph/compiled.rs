//! Compiled, cacheable form of a pipeline graph.
//!
//! [`CompiledGraph`] is a pure function of the plugin kwargs (`graph_json`):
//! JSON parsing, topological ordering, nested-parameter hoisting, and static
//! (all-literal) op resolution
//! all happen once at compile time. Because the plugin is registered as
//! elementwise, the streaming engine invokes it once **per morsel** — the
//! process-wide cache ([`get_or_compile`]) makes repeat invocations pay only
//! a hash lookup instead of re-parsing and re-compiling the graph.
//!
//! # Cache-safety invariants
//!
//! A `CompiledGraph` must contain **nothing derived from the data**:
//!
//! - no input dtypes (the `"auto"` dtype/ndim resolution from the input
//!   column type happens per call in [`resolved_output_specs`]),
//! - no shapes, row counts, null masks, or any first-row-derived facts.
//!
//! Per-row decode, per-row shapes, and per-row nulls remain exactly per-row,
//! so one cached graph is valid for inputs of any size, dtype mix, or null
//! pattern. A cache hit additionally requires **full string equality** of the
//! kwargs — the hash alone is never trusted.

use polars::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use view_buffer::ops::{Domain, NodeOutput};
use view_buffer::{PlannedDType, ViewBuffer, ViewDto, ViewExpr};

use crate::formats::source::Source;
use crate::ops::TypedOp;
use crate::params::ParamCtx;

use super::row::{run_step, RowOutputs};
use super::step::GraphStep;
use smallvec::SmallVec;

use super::decode::{
    build_series_from_spec, decode_source_row, dtype_from_polars_leaf, null_row_result_for_spec,
    RowFetch, RowParts,
};
use super::encode::encode_node_output;
use super::types::{NodeInput, OutputSpec, OutputValue, RowErrorPolicy, RowResult, UnifiedGraph};
use crate::plan::State;

/// The exact graph JSON a graph was compiled from. Stored on the compiled
/// graph so cache hits are validated by full equality, never by hash alone.
struct GraphKwargsKey {
    graph_json: String,
}

/// A per-op resolver, fixed at graph-compile time.
pub(crate) enum OpResolver {
    /// All params literal: the `GraphStep` is resolved once and borrowed per row.
    Static(GraphStep),
    /// Has at least one dynamic (slot-bound) param: re-resolved per row with
    /// direct typed slot reads (no string-keyed lookups, no `AnyValue`).
    Dynamic(TypedOp),
}

/// One op of a node's chain, prepared at compile time.
struct CompiledStep {
    resolver: OpResolver,
    /// The op's wire name, which prefixes the errors it raises per row.
    name: &'static str,
    /// Position in `plan` of each node the op reads, in
    /// [`GraphStep::operands`] order: the one list a step's other-node reads
    /// come from ([`RowOutputs::operands`]), so ids are never hashed per row.
    operands: SmallVec<[usize; 2]>,
}

/// One executed node, prepared at compile time.
struct NodePlan {
    id: String,
    /// What the node reads.
    input: PlannedInput,
    /// The node's ops, compiled.
    steps: Vec<CompiledStep>,
    /// A step reads its input's shape through `this`, so each row carries the
    /// shape through the steps as they resolve ([`Carried`]).
    reads_shape: bool,
}

/// A node's input ([`NodeInput`]), prepared for the row loop.
enum PlannedInput {
    /// A root, decoding a plugin input (boxed: a source's settings are large
    /// beside a node reference).
    Decode(Box<PlannedDecode>),
    /// Reads the node at `plan` position `upstream`. `take`: this node is its
    /// only reader ([`node_readers`]), so it moves the output out instead of
    /// sharing it.
    Node { upstream: usize, take: bool },
}

/// A root's decode: plugin input `column` through `source`; an `auto` source
/// is routed to a concrete one once per batch.
struct PlannedDecode {
    column: usize,
    source: Source,
    /// The source declared `on_error: "null"`: a decode error becomes a null
    /// node output instead of failing the row.
    source_null: bool,
    /// Cloud credentials for a `file_path` source, parsed once at the edge
    /// from the source spec's string map.
    cloud_options: Option<crate::cloud::CloudOptions>,
    /// Path allowlist; the unrestricted default when the source declared no
    /// `allowed_roots`, which is every existing pipeline.
    path_policy: crate::fetch::PathPolicy,
}

impl PlannedInput {
    /// A root's decode.
    fn decode(&self) -> Option<&PlannedDecode> {
        match self {
            PlannedInput::Decode(decode) => Some(decode),
            PlannedInput::Node { .. } => None,
        }
    }
}

/// A pipeline graph compiled for repeated execution.
///
/// See the module docs for what may and may not live in here.
pub struct CompiledGraph {
    /// Parsed graph with every expression param bound to an input slot.
    graph: UnifiedGraph,
    /// The executed nodes in topological order, each with everything the row
    /// loop needs about it. Rows address nodes by position here, never by
    /// hashing their id: the id-keyed maps this replaced cost about a third
    /// of the per-row executor time (CR-37).
    plan: Vec<NodePlan>,
    /// Node id → position in `plan`, for cross-node operand reads (`Binary`,
    /// `ApplyMask`, `ChannelMerge`, shape references), which name a node.
    node_index: HashMap<String, usize>,
    /// How many plugin inputs this graph reads: one past the highest slot or
    /// column binding. Checked against each call's inputs.
    min_inputs: usize,
    /// The exact kwargs this graph was compiled from, kept for exact-match
    /// cache validation.
    key: GraphKwargsKey,
    /// Which threads executed rows of this graph and, when armed, a wait
    /// for a second one (test instrumentation).
    #[cfg(test)]
    rendezvous: crate::row_split::Rendezvous,
    /// Threads whose rows wait until they are released, so a test can keep
    /// calls running while another starts (test instrumentation).
    #[cfg(test)]
    held: Mutex<std::collections::HashSet<std::thread::ThreadId>>,
    /// Signalled when `held` is emptied (test instrumentation).
    #[cfg(test)]
    released: std::sync::Condvar,
    /// Rows waiting on `held` now (test instrumentation).
    #[cfg(test)]
    held_rows: AtomicUsize,
    /// How many times a buffer-op segment was planned (test instrumentation).
    #[cfg(test)]
    plan_builds: AtomicUsize,
}

/// Per-call execution state: everything derived from the actual input series.
struct ExecState<'a> {
    inputs: &'a [Series],
    /// How many rows this call (one batch under streaming) holds.
    rows: usize,
    /// Output specs with `"auto"` dtype/ndim resolved from the input column
    /// type, sorted by alias.
    resolved_outputs: Vec<(String, OutputSpec)>,
    /// Each path-reading (`file_path`, or `"auto"` over a `String` column)
    /// node's fetch window over this call's column, aligned with `plan`.
    /// Per-path errors surface at their row so the usual error policies
    /// apply.
    fetchers: Vec<Option<crate::fetch::Fetcher<'a>>>,
    /// The concrete source each `"auto"` source node reads this batch's column
    /// as ([`Source::route`]), aligned with `plan`. The column dtype is
    /// constant across rows, so this is taken once per batch; a routing error
    /// is stored and surfaced at its row so the usual error policies apply.
    /// `None` for a concrete source.
    routed_sources: Vec<Option<Result<Source, String>>>,
    /// Position in `plan` of each resolved output's node (aligned with
    /// `resolved_outputs`).
    output_nodes: Vec<Option<usize>>,
}

impl CompiledGraph {
    /// Compile a graph from the plugin kwargs.
    pub fn compile(graph_json: &str) -> PolarsResult<Self> {
        let mut graph = UnifiedGraph::from_json(graph_json)?;
        let min_inputs = prepare_graph_params(&mut graph)?;
        // A graph that does not plan is refused whole, before any row runs.
        resolved_output_specs(&graph, &[])?;

        // `_error` is reserved for the error-message field of the
        // null_with_message policy; an output alias would collide with it.
        if graph.on_error == RowErrorPolicy::NullWithMessage && graph.outputs.contains_key("_error")
        {
            return Err(polars_err!(ComputeError:
                "Output alias '_error' is reserved when on_error='null_with_message'"
            ));
        }

        // Resolve all-literal ops once; anything slot-bound re-resolves per row.
        let empty_ctx = ParamCtx::empty();
        let order = graph.topological_order().to_vec();
        let node_index: HashMap<String, usize> = order
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let readers = node_readers(&graph, &order, &node_index);
        let mut plan: Vec<NodePlan> = Vec::with_capacity(order.len());
        for node_id in &order {
            let node = &graph.nodes[node_id];
            let mut steps: Vec<CompiledStep> = Vec::with_capacity(node.ops.len());
            for spec in &node.ops {
                let resolver = if spec.is_static() {
                    OpResolver::Static(spec.resolve(0, &empty_ctx)?)
                } else {
                    OpResolver::Dynamic(spec.clone())
                };
                // Parsing refused a reference to a node the graph does not
                // have, and every node an executed node reads is executed
                // before it.
                let operands = spec
                    .operands()
                    .iter()
                    .map(|node| node_index[&node.0])
                    .collect();
                steps.push(CompiledStep {
                    resolver,
                    name: spec.name(),
                    operands,
                });
            }
            let input = match &node.input {
                NodeInput::Decode { column, source } => {
                    PlannedInput::Decode(Box::new(PlannedDecode {
                        column: *column,
                        source: source.clone(),
                        source_null: source.nulls_on_error(),
                        cloud_options: source
                            .path_settings()
                            .0
                            .map(crate::cloud::CloudOptions::from_map),
                        path_policy: source
                            .path_settings()
                            .1
                            .map(crate::fetch::PathPolicy::new)
                            .unwrap_or_default(),
                    }))
                }
                NodeInput::Node(up) => {
                    let upstream = node_index[up];
                    PlannedInput::Node {
                        upstream,
                        take: readers[upstream] == 1,
                    }
                }
            };
            plan.push(NodePlan {
                id: node_id.clone(),
                input,
                steps,
                reads_shape: node.ops.iter().any(TypedOp::reads_shape),
            });
        }

        Ok(CompiledGraph {
            graph,
            plan,
            node_index,
            min_inputs,
            key: GraphKwargsKey {
                graph_json: graph_json.to_string(),
            },
            #[cfg(test)]
            rendezvous: crate::row_split::Rendezvous::new(false),
            #[cfg(test)]
            held: Mutex::default(),
            #[cfg(test)]
            released: std::sync::Condvar::new(),
            #[cfg(test)]
            held_rows: AtomicUsize::new(0),
            #[cfg(test)]
            plan_builds: AtomicUsize::new(0),
        })
    }

    /// The compiled form of one node, by id.
    #[cfg(test)]
    fn node_plan(&self, node_id: &str) -> &NodePlan {
        &self.plan[self.node_index[node_id]]
    }

    /// The parsed (compile-time) graph.
    pub fn graph(&self) -> &UnifiedGraph {
        &self.graph
    }

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
                    crate::geom_columns::ContourColumn::new(&inputs[*col]).check_layout()?;
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
    fn execute_rows(
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
            // took down depended on the morsel size (CR-34). The per-row state
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
    fn execute_one_row<'g>(
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
    fn run_row_nodes<'g>(
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
    fn route_auto_sources(&self, inputs: &[Series]) -> Vec<Option<Result<Source, String>>> {
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
    fn fetchers<'a>(&'a self, inputs: &'a [Series]) -> Vec<Option<crate::fetch::Fetcher<'a>>> {
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
struct PendingSegment<'s> {
    /// The ops, borrowed from the row's resolved steps; cloned only when the
    /// segment has to be planned. Inline, since one is built per node per row.
    ops: smallvec::SmallVec<[&'s ViewDto; 8]>,
    /// Index of the segment's first op in the node's op list: its cache slot.
    start: Option<usize>,
    /// Every op is static (all-literal), so its plan is the same for every
    /// row with the same source layout.
    cacheable: bool,
}

impl<'s> PendingSegment<'s> {
    fn push(&mut self, step_idx: usize, op: &'s ViewDto, is_static: bool) {
        if self.ops.is_empty() {
            self.start = Some(step_idx);
            self.cacheable = true;
        }
        self.cacheable &= is_static;
        self.ops.push(op);
    }

    fn clear(&mut self) {
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
struct CachedPlan {
    dtype: view_buffer::DType,
    shape: Vec<usize>,
    strides: Vec<isize>,
    /// Shared, so a hit hands the list out without copying it.
    steps: Arc<[view_buffer::execution::PlanStep]>,
}

/// Distinct source layouts remembered per segment. A column whose rows keep
/// changing shape stops being cached once this many are held, so a call's
/// cache stays small; such segments plan per row, as they did before CR-37.
const PLAN_CACHE_LAYOUTS: usize = 16;

/// The plans of every static buffer-op segment for one call, shared by the
/// row ranges that call runs in parallel: a segment is planned once per
/// distinct source layout per call (CR-37), whichever thread meets it first.
/// Per call, so nothing data-derived outlives it.
struct PlanCache {
    /// Per node, per segment start: the layouts planned so far.
    slots: Vec<Vec<RwLock<Vec<CachedPlan>>>>,
}

impl PlanCache {
    fn new(plan: &[NodePlan]) -> Self {
        PlanCache {
            slots: plan
                .iter()
                .map(|np| np.steps.iter().map(|_| RwLock::default()).collect())
                .collect(),
        }
    }
}

/// The steps planned for `source`'s layout, if `plans` holds them.
fn cached_steps(
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
fn run_segment(
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
type RangeRows = (Vec<Vec<RowResult>>, Vec<Option<String>>);
/// A row range's result, with its failure already in `PolarsError` form.
type RangeOutcome = PolarsResult<RangeRows>;

/// The message a caught panic carried.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// How many times each node's output is read, by position in `order`: as a
/// node's input (its first upstream), as an operand of a step
/// ([`GraphStep::operands`], the one list of the other nodes a step reads)
/// and as a graph output.
///
/// A node read exactly once, as another node's input, is that node's to
/// consume: nothing reads it afterwards, so its buffer can be moved rather
/// than shared. Anything else keeps it shared, so an unrecognised reader can
/// only cost a copy, never corrupt a value another reader sees.
fn node_readers(
    graph: &UnifiedGraph,
    order: &[String],
    node_index: &HashMap<String, usize>,
) -> Vec<usize> {
    let mut readers = vec![0usize; order.len()];
    let mut read = |id: &str| {
        if let Some(&i) = node_index.get(id) {
            readers[i] += 1;
        }
    };
    for node_id in order {
        for dependency in graph.nodes[node_id].dependencies() {
            read(dependency);
        }
    }
    for spec in graph.outputs.values() {
        read(&spec.node);
    }
    readers
}

/// What a node's next step is given, as its `this` parameters read it: the
/// domain, and for a buffer its shape when known before the step runs.
struct Carried {
    domain: Domain,
    shape: Option<Vec<usize>>,
}

impl Carried {
    /// The node's input.
    fn of(input: &NodeOutput) -> Self {
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
    fn after(&self, step: &GraphStep, outputs: &RowOutputs, operands: &[usize]) -> Self {
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

/// Prepare every parameter in the graph for execution, returning how many
/// plugin inputs the graph reads (one past the highest slot or column binding).
///
/// Slots arrive already positional, so nothing is bound by name; this only
/// finds the highest one.
fn prepare_graph_params(graph: &mut UnifiedGraph) -> PolarsResult<usize> {
    let mut inputs = 1;
    for node in graph.nodes.values_mut() {
        if let NodeInput::Decode { column, source } = &node.input {
            inputs = inputs.max(column + 1);
            source.visit_inputs(&mut |_, input| {
                if let view_buffer::mode::RowInput::Slot(slot) = input {
                    inputs = inputs.max(slot + 1);
                }
            });
        }
        for op in &node.ops {
            inputs = inputs.max(op.min_inputs());
        }
    }
    Ok(inputs)
}

/// Plan every node of `graph` and return each output's spec, sorted by alias.
///
/// The one planner: each node starts from its source's state (a root) or its
/// primary upstream's final state, and each op is applied by [`plan::step`] —
/// the function the Python builder calls per append — with the states planned
/// so far as the nodes an op may read by id. With `input_dtypes`, a root whose
/// source resolves its element type or rank from the input column
/// ([`Source::resolves_from_column`]) starts from what that column reveals.
///
/// Shared by the schema (`unified_output_dtype`) and execution entry points,
/// so the published schema and the executed one cannot diverge. Per call,
/// never cached: the column refinement depends on the input dtypes.
pub(crate) fn resolved_output_specs(
    graph: &UnifiedGraph,
    input_dtypes: &[DataType],
) -> PolarsResult<Vec<(String, OutputSpec)>> {
    let mut states = crate::plan::Refs::new();
    for node_id in graph.topological_order() {
        let node = &graph.nodes[node_id];
        let mut state = match &node.input {
            NodeInput::Decode { column, source } => {
                let state = crate::plan::source_state(source);
                match input_dtypes.get(*column) {
                    Some(dt) if source.resolves_from_column() => refine_by_column(state, dt),
                    _ => state,
                }
            }
            NodeInput::Node(up) => states[up].clone(),
        };
        for op in &node.ops {
            state = crate::plan::step(op, &state, &states)
                .map_err(|e| polars_err!(ComputeError: "node '{}': {}", node_id, e))?;
        }
        states.insert(node_id.clone(), state);
    }

    let mut specs: Vec<(String, OutputSpec)> = graph
        .outputs
        .iter()
        // Every output's node was planned: parsing refused an unknown one,
        // and the topological order starts from the outputs.
        .map(|(alias, out)| {
            (
                alias.clone(),
                OutputSpec::planned(
                    out,
                    &states[&out.node],
                    graph.ends_in_histogram_buckets(&out.node),
                ),
            )
        })
        .collect();
    specs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(specs)
}

/// A root's state with what its input column reveals: a `List`/`Array`
/// column's nesting depth is the rank, and its leaf type (when it maps to a
/// buffer element) the dtype. A binary or string column reveals neither — a
/// PNG decodes u8 or u16 — so its state is left as the source planned it.
fn refine_by_column(mut state: State, column: &DataType) -> State {
    use crate::plan::PlannedShape;

    let (leaf, sizes) = peel_nesting(column);
    if sizes.is_empty() {
        return state;
    }
    // Every size the column's type fixes (an `Array` level) is known; a
    // `List` level's varies per row. Known facts are planned, never dropped.
    match &state.shape {
        PlannedShape::Unranked { .. } => state.shape = PlannedShape::Ranked(sizes),
        PlannedShape::Ranked(planned) if planned.len() == sizes.len() => {
            state.shape =
                PlannedShape::Ranked(planned.iter().zip(&sizes).map(|(p, c)| p.or(*c)).collect());
        }
        PlannedShape::Ranked(_) => {}
    }
    if !state.dtype.is_concrete() {
        if let Some(dtype) = dtype_from_polars_leaf(&leaf) {
            state.dtype = PlannedDType::Known(dtype);
        }
    }
    state
}

/// Peel List/Array nesting: the leaf dtype, and each level's size, outermost
/// first — `Some(n)` for a fixed-size `Array`, `None` for a `List`.
fn peel_nesting(dt: &DataType) -> (DataType, Vec<Option<usize>>) {
    let (inner, size) = match dt {
        DataType::List(inner) => (inner, None),
        DataType::Array(inner, n) => (inner, Some(*n)),
        other => return (other.clone(), Vec::new()),
    };
    let (leaf, mut sizes) = peel_nesting(inner);
    sizes.insert(0, size);
    (leaf, sizes)
}

// ============================================================================
// Process-wide compiled-graph cache
// ============================================================================

/// Maximum number of compiled graphs kept resident. Each entry is small
/// (the parsed spec, no data), so this mostly bounds pathological churn from
/// many distinct pipelines in one process.
const GRAPH_CACHE_CAP: usize = 32;

type CacheEntries = Vec<(u64, Arc<CompiledGraph>)>;

fn graph_cache() -> &'static Mutex<CacheEntries> {
    static CACHE: OnceLock<Mutex<CacheEntries>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn cache_key_hash(graph_json: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    graph_json.hash(&mut hasher);
    hasher.finish()
}

/// Fetch the compiled form of a graph, compiling and caching on miss.
///
/// A hit requires both the hash **and** full equality of the kwargs against
/// the stored copy — the hash alone is never trusted. Most-recently-used
/// entries are kept at the front; the lock is never held during compilation.
pub(crate) fn get_or_compile(graph_json: &str) -> PolarsResult<Arc<CompiledGraph>> {
    let hash = cache_key_hash(graph_json);

    {
        let mut cache = graph_cache().lock().unwrap();
        if let Some(pos) = cache
            .iter()
            .position(|(h, compiled)| *h == hash && compiled.key.graph_json == graph_json)
        {
            let entry = cache.remove(pos);
            let compiled = entry.1.clone();
            cache.insert(0, entry);
            return Ok(compiled);
        }
    }

    // Compile outside the lock; concurrent misses may compile the same graph
    // twice, which is harmless (the result is deterministic).
    let compiled = Arc::new(CompiledGraph::compile(graph_json)?);

    let mut cache = graph_cache().lock().unwrap();
    let already_present = cache
        .iter()
        .any(|(h, c)| *h == hash && c.key.graph_json == graph_json);
    if !already_present {
        cache.insert(0, (hash, compiled.clone()));
        cache.truncate(GRAPH_CACHE_CAP);
    }
    Ok(compiled)
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

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_GRAPH: &str = r#"{
        "version": 2,
        "nodes": {
            "n0": {
                "input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                "ops": [{"op": "relu"}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        }
    }"#;

    const DYNAMIC_GRAPH: &str = r#"{
        "version": 2,
        "nodes": {
            "n0": {
                "input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                "ops": [{"op": "scale", "factor": {"$slot": 1}}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        }
    }"#;

    /// Pin the live decode path for `blob` and `raw` sources: both are decoded
    /// by the zero-copy branch of `decode::decode_source_row`
    /// (`decode_binary_row`),
    /// NOT by `execute::decode_image_bytes` — its blob/raw arms are deliberately
    /// absent. This end-to-end test must keep passing when those dead arms are
    /// deleted.
    #[test]
    fn blob_and_raw_sources_decode_via_zero_copy() {
        // blob: f32 buffer round-trips through source(blob) → relu → sink(blob).
        let buf = ViewBuffer::from_vec_with_shape(vec![1.0f32, -2.0, 3.0], vec![3]);
        let input = Series::new("b".into(), &[buf.to_blob()]);
        let compiled = CompiledGraph::compile(SIMPLE_GRAPH).unwrap();
        let out = compiled.execute(&[input]).unwrap();
        let out_bytes = out.binary().unwrap().get(0).unwrap();
        let decoded = ViewBuffer::from_blob(out_bytes).unwrap();
        assert_eq!(decoded.as_slice::<f32>(), &[1.0, 0.0, 3.0]);

        // raw: u8 bytes with an explicit dtype decode as a 1-D u8 buffer
        // (relu then promotes to f32 per the scalar-op dtype contract).
        const RAW_GRAPH: &str = r#"{
            "version": 2,
            "nodes": {
                "n0": {
                    "input": {"decode": {"column": 0, "source": {"format": "raw", "dtype": "u8"}}},
                    "ops": [{"op": "relu"}]
                }
            },
            "outputs": {
                "_output": {"node": "n0", "sink": {"format": "blob"}}
            }
        }"#;
        let input = Series::new("r".into(), &[vec![1u8, 2, 3]]);
        let compiled = CompiledGraph::compile(RAW_GRAPH).unwrap();
        let out = compiled.execute(&[input]).unwrap();
        let out_bytes = out.binary().unwrap().get(0).unwrap();
        let decoded = ViewBuffer::from_blob(out_bytes).unwrap();
        assert_eq!(decoded.as_slice::<f32>(), &[1.0, 2.0, 3.0]);
    }

    use crate::ops::graph::Role;
    use std::cell::RefCell;
    use std::collections::BTreeSet;

    /// The variant a step belongs to.
    ///
    /// Exhaustive, so adding a `GraphStep` fails to compile here. That alone
    /// only forced *this match* to grow: the per-variant graphs in
    /// `every_graph_step_variant_executes` are hand-written, and
    /// `assert_step_covered` was called exactly once — with a `Buffer` step —
    /// so nothing checked that the other nine had a graph. Two tests close
    /// that now: `every_graph_step_variant_is_reachable_from_a_known_op`
    /// over the catalogue's samples, the same way `every_graph_geometry_op_executes`
    /// does in `encode.rs`, and the coverage assertion at the end of
    /// `every_graph_step_variant_executes`, which records what that test
    /// actually ran.
    fn step_name(step: &GraphStep) -> &'static str {
        match step {
            GraphStep::Buffer(_) => "Buffer",
            GraphStep::Geometry(_) => "Geometry",
            GraphStep::Reduction(_) => "Reduction",
            GraphStep::Histogram(_) => "Histogram",
            GraphStep::Lut(_) => "Lut",
            GraphStep::PerceptualHash(_) => "PerceptualHash",
            // A graph op is named by its role: each executes differently.
            GraphStep::Graph(graph) => match graph.role() {
                Role::Binary(..) => "Binary",
                Role::Regions(..) => "Regions",
                Role::ApplyMask { .. } => "ApplyMask",
                Role::ReduceMasked { .. } => "ReduceMasked",
                Role::MaskedSelect { .. } => "MaskedSelect",
                Role::ResizeLike { .. } => "ResizeLike",
                Role::ChannelMerge { .. } => "ChannelMerge",
                Role::ExtractShape => "ExtractShape",
                Role::AssertShape { .. } => "AssertShape",
                Role::LabelReduce { .. } => "LabelReduce",
            },
        }
    }

    /// The variants `step_name` acknowledges, read back from this file.
    fn acknowledged_steps() -> Vec<String> {
        let src = include_str!("compiled.rs");
        let body = src
            .split("fn step_name(step: &GraphStep) -> &'static str {")
            .nth(1)
            .expect("step_name's definition moved — this scan reads nothing");
        let body = body
            .split("\n    }")
            .next()
            .expect("step_name's body has no closing brace");
        // An engine step's variant, or a graph op's role; `Graph` itself is
        // only the role's container.
        let names: Vec<String> = body
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                line.strip_prefix("GraphStep::")
                    .or_else(|| line.strip_prefix("Role::"))
            })
            .filter_map(|rest| rest.split([' ', '(']).next())
            .filter(|name| *name != "Graph")
            .map(str::to_string)
            .collect();
        assert!(
            names.len() >= 10,
            "parsed {} arms from step_name; the scan is out of date",
            names.len()
        );
        names
    }

    /// Every `GraphStep` variant is reachable from some registered op.
    ///
    /// A variant no op produces is dead vocabulary that every match still has
    /// to answer for; a variant that exists but is unreachable is also one the
    /// execution graphs below cannot really be covering. Driven from the
    /// typed catalogue's samples rather than a probe list, so the axis is the
    /// op registry.
    #[test]
    fn every_graph_step_variant_is_reachable_from_a_known_op() {
        let mut reachable: BTreeSet<&'static str> = BTreeSet::new();
        for op in crate::ops::TypedOp::samples() {
            let step = op
                .resolve(0, &ParamCtx::empty())
                .expect("a registered sample resolves");
            reachable.insert(step_name(&step));
        }

        let missing: Vec<String> = acknowledged_steps()
            .into_iter()
            .filter(|name| !reachable.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "these GraphStep variants are acknowledged but no registered op \
             produces them: {missing:?}"
        );
    }

    thread_local! {
        /// `GraphStep` variants `exec` has compiled on this thread.
        ///
        /// Thread-local because each `#[test]` runs on its own thread, so this
        /// records exactly what the calling test ran and nothing another test
        /// happened to execute in parallel. Written by [`exec`], read by
        /// `every_graph_step_variant_executes`.
        static EXECUTED_STEPS: RefCell<BTreeSet<&'static str>> =
            const { RefCell::new(BTreeSet::new()) };
    }

    fn exec(graph: &str, inputs: &[Series]) -> Series {
        let compiled = CompiledGraph::compile(graph).expect("graph must compile");
        EXECUTED_STEPS.with(|seen| {
            let mut seen = seen.borrow_mut();
            for step in compiled.plan.iter().flat_map(|np| &np.steps) {
                // A literal-param op holds its step already; a slot-bound one
                // re-resolves per row, and resolving it against an empty
                // context is enough to learn which variant it is. Failures are
                // skipped rather than asserted: this is bookkeeping for the
                // coverage assertion, and the execute() below is what proves
                // the graph runs.
                let step = match &step.resolver {
                    OpResolver::Static(step) => Some(Cow::Borrowed(step)),
                    OpResolver::Dynamic(spec) => {
                        spec.resolve(0, &ParamCtx::empty()).ok().map(Cow::Owned)
                    }
                };
                if let Some(step) = step {
                    seen.insert(step_name(&step));
                }
            }
        });
        compiled.execute(inputs).expect("graph must execute")
    }

    fn mask_blob() -> Vec<u8> {
        // 4x4 single-channel u8 with a bright 2x2 block.
        let mut data = vec![0u8; 16];
        for y in 1..3 {
            for x in 1..3 {
                data[y * 4 + x] = 255;
            }
        }
        ViewBuffer::from_vec_with_shape(data, vec![4, 4, 1]).to_blob()
    }

    /// Every `GraphStep` variant executes end-to-end through
    /// `CompiledGraph::execute`.
    ///
    /// The graphs below are hand-written, one per variant, and the assertion
    /// at the end is what makes them a *set*: `exec` records the variants each
    /// graph compiled to, and the recorded set must cover every arm
    /// `step_name` acknowledges. Before that, this test called
    /// `assert_step_covered` exactly once — with a `Buffer` step — so a new
    /// variant could be acknowledged, be reachable from an op, and still have
    /// no graph here, while the test's name went on claiming otherwise.
    #[test]
    fn every_graph_step_variant_executes() {
        let f32_blob =
            ViewBuffer::from_vec_with_shape(vec![1.0f32, -2.0, 3.0, 4.0], vec![2, 2]).to_blob();

        // Buffer (relu executes via the fused ViewExpr run).
        let out = exec(
            SIMPLE_GRAPH,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        assert_eq!(out.null_count(), 0);

        // Binary: n1 = n0 + n0.
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "add", "other": "n0"}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let doubled = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(doubled.as_slice::<f32>(), &[2.0, -4.0, 6.0, 8.0]);

        // ApplyMask: mask a buffer with itself (u8 mask semantics).
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "apply_mask", "mask": "n0", "invert": false}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}}
            }"#,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        assert_eq!(out.null_count(), 0);

        // ReduceMasked: a buffer's mean over the pixels it selects itself.
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "reduce_masked", "mask": "n0", "reduction": "mean"}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "native"}}}
            }"#,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        assert_eq!(out.f64().unwrap().get(0), Some(255.0));

        // MaskedSelect: a buffer's values where it is set itself.
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "masked_select", "mask": "n0"}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}}
            }"#,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        let selected = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(selected.shape(), &[4]);
        assert_eq!(selected.to_contiguous().as_slice::<u8>(), &[255; 4]);

        // ResizeLike: a buffer onto its own grid is itself.
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "resize_like", "like": "n0", "filter": "nearest"}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}}
            }"#,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        let same = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        let input = ViewBuffer::from_blob(&mask_blob()).unwrap();
        assert_eq!(same.shape(), input.shape());

        // ChannelMerge: [2,2] + [2,2] -> [2,2,2].
        let single = ViewBuffer::from_vec_with_shape(vec![1u8, 2, 3, 4], vec![2, 2]).to_blob();
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "channel_merge", "others": ["n0"]}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}}
            }"#,
            &[Series::new("b".into(), &[single])],
        );
        let merged = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(merged.shape(), &[2, 2, 2]);

        // Geometry: extract_contours from a binary mask (buffer -> contour).
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "extract_contours", "mode": "external", "method": "simple"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}}
            }"#,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        assert_eq!(out.null_count(), 0);

        // Reduction: global sum -> scalar.
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "reduce_sum"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        assert_eq!(out.f64().unwrap().get(0), Some(6.0));

        // Histogram: a counts vector (u64).
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "histogram", "bins": 4, "range": null,
                                           "closed": "left", "output": "counts"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "list"}}}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let counts = out.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(counts.dtype(), &DataType::UInt64);
        assert_eq!(counts.sum::<u64>().unwrap(), 4);

        // Lut: a 2-entry palette over a [2, 2] u8 buffer -> [2, 2, 2].
        let indices = ViewBuffer::from_vec_with_shape(vec![0u8, 1, 1, 0], vec![2, 2]).to_blob();
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "apply_lut", "table": [[10, 20], [30, 40]],
                                           "out_dtype": "u8"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&indices))],
        );
        let mapped = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(mapped.shape(), &[2, 2, 2]);
        assert_eq!(mapped.as_slice::<u8>(), &[10, 20, 30, 40, 30, 40, 10, 20]);

        // PerceptualHash: image buffer -> 1-D u8 fingerprint (vector domain;
        // u8, so the typed list/array sinks preserve UInt8).
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "perceptual_hash",
                                           "algorithm": "average", "hash_size": 64}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "list"}}}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let hash = out.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(hash.dtype(), &DataType::UInt8);
        assert_eq!(hash.len(), 8);

        // ExtractShape: dimension vector.
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "extract_shape"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        assert_eq!(out.null_count(), 0);

        // LabelReduce: score one contour region from an expression column.
        let contour = view_buffer::geometry::Contour::from_tuples(&[
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
        ]);
        let contour_av = crate::contour::contour_to_anyvalue(&contour);
        let inner = Series::from_any_values("".into(), &[contour_av], false).unwrap();
        let row = AnyValue::List(inner);
        let cont_col = Series::from_any_values("cont".into(), &[row], false).unwrap();
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "label_reduce",
                                           "contours": {"$slot": 1},
                                           "reduction": "max",
                                           "region_mode": "interior"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}}
            }"#,
            &[Series::new("b".into(), &[mask_blob()]), cont_col.clone()],
        );
        assert_eq!(out.null_count(), 0);

        // Regions: a contour set intersected with itself is itself.
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "contour"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "intersection", "other": "n0"}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "native"}}}
            }"#,
            &[cont_col],
        );
        assert_eq!(out.null_count(), 0);
        assert_eq!(out.list().unwrap().get_as_series(0).unwrap().len(), 1);

        // AssertShape: a declaration that holds passes the data through; one
        // that does not fails the row naming what the user wrote.
        let declared = |dims: &str| {
            format!(
                r#"{{
                    "version": 2,
                    "nodes": {{"n0": {{"input": {{"decode": {{"column": 0, "source": {{"format": "blob"}}}}}},
                                      "ops": [{{"op": "assert_shape", "dims": {dims}}}]}}}},
                    "outputs": {{"_output": {{"node": "n0", "sink": {{"format": "blob"}}}}}}
                }}"#
            )
        };
        let blob_input = [Series::new("b".into(), std::slice::from_ref(&f32_blob))];
        assert_eq!(exec(&declared("[2, 2]"), &blob_input).null_count(), 0);
        let err = CompiledGraph::compile(&declared("[2, 3]"))
            .unwrap()
            .execute(&blob_input)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("assert_shape(dims=[2, 3]) does not hold: the data is [2, 2]"),
            "{err}"
        );

        // The set assertion. Without it the graphs above are a list somebody
        // remembered to extend, which is the shape this repo keeps regretting.
        let executed = EXECUTED_STEPS.with(|seen| seen.borrow().clone());
        let missing: Vec<String> = acknowledged_steps()
            .into_iter()
            .filter(|name| !executed.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "these GraphStep variants have no graph in this test, so nothing \
             here executes them: {missing:?} (executed: {executed:?})"
        );
    }

    /// An axis reduction over a 1-D buffer yields shape [1] but its declared
    /// domain is Buffer — the executor must follow the op's output_domain()
    /// (the planning authority), not sniff the [1] shape into a scalar.
    #[test]
    fn axis_reduction_of_1d_buffer_stays_buffer() {
        let blob = ViewBuffer::from_vec_with_shape(vec![1.0f32, 5.0, 3.0], vec![3]).to_blob();
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "reduce_max",
                                           "axis": 0}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}}
            }"#,
            &[Series::new("b".into(), &[blob])],
        );
        let buf = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(buf.shape(), &[1], "axis reduction must stay a buffer");
        assert_eq!(buf.as_slice::<f32>(), &[5.0]);
    }

    /// Percentile is a global scalar reduction end-to-end (regression for the
    /// old DTO match that mislabeled it Buffer).
    #[test]
    fn percentile_reduction_is_scalar() {
        let blob = ViewBuffer::from_vec_with_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).to_blob();
        let out = exec(
            r#"{
                "version": 2,
                "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}},
                                  "ops": [{"op": "reduce_percentile",
                                           "q": 50.0}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}}
            }"#,
            &[Series::new("b".into(), &[blob])],
        );
        assert_eq!(out.dtype(), &DataType::Float64);
        assert!(out.f64().unwrap().get(0).is_some());
    }

    #[test]
    fn cache_hit_returns_same_compilation() {
        let a = get_or_compile(SIMPLE_GRAPH).unwrap();
        let b = get_or_compile(SIMPLE_GRAPH).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "identical kwargs must hit the cache, not recompile"
        );
    }

    const RAW_CHAIN_GRAPH: &str = r#"{
        "version": 2,
        "nodes": {
            "n0": {
                "input": {"decode": {"column": 0, "source": {"format": "raw", "dtype": "u8"}}},
                "ops": [{"op": "invert"}, {"op": "scale", "factor": 2.0}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        }
    }"#;

    fn plans_during(compiled: &CompiledGraph, f: impl FnOnce()) -> usize {
        let before = compiled.plan_builds.load(Ordering::Relaxed);
        f();
        compiled.plan_builds.load(Ordering::Relaxed) - before
    }

    /// A static op segment is planned once per source dtype/shape/strides in a
    /// call, not once per row (CR-37), and replanned when any of them changes.
    #[test]
    fn static_segments_plan_once_per_source_layout() {
        let rows: Vec<Vec<u8>> = vec![
            vec![0, 10, 20, 30],
            vec![1, 11, 21, 31],
            vec![2, 12, 22, 32],
            vec![5, 6, 7, 8, 9, 10],
        ];
        let compiled = CompiledGraph::compile(RAW_CHAIN_GRAPH).unwrap();
        let input = Series::new("r".into(), &rows);
        let mut out = None;
        let plans = plans_during(&compiled, || {
            out = Some(compiled.execute(&[input]).unwrap())
        });
        assert_eq!(
            plans, 2,
            "three [4] rows share one plan, the [6] row needs its own"
        );

        // Each row equals the same row executed alone.
        let out = out.unwrap();
        for (i, row) in rows.iter().enumerate() {
            let alone = compiled
                .execute(&[Series::new("r".into(), std::slice::from_ref(row))])
                .unwrap();
            assert_eq!(
                out.binary().unwrap().get(i),
                alone.binary().unwrap().get(0),
                "row {i}"
            );
        }
    }

    /// Replaying a cached plan allocates nothing: a hit is compared against
    /// the source's own shape and strides and hands out a shared step list.
    /// It runs once per segment per row, and on a small row a key copy and a
    /// step-list copy were a measurable share of the row (performance plan,
    /// phase 5).
    #[test]
    fn a_cache_hit_allocates_nothing() {
        let source = ViewBuffer::from_vec_with_shape(vec![1u8; 8 * 8 * 3], vec![8, 8, 3]);
        let ops = [ViewDto::Compute(view_buffer::ComputeOp::Invert)];
        let refs: Vec<&ViewDto> = ops.iter().collect();
        let cache = RwLock::default();
        let cfg = view_buffer::OptConfig::default();
        run_segment(
            NodeOutput::from_buffer(source.clone()),
            &refs,
            Some(&cache),
            &cfg,
        )
        .unwrap();

        let plans = cache.read().unwrap();
        let (hit, count) =
            crate::test_alloc::large_allocations(1, || cached_steps(&plans, &source).is_some());
        assert!(hit, "the layout planned above was not found");
        assert_eq!(count, 0, "{count} allocations to find a cached plan");
    }

    /// Image-sized allocations one single-row call of `graph` makes (one
    /// row runs inline, so the per-thread count sees all of it).
    fn copies_in_call(graph: &str, input: &Series, image_bytes: usize) -> usize {
        let compiled = CompiledGraph::compile(graph).unwrap();
        let (out, count) = crate::test_alloc::large_allocations(image_bytes, || {
            compiled.execute(std::slice::from_ref(input)).unwrap()
        });
        assert_eq!(out.null_count(), 0);
        count
    }

    /// A numpy row read in place from a blob column holds its own bytes and
    /// no other row's: the column's rows share one data buffer, and handing a
    /// row that whole buffer made `to_list()` of an n-row column materialise
    /// n copies of it (the `zero_copy_blob` benchmark ran 22x slower).
    #[test]
    fn a_blob_row_read_in_place_carries_only_its_bytes() {
        let blobs: Vec<Vec<u8>> = (0..4u8)
            .map(|i| ViewBuffer::from_vec_with_shape(vec![i; 64 * 64], vec![64, 64, 1]).to_blob())
            .collect();
        let input = Series::new("b".into(), &blobs);
        let graph = r#"{"version": 2, "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob", "dtype": "u8"}}}}},
            "outputs": {"_output": {"node": "n0", "sink": {"format": "numpy"}}}}"#;
        let out = CompiledGraph::compile(graph)
            .unwrap()
            .execute(&[input])
            .unwrap();
        let data = out.struct_().unwrap().field_by_name("data").unwrap();
        for (i, row) in data.binary().unwrap().iter().enumerate() {
            let row = row.unwrap();
            assert!(row.len() < 2 * 64 * 64, "row {i} holds {} bytes", row.len());
            assert!(row.contains(&(i as u8)), "row {i} lost its pixels");
        }
    }

    /// A `list` or `array` sink copies each row's values once, into the
    /// column's flat values buffer, whatever the row's layout: it costs one
    /// values-sized allocation more than the zero-copy `numpy` sink over the
    /// same graph, never a per-row intermediate. (f64 elements, so a list's
    /// offsets — one 8-byte offset per innermost list — stay below the
    /// values' size and only copies of the values are counted.)
    #[test]
    fn tensor_sinks_copy_each_row_once() {
        let (h, w, c) = (64usize, 48usize, 3usize);
        let image_bytes = h * w * c * 8;
        let blob = ViewBuffer::from_vec_with_shape(
            (0..h * w * c).map(|i| i as f64).collect::<Vec<f64>>(),
            vec![h, w, c],
        )
        .to_blob();
        let input = Series::new("b".into(), std::slice::from_ref(&blob));
        let graph = |ops: &str, sink: &str| {
            format!(
                r#"{{"version": 2, "nodes": {{"n0": {{"input": {{"decode": {{"column": 0, "source": {{"format": "blob", "dtype": "f64"}}}}}},
                                     "ops": [{ops}]}}}},
                    "outputs": {{"_output": {{"node": "n0", "sink": {sink}}}}}}}"#
            )
        };
        for (layout, ops, shape) in [
            ("contiguous", "", format!("[{h}, {w}, {c}]")),
            (
                "transposed",
                r#"{"op": "transpose", "axes": [1, 0, 2]}"#,
                format!("[{w}, {h}, {c}]"),
            ),
        ] {
            let baseline =
                copies_in_call(&graph(ops, r#"{"format": "numpy"}"#), &input, image_bytes);
            for sink in [
                r#"{"format": "list"}"#.to_string(),
                format!(r#"{{"format": "array", "shape": {shape}}}"#),
            ] {
                let copies = copies_in_call(&graph(ops, &sink), &input, image_bytes);
                assert_eq!(
                    copies,
                    baseline + 1,
                    "{layout} row, sink {sink}: {copies} image-sized allocations, \
                     the numpy sink makes {baseline}"
                );
            }
        }
    }

    /// f32 `[h, w, c]` image-sized allocations one single-row call makes, and
    /// its output.
    fn f32_copies_in_call(graph: &str, input: &Series, elems: usize) -> (Series, usize) {
        let compiled = CompiledGraph::compile(graph).unwrap();
        crate::test_alloc::large_allocations(elems * 4, || {
            compiled.execute(std::slice::from_ref(input)).unwrap()
        })
    }

    /// A row's values as f32, whichever numpy-struct field holds them.
    fn f32_values(out: &Series, field: Option<&str>) -> Vec<f32> {
        let s = match field {
            Some(name) => out.struct_().unwrap().field_by_name(name).unwrap(),
            None => out.clone(),
        };
        let data = s.struct_().unwrap().field_by_name("data").unwrap();
        let bytes = data
            .binary()
            .unwrap()
            .get(0)
            .expect("a null row: a reader lost the output another node consumed")
            .to_vec();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect()
    }

    /// A fused f32 op runs in place on a buffer its node owns: at the start
    /// of a node's ops, and on an upstream node's output that nothing else
    /// reads. An output some other reader still needs (a graph output, an
    /// operand) is neither written nor taken from that reader.
    #[test]
    fn scalar_ops_run_in_place_on_an_owned_buffer() {
        let elems = 32 * 24 * 3;
        let values: Vec<u8> = (0..elems).map(|i| (i % 200) as u8).collect();
        let blob = ViewBuffer::from_vec_with_shape(values.clone(), vec![32, 24, 3]).to_blob();
        let input = Series::new("b".into(), std::slice::from_ref(&blob));
        let doubled: Vec<f32> = values.iter().map(|&v| v as f32 * 2.0).collect();
        let cast: Vec<f32> = values.iter().map(|&v| v as f32).collect();

        // n0 casts (one f32 allocation); n1 scales n0's buffer, which only n1
        // reads, so it writes in place.
        let chain = r#"{
            "version": 2,
            "nodes": {
                "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob", "dtype": "u8"}}},
                       "ops": [{"op": "cast", "dtype": "f32"}]},
                "n1": {"input": {"node": "n0"},
                       "ops": [{"op": "scale", "factor": 2.0}]}
            },
            "outputs": {"_output": {"node": "n1", "sink": {"format": "numpy"}}}
        }"#;
        let (out, copies) = f32_copies_in_call(chain, &input, elems);
        assert_eq!(f32_values(&out, None), doubled);
        assert_eq!(copies, 1, "the scale copied a buffer only it reads");

        // n0 is an output too: n1 must not write into it.
        let shared = r#"{
            "version": 2,
            "nodes": {
                "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob", "dtype": "u8"}}},
                       "ops": [{"op": "cast", "dtype": "f32"}]},
                "n1": {"input": {"node": "n0"},
                       "ops": [{"op": "scale", "factor": 2.0}]}
            },
            "outputs": {"a": {"node": "n0", "sink": {"format": "numpy"}},
                        "b": {"node": "n1", "sink": {"format": "numpy"}}}
        }"#;
        let (out, copies) = f32_copies_in_call(shared, &input, elems);
        assert_eq!(
            f32_values(&out, Some("a")),
            cast,
            "n0's output was overwritten"
        );
        assert_eq!(f32_values(&out, Some("b")), doubled);
        assert_eq!(copies, 2);

        // n1 reads n0 twice (as its input and as the other operand): n0's
        // buffer is not n1's to consume.
        let operand = r#"{
            "version": 2,
            "nodes": {
                "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob", "dtype": "u8"}}},
                       "ops": [{"op": "cast", "dtype": "f32"}]},
                "n1": {"input": {"node": "n0"},
                       "ops": [{"op": "scale", "factor": 2.0}, {"op": "add", "other": "n0"}]}
            },
            "outputs": {"_output": {"node": "n1", "sink": {"format": "numpy"}}}
        }"#;
        let (out, _) = f32_copies_in_call(operand, &input, elems);
        let tripled: Vec<f32> = values.iter().map(|&v| v as f32 * 3.0).collect();
        assert_eq!(
            f32_values(&out, None),
            tripled,
            "the operand saw the scaled buffer"
        );

        // One node: the cast's fresh buffer is the scale's to write.
        let single = r#"{
            "version": 2,
            "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob", "dtype": "u8"}}},
                             "ops": [{"op": "cast", "dtype": "f32"},
                                     {"op": "assert_shape", "dims": [32, 24, 3]},
                                     {"op": "scale", "factor": 2.0}]}},
            "outputs": {"_output": {"node": "n0", "sink": {"format": "numpy"}}}
        }"#;
        let (out, copies) = f32_copies_in_call(single, &input, elems);
        assert_eq!(f32_values(&out, None), doubled);
        assert_eq!(
            copies, 1,
            "the scale after a segment boundary copied its own buffer"
        );
    }

    /// One call spreads its rows over the plugin's thread pool, so a
    /// single-chunk frame on the in-memory engine is not single-threaded
    /// (CR-32), and the rows still come back in order.
    #[test]
    fn a_call_runs_its_rows_on_several_threads() {
        let _pool = crate::row_split::exclusive_pool();
        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;
        if THREAD_POOL.current_num_threads() < 2 {
            eprintln!("skipped: the pool has a single thread");
            return;
        }
        let rows: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8; 64]).collect();
        let compiled = CompiledGraph::compile(RAW_CHAIN_GRAPH).unwrap();
        // The call's second row waits (up to 5 s) for a second thread to
        // run a row: a parallel call gets one at once, a sequential call
        // times out.
        compiled.rendezvous.reset(true);
        let out = compiled.execute(&[Series::new("r".into(), &rows)]).unwrap();
        let threads = compiled.rendezvous.threads();
        assert!(threads > 1, "256 rows ran on {threads} thread(s)");
        for (i, row) in rows.iter().enumerate() {
            // Each row must equal the same row executed alone.
            let alone = compiled
                .execute(&[Series::new("r".into(), std::slice::from_ref(row))])
                .unwrap();
            assert_eq!(
                out.binary().unwrap().get(i),
                alone.binary().unwrap().get(0),
                "row {i}"
            );
        }
    }

    /// Run `during` while `calls` one-row calls of `compiled` are running,
    /// each held inside its row on a thread of its own, then release them.
    fn while_calls_run<R>(compiled: &CompiledGraph, calls: usize, during: impl FnOnce() -> R) -> R {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..calls)
                .map(|_| {
                    scope.spawn(|| {
                        crate::row_split::exclusive::admit_current_thread();
                        compiled
                            .held
                            .lock()
                            .unwrap()
                            .insert(std::thread::current().id());
                        let row = vec![vec![7u8; 64]];
                        let out = compiled.execute(&[Series::new("r".into(), &row)]).unwrap();
                        assert_eq!(out.len(), 1);
                    })
                })
                .collect();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while compiled.held_rows.load(Ordering::SeqCst) < calls {
                assert!(
                    std::time::Instant::now() < deadline,
                    "held calls never started"
                );
                std::thread::yield_now();
            }
            let out = during();
            compiled.held.lock().unwrap().clear();
            compiled.released.notify_all();
            for handle in handles {
                handle.join().unwrap();
            }
            out
        })
    }

    /// Whether a call spreads depends on what runs now, never on what ran
    /// before: a lone call after calls that overlapped (a streaming query's
    /// morsels) spreads. It used to inherit the last call's overlap, so an
    /// eager query after a streaming one ran on one thread (3.9x slower).
    #[test]
    fn a_lone_call_spreads_whatever_ran_before() {
        let _pool = crate::row_split::exclusive_pool();
        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;
        if THREAD_POOL.current_num_threads() < 2 {
            eprintln!("skipped: the pool has a single thread");
            return;
        }
        let compiled = CompiledGraph::compile(RAW_CHAIN_GRAPH).unwrap();
        // Two calls that overlap, as a streaming query's morsels do.
        while_calls_run(&compiled, 1, || threads_of_a_call(&compiled, false));
        let threads = threads_of_a_call(&compiled, true);
        assert!(threads > 1, "a lone call ran on {threads} thread(s)");
    }

    /// A call that starts while a smaller call of its graph runs (a large
    /// row group's morsel beside a small one) still takes the pool's idle
    /// threads: overlapping is not the same as the cores being busy.
    #[test]
    fn a_call_beside_a_small_call_takes_the_idle_threads() {
        let _pool = crate::row_split::exclusive_pool();
        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;
        if THREAD_POOL.current_num_threads() < 2 {
            eprintln!("skipped: the pool has a single thread");
            return;
        }
        let compiled = CompiledGraph::compile(RAW_CHAIN_GRAPH).unwrap();
        let threads = while_calls_run(&compiled, 1, || threads_of_a_call(&compiled, true));
        assert!(threads > 1, "the call ran on {threads} thread(s)");
    }

    /// While as many calls run as the pool has threads (the streaming
    /// engine's steady state), a call runs its rows on its own thread: the
    /// cores are busy, and handing its rows to the pool would only move
    /// every row's buffers between threads.
    #[test]
    fn a_call_runs_inline_while_the_pool_is_busy() {
        let _pool = crate::row_split::exclusive_pool();
        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;
        let busy = THREAD_POOL.current_num_threads();
        let compiled = CompiledGraph::compile(RAW_CHAIN_GRAPH).unwrap();
        let threads = while_calls_run(&compiled, busy, || threads_of_a_call(&compiled, false));
        assert_eq!(threads, 1);
    }

    /// How many threads ran rows of one 256-row call; with `rendezvous`, a
    /// spreading call's second row waits for a second thread.
    fn threads_of_a_call(compiled: &CompiledGraph, rendezvous: bool) -> usize {
        let rows: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8; 64]).collect();
        compiled.rendezvous.reset(rendezvous);
        let out = compiled.execute(&[Series::new("r".into(), &rows)]).unwrap();
        assert_eq!(out.len(), rows.len());
        compiled.rendezvous.threads()
    }

    /// A segment with a per-row parameter is planned every row.
    #[test]
    fn dynamic_segments_are_not_cached() {
        let compiled = CompiledGraph::compile(DYNAMIC_GRAPH).unwrap();
        let buf = ViewBuffer::from_vec_with_shape(vec![1.0f32, 2.0], vec![2]);
        let blobs = Series::new("b".into(), &[buf.to_blob(), buf.to_blob(), buf.to_blob()]);
        let factors = Series::new("f".into(), &[1.0f64, 2.0, 3.0]);
        let mut out = None;
        let plans = plans_during(&compiled, || {
            out = Some(compiled.execute(&[blobs, factors]).unwrap())
        });
        assert_eq!(plans, 3);
        let out = out.unwrap();
        let third = ViewBuffer::from_blob(out.binary().unwrap().get(2).unwrap()).unwrap();
        assert_eq!(third.as_slice::<f32>(), &[3.0, 6.0]);
    }

    #[test]
    fn static_ops_are_precompiled_and_dynamic_are_not() {
        let compiled = CompiledGraph::compile(SIMPLE_GRAPH).unwrap();
        assert!(matches!(
            compiled.node_plan("n0").steps[0].resolver,
            OpResolver::Static(_)
        ));

        let compiled = CompiledGraph::compile(DYNAMIC_GRAPH).unwrap();
        assert!(matches!(
            compiled.node_plan("n0").steps[0].resolver,
            OpResolver::Dynamic(_)
        ));
        // The dynamic op's expr param must have been bound to a slot:
        // 1 source column + position 0 → absolute slot 1.
        match &compiled.node_plan("n0").steps[0].resolver {
            OpResolver::Dynamic(op) => {
                let mut slots = Vec::new();
                op.visit_inputs(&mut |name, input| slots.push((name, input)));
                assert_eq!(slots, [("factor", view_buffer::mode::RowInput::Slot(1))]);
            }
            _ => panic!("expected a dynamic typed op"),
        }
    }

    #[test]
    fn a_slot_beyond_the_call_inputs_is_an_error() {
        // DYNAMIC_GRAPH reads its factor from input 1; a call with only the
        // image column must fail up front, not index past the inputs per row.
        let compiled = CompiledGraph::compile(DYNAMIC_GRAPH).unwrap();
        let buf = ViewBuffer::from_vec_with_shape(vec![1.0f32], vec![1]);
        let err = compiled
            .execute(&[Series::new("b".into(), &[buf.to_blob()])])
            .unwrap_err();
        assert!(err.to_string().contains("reads 2 input columns"), "{err}");
    }
    // --- Compile-time structural validation ---
    //
    // Each case used to fail late (per-row error), silently (all-null
    // output), or with a panic. They must now be clear compile errors.

    fn compile_err(graph_json: &str) -> String {
        CompiledGraph::compile(graph_json)
            .err()
            .expect("malformed graph must fail to compile")
            .to_string()
    }

    #[test]
    fn output_referencing_unknown_node_is_a_compile_error() {
        // Previously: silent all-null output column.
        let err = compile_err(
            r#"{
            "version": 2,
            "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}}},
            "outputs": {"_output": {"node": "nope", "sink": {"format": "blob"}}}
        }"#,
        );
        assert!(err.contains("unknown node 'nope'"), "{err}");
    }

    #[test]
    fn unknown_input_node_is_a_compile_error() {
        let err = compile_err(
            r#"{
            "version": 2,
            "nodes": {
                "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                "n1": {"input": {"node": "ghost"}}
            },
            "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}}
        }"#,
        );
        assert!(
            err.contains("node 'n1' reads node 'ghost', which is not in the graph"),
            "{err}"
        );
    }

    #[test]
    fn unknown_source_format_is_a_compile_error() {
        let err = compile_err(
            r#"{
            "version": 2,
            "nodes": {"n0": {"input": {"decode": {"column": 0, "source": {"format": "carrier_pigeon"}}}}},
            "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}}
        }"#,
        );
        assert!(
            err.contains("unknown source format 'carrier_pigeon'"),
            "{err}"
        );
    }

    // --- One operand path (CR-117) ---

    /// Every registered op that reads another node, with its operands renamed
    /// to `to`, and the domain its own input and its operands are in.
    fn operand_reading_samples(to: &str) -> Vec<(TypedOp, Domain, Domain)> {
        let samples: Vec<_> = TypedOp::samples()
            .into_iter()
            .filter(|op| !op.operands().is_empty())
            .map(|mut op| {
                for node in op.operands_mut() {
                    node.0 = to.to_string();
                }
                let input = op.input_domains()[0];
                let operand = op.operand_domains()[0];
                (op, input, operand)
            })
            .collect();
        // Binary ops, apply_mask, channel_merge, the region ops, rasterize.
        assert!(
            samples.len() >= 10,
            "only {} operand-reading samples: the filter matches nothing it should",
            samples.len()
        );
        samples
    }

    /// A node reference that no node answers is refused when the graph
    /// compiles, by one check with one message for every op that reads a
    /// node, whatever kind of operand it is (another image, a mask, merged
    /// channels, a canvas). Driven by the catalogue's samples, so a new
    /// operand-reading op is covered by being registered.
    #[test]
    fn a_dangling_node_reference_is_one_compile_error_for_every_reading_op() {
        for (op, _, _) in operand_reading_samples("ghost") {
            let name = op.name();
            let graph = format!(
                r#"{{"version": 2, "nodes": {{"n0": {{"input": {{"decode": {{"column": 0, "source": {{"format": "blob"}}}}}}}},
                      "n1": {{"input": {{"node": "n0"}}, "ops": [{op}]}}}},
                    "outputs": {{"_output": {{"node": "n1", "sink": {{"format": "blob"}}}}}}}}"#,
                op = serde_json::to_string(&op).unwrap()
            );
            let err = compile_err(&graph);
            let expected =
                format!("node 'n1': {name}() reads node 'ghost', which is not in the graph");
            assert!(err.contains(&expected), "{name}: {err}");
        }
    }

    /// A null operand nulls the node reading it, for every op that reads a
    /// node: the row's other outputs are untouched and nothing errors. The
    /// executor reads operands at one place, so this holds for all of them or
    /// for none; the sweep is what would show a step that reads its operand
    /// another way.
    #[test]
    fn a_null_operand_nulls_the_node_for_every_reading_op() {
        let contour = view_buffer::geometry::Contour::from_tuples(&[
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
        ]);
        let inner = Series::from_any_values(
            "".into(),
            &[crate::contour::contour_to_anyvalue(&contour)],
            false,
        )
        .unwrap();
        let contours =
            Series::from_any_values("c".into(), &[AnyValue::List(inner)], false).unwrap();
        let column = |domain: Domain, null: bool| -> (Series, &'static str) {
            match (domain, null) {
                (Domain::Buffer, false) => (Series::new("b".into(), &[mask_blob()]), "blob"),
                (Domain::Buffer, true) => {
                    (Series::full_null("b".into(), 1, &DataType::Binary), "blob")
                }
                (Domain::Contour, false) => (contours.clone(), "contour"),
                (Domain::Contour, true) => (
                    Series::full_null("c".into(), 1, contours.dtype()),
                    "contour",
                ),
                (other, _) => panic!("no test column for the {other:?} domain"),
            }
        };
        for (op, input, operand) in operand_reading_samples("n2") {
            let name = op.name();
            let (input_col, input_source) = column(input, false);
            let (operand_col, operand_source) = column(operand, true);
            let graph = format!(
                r#"{{"version": 2,
                    "nodes": {{"n0": {{"input": {{"decode": {{"column": 0, "source": {{"format": "{input_source}"}}}}}}}},
                      "n2": {{"input": {{"decode": {{"column": 1, "source": {{"format": "{operand_source}"}}}}}}}},
                      "n1": {{"input": {{"node": "n0"}}, "ops": [{op}]}}}},
                    "outputs": {{"read": {{"node": "n1", "sink": {{"format": "{sink}"}}}},
                                 "input": {{"node": "n0", "sink": {{"format": "{input_sink}"}}}}}}}}"#,
                op = serde_json::to_string(&op).unwrap(),
                sink = if op.output_domain(input) == Domain::Contour {
                    "native"
                } else {
                    "blob"
                },
                input_sink = if input == Domain::Contour {
                    "native"
                } else {
                    "blob"
                },
            );
            let out = CompiledGraph::compile(&graph)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .execute(&[input_col, operand_col])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let out = out.struct_().unwrap();
            let field = |n: &str| out.field_by_name(n).unwrap().null_count();
            assert_eq!(
                field("read"),
                1,
                "{name}: a null operand must null the node"
            );
            assert_eq!(
                field("input"),
                0,
                "{name}: the input's own output is not null"
            );
        }
    }
}
