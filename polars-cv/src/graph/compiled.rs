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

use pyo3_polars::export::polars_core::runtime::THREAD_POOL;
use view_buffer::geometry::label::score_contours_on_buffer;
use view_buffer::ops::{Domain, NodeOutput};
use view_buffer::{Op, PlannedDType, ViewBuffer, ViewDto, ViewExpr};

use crate::contour::parse_contour_list;
use crate::execute::decode_image_bytes;
use crate::formats::source::Source;
use crate::ops::graph::Role;
use crate::ops::{NodeRef, TypedOp};
use crate::params::ParamCtx;
use view_buffer::geometry::ops::RasterSize;

use super::step::GraphStep;

use super::decode::{
    build_series_from_spec, decode_binary_zero_copy, decode_list_or_array_source,
    dtype_from_polars_leaf, get_binary_row_buffer, null_row_result_for_spec,
};
use super::encode::{encode_node_output, execute_geometry_op};
use super::types::{OutputSpec, OutputValue, RowErrorPolicy, RowResult, UnifiedGraph};
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
    /// `rasterize(shape=<node>)`: output dimensions come from another node's
    /// buffer at execution time (the referenced node is an upstream
    /// dependency, so it has already run); the remaining params resolve from
    /// the spec like any dynamic op.
    RasterizeShapeRef {
        op: view_buffer::GeometryOp<view_buffer::mode::Wire>,
        shape_node: String,
    },
}

/// One op of a node's chain, resolved for the current row.
enum ResolvedStep<'a> {
    Step(Cow<'a, GraphStep>),
    RasterizeShapeRef {
        op: &'a view_buffer::GeometryOp<view_buffer::mode::Wire>,
        shape_node: &'a str,
    },
}

/// One executed node, prepared at compile time.
struct NodePlan {
    id: String,
    /// The node's source (its ops are compiled into `resolvers`); an `auto`
    /// one is routed to a concrete source once per batch.
    source: Source,
    /// Input column, for a root node.
    column: Option<usize>,
    /// Position in `plan` of the node this one reads, for a non-root node.
    upstream: Option<usize>,
    /// The source declared `on_error: "null"`: a decode error becomes a null
    /// node output instead of failing the row.
    source_null: bool,
    /// Cloud credentials for a `file_path` source, parsed once at the edge
    /// from the source spec's string map.
    cloud_options: Option<crate::cloud::CloudOptions>,
    /// Path allowlist; the unrestricted default when the source declared no
    /// `allowed_roots`, which is every existing pipeline.
    path_policy: crate::fetch::PathPolicy,
    /// Op resolvers, aligned with the node's `ops`.
    resolvers: Vec<OpResolver>,
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
    /// Which threads executed rows of this graph (test instrumentation).
    #[cfg(test)]
    row_threads: Mutex<std::collections::HashSet<std::thread::ThreadId>>,
    /// Signalled when a thread first executes a row (test instrumentation).
    #[cfg(test)]
    row_threads_seen: std::sync::Condvar,
    /// When set, a row waits (bounded) until a second thread has run a row,
    /// so a test of parallelism does not depend on scheduling luck.
    #[cfg(test)]
    rendezvous: std::sync::atomic::AtomicBool,
    /// How many times a buffer-op segment was planned (test instrumentation).
    #[cfg(test)]
    plan_builds: AtomicUsize,
}

/// Per-call execution state: everything derived from the actual input series.
struct ExecState<'a> {
    inputs: &'a [Series],
    /// Output specs with `"auto"` dtype/ndim resolved from the input column
    /// type, sorted by alias.
    resolved_outputs: Vec<(String, OutputSpec)>,
    /// Bytes of this batch's remote `file_path` sources, fetched concurrently
    /// up front (node_id → batch). Converts per-row network latency into
    /// per-batch latency; per-path errors surface at their row so the usual
    /// error policies apply.
    prefetched: Vec<Option<crate::fetch::FetchedBatch>>,
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
        validate_graph_structure(&graph)?;
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
        let mut plan: Vec<NodePlan> = Vec::with_capacity(order.len());
        for node_id in &order {
            let node = &graph.nodes[node_id];
            let mut resolvers: Vec<OpResolver> = Vec::with_capacity(node.ops.len());
            for spec in &node.ops {
                // rasterize(shape=<node>) takes its canvas from another
                // node's output, not a param, so it gets a dedicated resolver.
                if let TypedOp::Geometry(
                    op @ view_buffer::GeometryOp::Rasterize {
                        size: RasterSize::FromNode(NodeRef(shape_node)),
                        ..
                    },
                ) = spec
                {
                    {
                        if !graph.nodes.contains_key(shape_node) {
                            return Err(polars_err!(ComputeError:
                                "Node '{}': rasterize shape reference '{}' is not a node in the graph",
                                node_id, shape_node
                            ));
                        }
                        resolvers.push(OpResolver::RasterizeShapeRef {
                            op: op.clone(),
                            shape_node: shape_node.clone(),
                        });
                        continue;
                    }
                }
                if spec.is_static() {
                    resolvers.push(OpResolver::Static(spec.resolve(0, &empty_ctx)?));
                } else {
                    resolvers.push(OpResolver::Dynamic(spec.clone()));
                }
            }
            let column = graph.column_bindings.get(node_id).copied();
            // A non-root node reads its first upstream; `validate_graph_structure`
            // has already refused a node with neither a binding nor an upstream,
            // and an upstream of an executed node is executed before it.
            let upstream = match column {
                Some(_) => None,
                None => Some(node_index[&node.upstream[0]]),
            };
            plan.push(NodePlan {
                id: node_id.clone(),
                column,
                upstream,
                source_null: node.source.nulls_on_error(),
                cloud_options: node
                    .source
                    .path_settings()
                    .0
                    .map(crate::cloud::CloudOptions::from_map),
                path_policy: node
                    .source
                    .path_settings()
                    .1
                    .map(crate::fetch::PathPolicy::new)
                    .unwrap_or_default(),
                resolvers,
                source: node.source.clone(),
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
            row_threads: Mutex::default(),
            #[cfg(test)]
            row_threads_seen: std::sync::Condvar::new(),
            #[cfg(test)]
            rendezvous: std::sync::atomic::AtomicBool::new(false),
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
            resolved_outputs,
            prefetched: self.prefetch_remote_sources(inputs),
            routed_sources: self.route_auto_sources(inputs),
            output_nodes,
        };

        // Rows are independent, so the call is split into contiguous row
        // ranges that run on the plugin's thread pool and are concatenated in
        // order. Without this a call used one core however many rows it held,
        // so the in-memory engine was single-threaded on a single-chunk frame
        // (CR-32). The pool is the plugin's own copy of polars' `THREAD_POOL`
        // (a plugin links its own polars-core, so it cannot join the host's);
        // it is sized by `POLARS_MAX_THREADS`, and callers block while their
        // rows run, so concurrent calls (streaming morsels) share its threads
        // rather than multiplying them.
        let ranges = row_ranges(len, THREAD_POOL.current_num_threads());
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
        let mut outcomes: Vec<Option<RangeOutcome>> = ranges.iter().map(|_| None).collect();
        if let [only] = ranges.as_slice() {
            outcomes[0] = Some(run_range(0, only.clone()));
        } else {
            let run_range = &run_range;
            THREAD_POOL.scope(|scope| {
                for ((range_idx, rows), slot) in
                    ranges.iter().cloned().enumerate().zip(outcomes.iter_mut())
                {
                    scope.spawn(move |_| *slot = Some(run_range(range_idx, rows)));
                }
            });
        }

        // Concatenate in row order. Under `on_error="raise"` the first failing
        // range holds the earliest failing row, so its error is the one a
        // sequential run would have reported.
        let mut results: Vec<Vec<RowResult>> = (0..state.resolved_outputs.len())
            .map(|_| Vec::with_capacity(len))
            .collect();
        let mut error_messages: Vec<Option<String>> = Vec::new();
        for outcome in outcomes {
            let (range_results, range_messages) =
                outcome.expect("every range ran to completion inside the scope")?;
            for (all, part) in results.iter_mut().zip(range_results) {
                all.extend(part);
            }
            error_messages.extend(range_messages);
        }

        let with_message = self.graph.on_error == RowErrorPolicy::NullWithMessage;
        if self.graph.is_single_output() && !with_message {
            let (_, spec) = &state.resolved_outputs[0];
            // Take ownership of the row results so the encoder can move each
            // row's bytes/buffers into Arrow instead of copying them.
            let data = results.swap_remove(0);
            build_series_from_spec(inputs[0].name().clone(), spec, data)
        } else {
            let mut fields: Vec<Series> = Vec::with_capacity(state.resolved_outputs.len() + 1);
            for ((alias, spec), data) in state.resolved_outputs.iter().zip(results) {
                let field_series = build_series_from_spec(PlSmallStr::from_str(alias), spec, data)?;
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
        let mut node_outputs: Vec<Option<NodeOutput>> = vec![None; self.plan.len()];
        let mut dto_scratch: Vec<ResolvedStep<'_>> = Vec::new();
        let start = rows.start;
        for row_idx in rows {
            if first_failure.load(Ordering::Relaxed) < range_idx {
                break;
            }
            node_outputs.iter_mut().for_each(|output| *output = None);
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
                        return Err(msg);
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
        node_outputs: &mut [Option<NodeOutput>],
        dto_scratch: &mut Vec<ResolvedStep<'g>>,
        plan_cache: &PlanCache,
        results: &mut [Vec<RowResult>],
    ) -> Result<(), String> {
        #[cfg(test)]
        {
            let mut seen = self.row_threads.lock().unwrap();
            seen.insert(std::thread::current().id());
            self.row_threads_seen.notify_all();
            // One wait per call: once a second thread has arrived (or the
            // wait timed out, as it does for a sequential run) rows proceed.
            if self.rendezvous.swap(false, Ordering::Relaxed) {
                let _ = self
                    .row_threads_seen
                    .wait_timeout_while(seen, std::time::Duration::from_secs(5), |s| s.len() < 2)
                    .unwrap();
            }
        }
        self.run_row_nodes(state, ctx, row_idx, node_outputs, dto_scratch, plan_cache)?;
        for (((alias, spec), node), rows) in state
            .resolved_outputs
            .iter()
            .zip(&state.output_nodes)
            .zip(results.iter_mut())
        {
            if let Some(output) = node.and_then(|i| node_outputs[i].as_ref()) {
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
                            OutputValue::TypedList { data, shape } => {
                                RowResult::TypedList(Some((data, shape)))
                            }
                            OutputValue::TypedArray { data, shape } => {
                                RowResult::TypedArray(Some((data, shape)))
                            }
                            OutputValue::NumpyStruct(buf) => RowResult::NumpyStruct(Some(buf)),
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

    /// Look up another node's output for the current row.
    ///
    /// An absent entry means one of two different things, and conflating them
    /// turns a legitimate null into a hard error. `Ok(None)` is the benign
    /// case: the node belongs to the graph but produced nothing for this row,
    /// because its input was null, its source failed under
    /// `source(on_error="null")`, or a per-row parameter was null under
    /// `on_null_param="null"`. The node reading it must then go null too.
    /// A name that is not in the graph at all stays an error.
    fn operand<'o>(
        &self,
        node_outputs: &'o [Option<NodeOutput>],
        id: &str,
        what: &str,
    ) -> Result<Option<&'o NodeOutput>, String> {
        match self.node_index.get(id) {
            Some(&i) => Ok(node_outputs[i].as_ref()),
            None if self.graph.nodes.contains_key(id) => Ok(None),
            None => Err(format!("{what} references unknown node '{id}'")),
        }
    }

    /// Whether a runtime `NodeOutput::Vector` may be read as a 1-D buffer here.
    ///
    /// The planner's `vector` *domain* and the runtime `Vector` *representation*
    /// are not the same set, and conflating them is a live hazard.
    /// `perceptual_hash` and the histogram vector modes plan as `vector` but
    /// execute as `Buffer`, so the only steps that ever receive a runtime
    /// `Vector` are those fed by `ExtractShape` or `LabelReduce`.
    ///
    /// Coercing one is sound only where the step's published output shape
    /// derives from *this* operand alone.
    ///
    /// - `Reduction` qualifies: rank and domain fold from its single input, so
    ///   `extract_shape().reduce_sum()` plans and executes the same scalar.
    /// - `Binary` does **not**. The planner publishes the *left* operand's rank
    ///   while execution broadcasts against the other operand, so coercing a
    ///   left-hand vector against a rank-3 image would execute at rank 3 after
    ///   the plan promised rank 1 — `sink("list")` silently flattening twelve
    ///   values into a three-element plan, and `sink("array", shape=[3])`
    ///   failing at execution. Refusing keeps the failure loud; there is no
    ///   coherent broadcast to publish at plan time.
    ///
    /// Exhaustive, so a new variant has to answer this question rather than
    /// inherit an answer.
    fn vector_reads_as_buffer(step: &GraphStep) -> bool {
        match step {
            GraphStep::Reduction(_) => true,
            GraphStep::Buffer(_)
            | GraphStep::Geometry(_)
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_) => false,
            GraphStep::Graph(graph) => match graph.role() {
                Role::Binary(..)
                | Role::ApplyMask { .. }
                | Role::ChannelMerge { .. }
                | Role::ExtractShape
                | Role::AssertShape { .. }
                | Role::LabelReduce { .. } => false,
            },
        }
    }

    /// The buffer a step consumes, honouring the step's *declared* input domains.
    ///
    /// **Read the contract; never restate it.** `GraphStep::input_domains` is the
    /// single authority `CLAUDE.md` names for what a step accepts, and the Python
    /// planner validates against it through `Plan::push`. Execution used to
    /// re-derive the same fact by hand at ten sites — `current_output.as_buffer()`
    /// with a hardcoded `"<Step> requires Buffer"` string each time — and the two
    /// had already diverged: `Reduction` declares `[Buffer, Vector]`, so the
    /// planner accepted `extract_shape().reduce_sum()` and execution then failed
    /// the row with "Reduction requires Buffer, got Vector".
    ///
    /// Reading the declared domains is what decides *rejection*;
    /// `vector_reads_as_buffer` decides the narrower question of whether a
    /// runtime vector can stand in for a buffer, which the domain vocabulary
    /// alone cannot answer.
    fn step_buffer_operand(
        output: &NodeOutput,
        step: &GraphStep,
        what: &str,
    ) -> Result<Arc<ViewBuffer>, String> {
        let domain = output.domain();
        let accepted = step.input_domains();
        match output {
            NodeOutput::Buffer(buf) => Ok(Arc::clone(buf)),
            NodeOutput::Vector(vals) if Self::vector_reads_as_buffer(step) => {
                Ok(Arc::new(ViewBuffer::from_vec(vals.as_ref().clone())))
            }
            NodeOutput::Vector(_) if accepted.contains(&Domain::Vector) => Err(format!(
                "{what} declares it accepts a vector, but cannot read one whose \
                 rank the plan did not fix: its output shape depends on another \
                 operand. Reduce or rasterize the vector first."
            )),
            _ => Err(format!(
                "{what} accepts {} input but received {domain:?}",
                accepted
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect::<Vec<_>>()
                    .join(" or ")
            )),
        }
    }

    /// Run every node of the graph for one row, leaving each node's output in
    /// `node_outputs` (no output encoding).
    fn run_row_nodes<'g>(
        &'g self,
        state: &ExecState<'_>,
        ctx: &ParamCtx<'_>,
        row_idx: usize,
        node_outputs: &mut [Option<NodeOutput>],
        dto_scratch: &mut Vec<ResolvedStep<'g>>,
        plan_cache: &PlanCache,
    ) -> Result<(), String> {
        let inputs = state.inputs;
        {
            // Labelled so a null per-row parameter can skip straight to the
            // next node: leaving this node out of `node_outputs` is exactly
            // how a null *input* already propagates (see the `else` branch of
            // `if let Some(input)` below, and `execute_one_row`).
            'nodes: for (idx, np) in self.plan.iter().enumerate() {
                let node_input: Option<NodeOutput> = if let Some(col_idx) = np.column {
                    let on_error_null = np.source_null;
                    // An `"auto"` source reads as the concrete source it was
                    // routed to once per batch (`route_auto_sources`).
                    let decode_result = match state.routed_sources[idx].as_ref() {
                        Some(Ok(routed)) => Ok(routed),
                        Some(Err(e)) => Err(e.clone()),
                        None => Ok(&np.source),
                    }
                    .and_then(|source| {
                        // Bounds are validated once per call in `execute()`.
                        decode_source_row(
                            np,
                            source,
                            &inputs[col_idx],
                            row_idx,
                            state.prefetched[idx].as_ref(),
                        )
                    });
                    match decode_result {
                        Ok(output) => output,
                        Err(_e) if on_error_null => None,
                        Err(e) => return Err(e),
                    }
                } else {
                    np.upstream.and_then(|u| node_outputs[u].clone())
                };
                if let Some(input) = node_input {
                    // Static ops are borrowed from the compiled graph; dynamic
                    // ops are re-resolved for this row through typed slot
                    // reads. The scratch Vec is reused so steady-state rows
                    // allocate nothing here.
                    dto_scratch.clear();
                    {
                        for resolver in &np.resolvers {
                            match resolver {
                                OpResolver::Static(step) => {
                                    dto_scratch.push(ResolvedStep::Step(Cow::Borrowed(step)))
                                }
                                OpResolver::Dynamic(spec) => {
                                    ctx.clear_null();
                                    match spec.resolve(row_idx, ctx) {
                                        Ok(step) => {
                                            dto_scratch.push(ResolvedStep::Step(Cow::Owned(step)))
                                        }
                                        // Under `on_null_param="null"` a null
                                        // parameter is not a failure — this
                                        // node just has no output for this row.
                                        Err(_) if ctx.took_null() => continue 'nodes,
                                        Err(e) => return Err(format!("Op resolution error: {e}")),
                                    }
                                }
                                OpResolver::RasterizeShapeRef { op, shape_node } => dto_scratch
                                    .push(ResolvedStep::RasterizeShapeRef { op, shape_node }),
                            }
                        }
                    }
                    let mut current_output = input;
                    // Engine-tier optimization toggles for this graph (Tier-2).
                    // `OptConfig` is `Copy`, so the closure captures a value and
                    // does not borrow `self`.
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
                    for (step_idx, step) in dto_scratch.iter().enumerate() {
                        let graph_step = match step {
                            ResolvedStep::RasterizeShapeRef { op, shape_node } => {
                                // Dimensions come from the referenced node's
                                // buffer (already executed: it is upstream).
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let Some(shape_output) = self.operand(
                                    node_outputs,
                                    shape_node,
                                    "Rasterize shape reference",
                                )?
                                else {
                                    continue 'nodes;
                                };
                                // Not a step input: this reads *another node's*
                                // output purely for its dimensions, so it wants a
                                // buffer regardless of what the current step accepts.
                                let shape_buf = shape_output.as_buffer().ok_or_else(|| {
                                    format!(
                                        "Rasterize shape reference '{shape_node}' must be a Buffer, got {:?}",
                                        shape_output.domain()
                                    )
                                })?;
                                let dims = shape_buf.shape();
                                if dims.len() < 2 {
                                    return Err(format!(
                                        "Rasterize shape reference '{shape_node}' must be at least 2D, got {}D",
                                        dims.len()
                                    ));
                                }
                                let height = dims[0] as u32;
                                let width = dims[1] as u32;
                                ctx.clear_null();
                                let resolved = view_buffer::mode::Resolve::resolve(
                                    *op,
                                    &crate::ops::param::RowValues { row: row_idx, ctx },
                                );
                                let geo_op = match resolved {
                                    Ok(geo_op) => geo_op.with_canvas(height, width),
                                    Err(_) if ctx.took_null() => continue 'nodes,
                                    Err(e) => return Err(e.to_string()),
                                };
                                current_output = execute_geometry_op(current_output, &geo_op)?;
                                continue;
                            }
                            ResolvedStep::Step(step) => step,
                        };
                        match graph_step.as_ref() {
                            GraphStep::Geometry(geo_op) => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                current_output = execute_geometry_op(current_output, geo_op)?;
                            }
                            GraphStep::Reduction(reduction_op) => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = Self::step_buffer_operand(
                                    &current_output,
                                    graph_step.as_ref(),
                                    "Reduction",
                                )?;
                                view_buffer::ops::validation::validate_concrete(
                                    reduction_op,
                                    &[current_buf.shape()],
                                    &[current_buf.dtype()],
                                )
                                .map_err(|e| format!("{}: {e}", reduction_op.name()))?;
                                let result = reduction_op.execute(&current_buf);
                                // The op's declared output domain (the same
                                // authority the planner reads) decides scalar
                                // vs buffer — not the result's shape, which is
                                // also [1] for an axis reduction of a 1-D
                                // buffer (a buffer-domain output).
                                current_output = if reduction_op.output_domain() == Domain::Scalar {
                                    let v = result.scalar_f64().ok_or_else(|| {
                                        format!(
                                            "Scalar reduction produced non-scalar shape {:?}",
                                            result.shape()
                                        )
                                    })?;
                                    NodeOutput::Scalar(v)
                                } else {
                                    NodeOutput::from_buffer(result)
                                };
                            }
                            GraphStep::Histogram(histogram_op) => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = Self::step_buffer_operand(
                                    &current_output,
                                    graph_step.as_ref(),
                                    "Histogram",
                                )?;
                                view_buffer::ops::validation::validate_concrete(
                                    histogram_op,
                                    &[current_buf.shape()],
                                    &[current_buf.dtype()],
                                )
                                .map_err(|e| format!("{}: {e}", histogram_op.name()))?;
                                let result = histogram_op.execute(&current_buf);
                                current_output = NodeOutput::from_buffer(result);
                            }
                            GraphStep::PerceptualHash(phash_op) => {
                                current_output =
                                    flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                let current_buf = Self::step_buffer_operand(
                                    &current_output,
                                    graph_step.as_ref(),
                                    "PerceptualHash",
                                )?;
                                // `apply_perceptual_hash` converts to u8 image
                                // format before hashing and returns a 1-D u8
                                // buffer. Like the histogram vector modes, the
                                // result rides as a Buffer at runtime; the
                                // planned `vector` domain (OutputSpec) selects
                                // the List encoding at sink time.
                                view_buffer::ops::validation::validate_concrete(
                                    phash_op,
                                    &[current_buf.shape()],
                                    &[current_buf.dtype()],
                                )
                                .map_err(|e| format!("{}: {e}", phash_op.name()))?;
                                let result = view_buffer::execution::runner::apply_perceptual_hash(
                                    (*current_buf).clone(),
                                    phash_op.clone(),
                                );
                                current_output = NodeOutput::from_buffer(result);
                            }
                            GraphStep::Graph(graph) => match graph.role() {
                                Role::Binary(op, other) => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf = Self::step_buffer_operand(
                                        &current_output,
                                        graph_step.as_ref(),
                                        "Binary op",
                                    )?;
                                    let Some(other_output) =
                                        self.operand(node_outputs, &other.0, "Binary op")?
                                    else {
                                        continue 'nodes;
                                    };
                                    let other_buf = Self::step_buffer_operand(
                                        other_output,
                                        graph_step.as_ref(),
                                        "Binary op other operand",
                                    )?;
                                    view_buffer::ops::validation::validate_concrete(
                                        &op,
                                        &[current_buf.shape(), other_buf.shape()],
                                        &[current_buf.dtype(), other_buf.dtype()],
                                    )
                                    .map_err(|e| format!("{}: {e}", op.name()))?;
                                    let result = op.execute(&current_buf, &other_buf);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                Role::ApplyMask { mask, invert } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf = Self::step_buffer_operand(
                                        &current_output,
                                        graph_step.as_ref(),
                                        "ApplyMask",
                                    )?;
                                    let Some(mask_output) =
                                        self.operand(node_outputs, &mask.0, "ApplyMask")?
                                    else {
                                        continue 'nodes;
                                    };
                                    let mask_buf = Self::step_buffer_operand(
                                        mask_output,
                                        graph_step.as_ref(),
                                        "ApplyMask mask",
                                    )?;
                                    view_buffer::validate_mask(
                                        current_buf.shape(),
                                        mask_buf.shape(),
                                    )
                                    .map_err(|e| format!("apply_mask: {e}"))?;
                                    let result =
                                        view_buffer::apply_mask(&current_buf, &mask_buf, *invert);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                                Role::AssertShape { dims, exact } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let shape: Vec<usize> = match &current_output {
                                        NodeOutput::Buffer(buf) => buf.shape().to_vec(),
                                        NodeOutput::Vector(vals) => vec![vals.len()],
                                        other => {
                                            return Err(format!(
                                            "assert_shape() declares a shape, but the data here \
                                                 is {}",
                                            other.domain().name()
                                        ))
                                        }
                                    };
                                    check_declared_shape(&shape, dims, *exact)?;
                                }
                                Role::ExtractShape => {
                                    // Extract shape from buffer and return as vector
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf = Self::step_buffer_operand(
                                        &current_output,
                                        graph_step.as_ref(),
                                        "ExtractShape",
                                    )?;
                                    let shape = current_buf.shape();
                                    // Return shape as f64 vector [height, width, channels]
                                    let shape_vec: Vec<f64> =
                                        shape.iter().map(|&d| d as f64).collect();
                                    current_output = NodeOutput::from_vector(shape_vec);
                                }
                                Role::LabelReduce {
                                    contours,
                                    reduction,
                                    region_mode,
                                } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf = Self::step_buffer_operand(
                                        &current_output,
                                        graph_step.as_ref(),
                                        "LabelReduce",
                                    )?;
                                    let contour_col =
                                        ctx.col(contours.0).map_err(|e| e.to_string())?;
                                    let contour_value = contour_col.get_any(row_idx).map_err(|e| {
                                        format!(
                                            "LabelReduce failed to read contours at row {row_idx}: {e}"
                                        )
                                    })?;
                                    if contour_value.is_null() {
                                        current_output = NodeOutput::from_vector(Vec::new());
                                        continue;
                                    }
                                    let contours =
                                        parse_contour_list(&contour_value).map_err(|e| {
                                            format!("LabelReduce contour parsing failed: {e}")
                                        })?;
                                    let scores = score_contours_on_buffer(
                                        &current_buf,
                                        &contours,
                                        *reduction,
                                        *region_mode,
                                    )?;
                                    current_output = NodeOutput::from_vector(scores);
                                }
                                Role::ChannelMerge { others } => {
                                    current_output =
                                        flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                                    let current_buf = Self::step_buffer_operand(
                                        &current_output,
                                        graph_step.as_ref(),
                                        "ChannelMerge",
                                    )?;
                                    // Owned first: `step_buffer_operand` hands back an
                                    // `Arc`, which must outlive the borrow the merge
                                    // call takes.
                                    let mut owned: Vec<Arc<ViewBuffer>> = vec![current_buf];
                                    for other_id in others {
                                        let Some(other_output) = self.operand(
                                            node_outputs,
                                            &other_id.0,
                                            "ChannelMerge",
                                        )?
                                        else {
                                            continue 'nodes;
                                        };
                                        // Same contract read as the current operand
                                        // above -- ChannelMerge declares `[Buffer]`, so
                                        // this refuses a vector, but it refuses it by
                                        // reading the contract rather than restating it.
                                        owned.push(Self::step_buffer_operand(
                                            other_output,
                                            graph_step.as_ref(),
                                            &format!("ChannelMerge operand '{}'", other_id.0),
                                        )?);
                                    }
                                    let all_bufs: Vec<&ViewBuffer> =
                                        owned.iter().map(|b| b.as_ref()).collect();
                                    view_buffer::validate_channel_merge(
                                        &all_bufs.iter().map(|b| b.shape()).collect::<Vec<_>>(),
                                        &all_bufs.iter().map(|b| b.dtype()).collect::<Vec<_>>(),
                                    )
                                    .map_err(|e| format!("channel_merge: {e}"))?;
                                    let result = view_buffer::apply_channel_merge(&all_bufs);
                                    current_output = NodeOutput::from_buffer(result);
                                }
                            },
                            // Fusable single-buffer engine ops accumulate and
                            // run as one ViewExpr chain at the next flush.
                            GraphStep::Buffer(dto) => {
                                let is_static =
                                    matches!(step, ResolvedStep::Step(Cow::Borrowed(_)));
                                pending_buffer_ops.push(step_idx, dto, is_static);
                            }
                        }
                    }
                    current_output = flush_buffer_ops(current_output, &mut pending_buffer_ops)?;
                    node_outputs[idx] = Some(current_output);
                }
            }
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
            .map(|np| np.source.route(inputs.get(np.column?)?))
            .collect()
    }

    /// Concurrently fetch every remote `file_path` source in this batch.
    ///
    /// Per-call (per morsel) and derived only from this batch's path values —
    /// nothing here is cached on the compiled graph. Distinct paths are
    /// fetched once; wrong-dtype columns are skipped so the row loop reports
    /// them with its usual error message. `"auto"` nodes are included too: an
    /// auto source over a `String` column resolves to `file_path`, and the
    /// `series.str()` check below naturally skips auto nodes bound to any other
    /// column type. Aligned with `plan`.
    fn prefetch_remote_sources(
        &self,
        inputs: &[Series],
    ) -> Vec<Option<crate::fetch::FetchedBatch>> {
        self.plan
            .iter()
            .map(|np| {
                if !matches!(np.source, Source::FilePath { .. } | Source::Auto { .. }) {
                    return None;
                }
                let ca = inputs.get(np.column?)?.str().ok()?;
                Some(crate::fetch::prefetch(
                    ca,
                    np.cloud_options.as_ref(),
                    &np.path_policy,
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
    /// segment has to be planned.
    ops: Vec<&'s ViewDto>,
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
    steps: Vec<view_buffer::execution::PlanStep>,
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
                .map(|np| np.resolvers.iter().map(|_| RwLock::default()).collect())
                .collect(),
        }
    }
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
    let buf = output
        .as_buffer()
        .ok_or_else(|| format!("Expected Buffer for pending ops, got {:?}", output.domain()))?;
    let source = (**buf).clone();
    let key = (
        source.dtype(),
        source.shape().to_vec(),
        source.strides_bytes().to_vec(),
    );
    let matches = |c: &CachedPlan| c.dtype == key.0 && c.shape == key.1 && c.strides == key.2;
    let execute = |source: ViewBuffer, steps| {
        let plan = view_buffer::execution::ExecutionPlan { source, steps };
        NodeOutput::from_buffer(plan.execute())
    };
    let plan = |source: ViewBuffer| -> Result<Vec<view_buffer::execution::PlanStep>, String> {
        let mut expr = ViewExpr::new_source(source);
        for op in ops {
            // The validated entry point: an op that cannot run on the
            // shape reaching it is this row's error, not a kernel panic.
            expr = expr
                .try_apply_op((*op).clone())
                .map_err(|e| format!("{}: {e}", op.as_op().name()))?;
        }
        Ok(expr.plan_with(cfg).steps)
    };
    let cached = |plans: &[CachedPlan]| plans.iter().find(|p| matches(p)).map(|p| p.steps.clone());

    let Some(cache) = cache else {
        let steps = plan(source.clone())?;
        return Ok((execute(source, steps), true));
    };
    if let Some(steps) = cached(&cache.read().unwrap()) {
        return Ok((execute(source, steps), false));
    }
    // Planned under the write lock: a range that misses on the same layout
    // meanwhile waits here and then finds this plan, rather than planning
    // its own. Only planning is serialised; execution happens after release.
    let mut plans = cache.write().unwrap();
    if let Some(steps) = cached(&plans) {
        drop(plans);
        return Ok((execute(source, steps), false));
    }
    let steps = plan(source.clone())?;
    if plans.len() < PLAN_CACHE_LAYOUTS {
        plans.push(CachedPlan {
            dtype: key.0,
            shape: key.1.clone(),
            strides: key.2.clone(),
            steps: steps.clone(),
        });
    }
    drop(plans);
    Ok((execute(source, steps), true))
}

/// Contiguous row ranges covering `0..len`, for `threads` workers.
///
/// A few ranges per thread so an expensive stretch of rows does not leave
/// the other threads idle; one range when there is nothing to split.
fn row_ranges(len: usize, threads: usize) -> Vec<Range<usize>> {
    const RANGES_PER_THREAD: usize = 4;
    let count = (threads * RANGES_PER_THREAD).clamp(1, len.max(1));
    let (base, extra) = (len / count, len % count);
    let mut start = 0;
    (0..count)
        .map(|i| {
            let end = start + base + usize::from(i < extra);
            let range = start..end;
            start = end;
            range
        })
        .collect()
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

/// Decode row `row` of a root node's column through its concrete `source`
/// (an `auto` source is routed per batch before it gets here).
fn decode_source_row(
    np: &NodePlan,
    source: &Source,
    series: &Series,
    row: usize,
    prefetched: Option<&crate::fetch::FetchedBatch>,
) -> Result<Option<NodeOutput>, String> {
    if series.dtype() == &DataType::Null {
        return Ok(None);
    }
    let binary = || {
        series.binary().map_err(|_| {
            format!(
                "Expected Binary column for node '{}', got {:?}",
                np.id,
                series.dtype()
            )
        })
    };
    let buffer = |buf| Some(NodeOutput::from_buffer(buf));
    match source {
        // The column's contour set; a mask is the `rasterize` op that
        // follows, if any.
        Source::Contour { .. } => match series.get(row) {
            Ok(value) if !value.is_null() => crate::contour::parse_contour_set(&value)
                .map(|set| Some(NodeOutput::from_contours(set)))
                .map_err(|e| format!("Contour decode error: {e}")),
            _ => Ok(None),
        },
        // `file_path` is fetch + decode: `crate::fetch` reads the bytes the
        // path names (applying its `PathPolicy` sandbox), then they decode as
        // image bytes.
        Source::FilePath { .. } => {
            let ca = series.str().map_err(|_| {
                format!(
                    "Expected String column for file_path source '{}', got {:?}",
                    np.id,
                    series.dtype()
                )
            })?;
            let Some(path) = ca.get(row) else {
                return Ok(None);
            };
            // Stage 1: bytes. Remote paths were fetched concurrently before
            // the row loop; local files are read inline.
            let empty;
            let batch = match prefetched {
                Some(b) => b,
                None => {
                    empty = crate::fetch::FetchedBatch::empty();
                    &empty
                }
            };
            let bytes =
                crate::fetch::row_bytes(batch, path, np.cloud_options.as_ref(), &np.path_policy)?;
            // Stage 2: the contents decode like image bytes.
            decode_image_bytes(&bytes, source)
                .map(buffer)
                .map_err(|e| format!("Decode error for file '{path}': {e}"))
        }
        Source::List { .. } | Source::Array { .. } => {
            decode_list_or_array_source(series, row, source.dtype(), source.require_contiguous())
                .map(|buf| buf.map(NodeOutput::from_buffer))
                .map_err(|e| format!("List/Array decode error: {e}"))
        }
        // Raw bytes take the declared dtype.
        Source::Raw { dtype, .. } => {
            let Some((bytes, offset, len)) = get_binary_row_buffer(binary()?, row) else {
                return Ok(None);
            };
            decode_binary_zero_copy(bytes, offset, len, Some(dtype.get()))
                .map(buffer)
                .map_err(|e| format!("Zero-copy decode error: {e}"))
        }
        // A blob carries its own dtype, which a declared one must match: the
        // planner (and identity elimination) takes the declaration as fact.
        Source::Blob { dtype, .. } => {
            let Some((bytes, offset, len)) = get_binary_row_buffer(binary()?, row) else {
                return Ok(None);
            };
            let buf = decode_binary_zero_copy(bytes, offset, len, None)
                .map_err(|e| format!("Zero-copy decode error: {e}"))?;
            match dtype.map(|d| d.get()) {
                Some(declared) if declared != buf.dtype() => Err(format!(
                    "the blob holds {} elements, but the source declares dtype=\"{}\". A \
                     blob carries its own dtype: drop the declaration, correct it, or \
                     .cast(\"{}\") after the source.",
                    buf.dtype().short_name(),
                    declared.short_name(),
                    declared.short_name()
                )),
                _ => Ok(buffer(buf)),
            }
        }
        Source::ImageBytes { .. } => match binary()?.get(row) {
            Some(bytes) => decode_image_bytes(bytes, source)
                .map(buffer)
                .map_err(|e| format!("Decode error: {e}")),
            None => Ok(None),
        },
        Source::Auto { .. } => Err(format!(
            "internal: auto source '{}' reached decoding unrouted",
            np.id
        )),
    }
}

/// Validate the structural invariants the executor relies on, at compile time.
///
/// Each of these used to fail late and badly: a typo'd output node silently
/// produced all-null rows, a non-root node without upstream panicked at
/// `upstream[0]`, and unknown source formats surfaced as per-row decode
/// errors. Compile-time rejection gives one clear error instead.
fn validate_graph_structure(graph: &UnifiedGraph) -> PolarsResult<()> {
    for (node_id, node) in &graph.nodes {
        if !graph.column_bindings.contains_key(node_id) && node.upstream.is_empty() {
            polars_bail!(ComputeError:
                "Node '{}' has neither an input column binding nor an upstream node",
                node_id
            );
        }
        for upstream_id in &node.upstream {
            if !graph.nodes.contains_key(upstream_id) {
                polars_bail!(ComputeError:
                    "Node '{}' references unknown upstream node '{}'",
                    node_id, upstream_id
                );
            }
        }
    }
    for (alias, spec) in &graph.outputs {
        if !graph.nodes.contains_key(&spec.node) {
            polars_bail!(ComputeError:
                "Output '{}' references unknown node '{}'",
                alias, spec.node
            );
        }
    }
    Ok(())
}

/// Prepare every parameter in the graph for execution, returning how many
/// plugin inputs the graph reads (one past the highest slot or column binding).
///
/// Slots arrive already positional, so nothing is bound by name; this only
/// finds the highest one.
fn prepare_graph_params(graph: &mut UnifiedGraph) -> PolarsResult<usize> {
    let mut inputs = graph
        .column_bindings
        .values()
        .map(|&idx| idx + 1)
        .max()
        .unwrap_or(1);
    for node in graph.nodes.values_mut() {
        node.source
            .visit_slots(&mut |_, slot| inputs = inputs.max(slot + 1));
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
        let mut state = match graph.column_bindings.get(node_id) {
            Some(&column) => {
                let state = crate::plan::source_state(&node.source);
                match input_dtypes.get(column) {
                    Some(dt) if node.source.resolves_from_column() => refine_by_column(state, dt),
                    _ => state,
                }
            }
            None => states[&node.upstream[0]].clone(),
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
        .map(|(alias, out)| {
            let planned = states.get(&out.node).ok_or_else(|| {
                polars_err!(ComputeError: "output '{}' names unknown node '{}'", alias, out.node)
            })?;
            Ok((
                alias.clone(),
                OutputSpec::planned(out, planned, graph.ends_in_histogram_buckets(&out.node)),
            ))
        })
        .collect::<PolarsResult<_>>()?;
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

/// Check one row's shape against an `assert_shape` declaration: the
/// declaration applied to the row's shape ([`crate::plan::apply_declaration`],
/// the reading the planner applies too).
///
/// The declaration is the user's statement about their data, so a mismatch is
/// reported as theirs: it names what they wrote and what arrived.
fn check_declared_shape(shape: &[usize], dims: &[Option<u32>], exact: bool) -> Result<(), String> {
    let declared: Vec<Option<usize>> = dims.iter().map(|d| d.map(|d| d as usize)).collect();
    let row = crate::plan::PlannedShape::Ranked(shape.iter().map(|&n| Some(n)).collect());
    if crate::plan::apply_declaration(&row, &declared, exact).is_ok() {
        return Ok(());
    }
    let size = |d: &Option<usize>| d.map_or("None".to_string(), |d| d.to_string());
    let written = if exact {
        format!(
            "dims=[{}]",
            declared.iter().map(size).collect::<Vec<_>>().join(", ")
        )
    } else {
        declared
            .iter()
            .enumerate()
            .filter_map(|(axis, d)| {
                d.map(|d| format!("{}={d}", crate::plan::declared_name(axis, false)))
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(format!(
        "assert_shape({written}) does not hold: the data is {shape:?}. An assertion states \
         what the data is; it does not change it. Correct the assertion, or drop it \
         and let the planner infer the shape."
    ))
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
        "nodes": {
            "n0": {
                "source": {"format": "blob"},
                "ops": [{"op": "relu"}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        },
        "column_bindings": {"n0": 0}
    }"#;

    const DYNAMIC_GRAPH: &str = r#"{
        "nodes": {
            "n0": {
                "source": {"format": "blob"},
                "ops": [{"op": "scale", "factor": {"$slot": 1}}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        },
        "column_bindings": {"n0": 0}
    }"#;

    /// Pin the live decode path for `blob` and `raw` sources: both are decoded
    /// by the zero-copy branch in `run_row_nodes` (`decode_binary_zero_copy`),
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
            "nodes": {
                "n0": {
                    "source": {"format": "raw", "dtype": "u8"},
                    "ops": [{"op": "relu"}]
                }
            },
            "outputs": {
                "_output": {"node": "n0", "sink": {"format": "blob"}}
            },
            "column_bindings": {"n0": 0}
        }"#;
        let input = Series::new("r".into(), &[vec![1u8, 2, 3]]);
        let compiled = CompiledGraph::compile(RAW_GRAPH).unwrap();
        let out = compiled.execute(&[input]).unwrap();
        let out_bytes = out.binary().unwrap().get(0).unwrap();
        let decoded = ViewBuffer::from_blob(out_bytes).unwrap();
        assert_eq!(decoded.as_slice::<f32>(), &[1.0, 2.0, 3.0]);
    }

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
            GraphStep::PerceptualHash(_) => "PerceptualHash",
            // A graph op is named by its role: each executes differently.
            GraphStep::Graph(graph) => match graph.role() {
                Role::Binary(..) => "Binary",
                Role::ApplyMask { .. } => "ApplyMask",
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
            for resolver in compiled.plan.iter().flat_map(|np| &np.resolvers) {
                // A literal-param op holds its step already; a slot-bound one
                // re-resolves per row, and resolving it against an empty
                // context is enough to learn which variant it is. Failures are
                // skipped rather than asserted: this is bookkeeping for the
                // coverage assertion, and the execute() below is what proves
                // the graph runs.
                let step = match resolver {
                    OpResolver::Static(step) => Some(Cow::Borrowed(step)),
                    OpResolver::Dynamic(spec) => {
                        spec.resolve(0, &ParamCtx::empty()).ok().map(Cow::Owned)
                    }
                    OpResolver::RasterizeShapeRef { op, .. } => {
                        view_buffer::mode::Resolve::resolve(
                            op,
                            &crate::ops::param::RowValues {
                                row: 0,
                                ctx: &ParamCtx::empty(),
                            },
                        )
                        .ok()
                        .map(|geo| Cow::Owned(GraphStep::Geometry(geo.with_canvas(1, 1))))
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
                "nodes": {
                    "n0": {"source": {"format": "blob"}},
                    "n1": {"source": {"format": "blob"}, "upstream": ["n0"],
                           "ops": [{"op": "add", "other": "n0"}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let doubled = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(doubled.as_slice::<f32>(), &[2.0, -4.0, 6.0, 8.0]);

        // ApplyMask: mask a buffer with itself (u8 mask semantics).
        let out = exec(
            r#"{
                "nodes": {
                    "n0": {"source": {"format": "blob"}},
                    "n1": {"source": {"format": "blob"}, "upstream": ["n0"],
                           "ops": [{"op": "apply_mask", "mask": "n0", "invert": false}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        assert_eq!(out.null_count(), 0);

        // ChannelMerge: [2,2] + [2,2] -> [2,2,2].
        let single = ViewBuffer::from_vec_with_shape(vec![1u8, 2, 3, 4], vec![2, 2]).to_blob();
        let out = exec(
            r#"{
                "nodes": {
                    "n0": {"source": {"format": "blob"}},
                    "n1": {"source": {"format": "blob"}, "upstream": ["n0"],
                           "ops": [{"op": "channel_merge", "others": ["n0"]}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), &[single])],
        );
        let merged = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(merged.shape(), &[2, 2, 2]);

        // Geometry: extract_contours from a binary mask (buffer -> contour).
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "extract_contours", "mode": "external", "method": "simple"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), &[mask_blob()])],
        );
        assert_eq!(out.null_count(), 0);

        // Reduction: global sum -> scalar.
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "reduce_sum"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        assert_eq!(out.f64().unwrap().get(0), Some(6.0));

        // Histogram: a counts vector (u64).
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "histogram", "bins": 4, "range": null,
                                           "closed": "left", "output": "counts"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "list"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let counts = out.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(counts.dtype(), &DataType::UInt64);
        assert_eq!(counts.sum::<u64>().unwrap(), 4);

        // PerceptualHash: image buffer -> 1-D u8 fingerprint (vector domain;
        // u8, so the typed list/array sinks preserve UInt8).
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "perceptual_hash",
                                           "algorithm": "average", "hash_size": 64}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "list"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), std::slice::from_ref(&f32_blob))],
        );
        let hash = out.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(hash.dtype(), &DataType::UInt8);
        assert_eq!(hash.len(), 8);

        // ExtractShape: dimension vector.
        let out = exec(
            r#"{
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "extract_shape"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}},
                "column_bindings": {"n0": 0}
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
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "label_reduce",
                                           "contours": {"$slot": 1},
                                           "reduction": "max",
                                           "region_mode": "interior"}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}},
                "column_bindings": {"n0": 0}
            }"#,
            &[Series::new("b".into(), &[mask_blob()]), cont_col],
        );
        assert_eq!(out.null_count(), 0);

        // AssertShape: a declaration that holds passes the data through; one
        // that does not fails the row naming what the user wrote.
        let declared = |dims: &str| {
            format!(
                r#"{{
                    "nodes": {{"n0": {{"source": {{"format": "blob"}},
                                      "ops": [{{"op": "assert_shape", "dims": {dims}}}]}}}},
                    "outputs": {{"_output": {{"node": "n0", "sink": {{"format": "blob"}}}}}},
                    "column_bindings": {{"n0": 0}}
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
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "reduce_max",
                                           "axis": 0}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}},
                "column_bindings": {"n0": 0}
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
                "nodes": {"n0": {"source": {"format": "blob"},
                                  "ops": [{"op": "reduce_percentile",
                                           "q": 50.0}]}},
                "outputs": {"_output": {"node": "n0", "sink": {"format": "native"}}},
                "column_bindings": {"n0": 0}
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
        "nodes": {
            "n0": {
                "source": {"format": "raw", "dtype": "u8"},
                "ops": [{"op": "invert"}, {"op": "scale", "factor": 2.0}]
            }
        },
        "outputs": {
            "_output": {"node": "n0", "sink": {"format": "blob"}}
        },
        "column_bindings": {"n0": 0}
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

    /// One call spreads its rows over the plugin's thread pool, so a
    /// single-chunk frame on the in-memory engine is not single-threaded
    /// (CR-32), and the rows still come back in order.
    #[test]
    fn a_call_runs_its_rows_on_several_threads() {
        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;
        if THREAD_POOL.current_num_threads() < 2 {
            eprintln!("skipped: the pool has a single thread");
            return;
        }
        let rows: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8; 64]).collect();
        let compiled = CompiledGraph::compile(RAW_CHAIN_GRAPH).unwrap();
        // The first row waits (up to 5 s) for a second thread to run a row:
        // a parallel call gets one at once, a sequential call times out.
        compiled.rendezvous.store(true, Ordering::Relaxed);
        let out = compiled.execute(&[Series::new("r".into(), &rows)]).unwrap();
        let threads = compiled.row_threads.lock().unwrap().len();
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
            compiled.node_plan("n0").resolvers[0],
            OpResolver::Static(_)
        ));

        let compiled = CompiledGraph::compile(DYNAMIC_GRAPH).unwrap();
        assert!(matches!(
            compiled.node_plan("n0").resolvers[0],
            OpResolver::Dynamic(_)
        ));
        // The dynamic op's expr param must have been bound to a slot:
        // 1 source column + position 0 → absolute slot 1.
        match &compiled.node_plan("n0").resolvers[0] {
            OpResolver::Dynamic(op) => {
                let mut slots = Vec::new();
                op.visit_slots(&mut |name, slot| slots.push((name, slot)));
                assert_eq!(slots, [("factor", 1)]);
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
            "nodes": {"n0": {"source": {"format": "blob"}}},
            "outputs": {"_output": {"node": "nope", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(err.contains("unknown node 'nope'"), "{err}");
    }

    #[test]
    fn node_without_binding_or_upstream_is_a_compile_error() {
        // Previously: panic at `upstream[0]` in the row loop.
        let err = compile_err(
            r#"{
            "nodes": {
                "n0": {"source": {"format": "blob"}},
                "orphan": {"source": {"format": "blob"}}
            },
            "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(
            err.contains("neither an input column binding nor an upstream"),
            "{err}"
        );
    }

    #[test]
    fn unknown_upstream_reference_is_a_compile_error() {
        let err = compile_err(
            r#"{
            "nodes": {
                "n0": {"source": {"format": "blob"}},
                "n1": {"source": {"format": "blob"}, "upstream": ["ghost"]}
            },
            "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(err.contains("unknown upstream node 'ghost'"), "{err}");
    }

    #[test]
    fn unknown_source_format_is_a_compile_error() {
        let err = compile_err(
            r#"{
            "nodes": {"n0": {"source": {"format": "carrier_pigeon"}}},
            "outputs": {"_output": {"node": "n0", "sink": {"format": "blob"}}},
            "column_bindings": {"n0": 0}
        }"#,
        );
        assert!(
            err.contains("unknown source format 'carrier_pigeon'"),
            "{err}"
        );
    }
}
