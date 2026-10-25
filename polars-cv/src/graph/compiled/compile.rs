//! Compiling a graph: parse, bind every parameter to its slot, plan the
//! outputs, and resolve every all-literal op once.

use polars::prelude::*;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::{atomic::AtomicUsize, Mutex};

use view_buffer::PlannedDType;

use crate::formats::source::Source;
use crate::ops::TypedOp;
use crate::params::ParamCtx;

use crate::graph::step::GraphStep;
use smallvec::SmallVec;

use crate::graph::source::dtype_from_polars_leaf;
use crate::graph::types::{NodeInput, OutputSpec, RowErrorPolicy, UnifiedGraph};
use crate::plan::State;

/// The exact graph JSON a graph was compiled from. Stored on the compiled
/// graph so cache hits are validated by full equality, never by hash alone.
pub(super) struct GraphKwargsKey {
    pub(super) graph_json: String,
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
pub(super) struct CompiledStep {
    pub(super) resolver: OpResolver,
    /// The op's wire name, which prefixes the errors it raises per row.
    pub(super) name: &'static str,
    /// Position in `plan` of each node the op reads, in
    /// [`GraphStep::operands`] order: the one list a step's other-node reads
    /// come from ([`RowOutputs::operands`]), so ids are never hashed per row.
    pub(super) operands: SmallVec<[usize; 2]>,
}

/// One executed node, prepared at compile time.
pub(super) struct NodePlan {
    pub(super) id: String,
    /// What the node reads.
    pub(super) input: PlannedInput,
    /// The node's ops, compiled.
    pub(super) steps: Vec<CompiledStep>,
    /// A step reads its input's shape through `this`, so each row carries the
    /// shape through the steps as they resolve ([`Carried`]).
    pub(super) reads_shape: bool,
}

/// A node's input ([`NodeInput`]), prepared for the row loop.
pub(super) enum PlannedInput {
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
pub(super) struct PlannedDecode {
    pub(super) column: usize,
    pub(super) source: Source,
    /// The source declared `on_error: "null"`: a decode error becomes a null
    /// node output instead of failing the row.
    pub(super) source_null: bool,
    /// Cloud credentials for a `file_path` source, parsed once at the edge
    /// from the source spec's string map.
    pub(super) cloud_options: Option<crate::cloud::CloudOptions>,
    /// Path allowlist; the unrestricted default when the source declared no
    /// `allowed_roots`, which is every existing pipeline.
    pub(super) path_policy: crate::fetch::PathPolicy,
}

impl PlannedInput {
    /// A root's decode.
    pub(super) fn decode(&self) -> Option<&PlannedDecode> {
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
    pub(super) graph: UnifiedGraph,
    /// The executed nodes in topological order, each with everything the row
    /// loop needs about it. Rows address nodes by position here, never by
    /// hashing their id: the id-keyed maps this replaced cost about a third
    /// of the per-row executor time.
    pub(super) plan: Vec<NodePlan>,
    /// Node id → position in `plan`, for cross-node operand reads (`Binary`,
    /// `ApplyMask`, `ChannelMerge`, shape references), which name a node.
    pub(super) node_index: HashMap<String, usize>,
    /// How many plugin inputs this graph reads: one past the highest slot or
    /// column binding. Checked against each call's inputs.
    pub(super) min_inputs: usize,
    /// The exact kwargs this graph was compiled from, kept for exact-match
    /// cache validation.
    pub(super) key: GraphKwargsKey,
    /// Which threads executed rows of this graph and, when armed, a wait
    /// for a second one (test instrumentation).
    #[cfg(test)]
    pub(super) rendezvous: crate::row_split::Rendezvous,
    /// Threads whose rows wait until they are released, so a test can keep
    /// calls running while another starts (test instrumentation).
    #[cfg(test)]
    pub(super) held: Mutex<std::collections::HashSet<std::thread::ThreadId>>,
    /// Signalled when `held` is emptied (test instrumentation).
    #[cfg(test)]
    pub(super) released: std::sync::Condvar,
    /// Rows waiting on `held` now (test instrumentation).
    #[cfg(test)]
    pub(super) held_rows: AtomicUsize,
    /// How many times a buffer-op segment was planned (test instrumentation).
    #[cfg(test)]
    pub(super) plan_builds: AtomicUsize,
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
                            .map(crate::cloud::CloudOptions::from_map)
                            .transpose()
                            .map_err(|e| polars_err!(ComputeError: "{e}"))?,
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
    pub(super) fn node_plan(&self, node_id: &str) -> &NodePlan {
        &self.plan[self.node_index[node_id]]
    }

    /// The parsed (compile-time) graph.
    pub fn graph(&self) -> &UnifiedGraph {
        &self.graph
    }
}

/// Prepare every parameter in the graph for execution, returning how many
/// plugin inputs the graph reads (one past the highest slot or column binding).
///
/// Slots arrive already positional, so nothing is bound by name; this only
/// finds the highest one.
pub(super) fn prepare_graph_params(graph: &mut UnifiedGraph) -> PolarsResult<usize> {
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
pub(super) fn refine_by_column(mut state: State, column: &DataType) -> State {
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
pub(super) fn peel_nesting(dt: &DataType) -> (DataType, Vec<Option<usize>>) {
    let (inner, size) = match dt {
        DataType::List(inner) => (inner, None),
        DataType::Array(inner, n) => (inner, Some(*n)),
        other => return (other.clone(), Vec::new()),
    };
    let (leaf, mut sizes) = peel_nesting(inner);
    sizes.insert(0, size);
    (leaf, sizes)
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
pub(super) fn node_readers(
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
