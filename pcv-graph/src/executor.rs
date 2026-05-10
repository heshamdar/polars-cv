//! Single-row graph executor.
//!
//! The Polars row loop and expression-bound parameter resolution live in the
//! `polars-cv` bridge layer; this module owns the actual graph walk:
//!
//! 1. Toposort nodes (cached on `CompiledGraph`).
//! 2. Look up source/op factories from the inventory registry.
//! 3. For each row: produce sources, execute ops in order, consume sinks.
//! 4. Catch panics so a misbehaving op surfaces as `ExecError::OpPanicked`
//!    rather than aborting the batch.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use thiserror::Error;
use view_buffer::ops::NodeOutput;

use crate::ir::{Graph, Inputs, Node, NodeId};
use crate::op::{ExecCtx, OpError, OpHandle, OpInputs};
use crate::params::ParamMap;
use crate::registry::find_op;
use crate::sink::{find_sink, Sink, SinkError, SinkRowOutput};
use crate::source::{find_source, Source, SourceError, SourceInputs};

/// Compiled form of a [`Graph`] — factories resolved, toposort cached.
///
/// Built once per query (per Polars batch); reused across rows.
pub struct CompiledGraph {
    pub graph: Graph,
    /// Toposorted node ids.
    pub order: Vec<NodeId>,
    /// Source nodes: factory + spec.
    pub sources: HashMap<NodeId, Arc<dyn Source>>,
    /// Op nodes: pre-built ops keyed by node id when all params are literal,
    /// or `None` when at least one param is expression-bound (re-built per
    /// row by [`Executor::execute_row`]).
    pub ops: HashMap<NodeId, Option<OpHandle>>,
    /// Sinks indexed by output binding position.
    pub sinks: Vec<Arc<dyn Sink>>,
}

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("unknown op `{op}` referenced by node `{node}`")]
    UnknownOp { op: String, node: NodeId },

    #[error("unknown source `{source_name}` referenced by node `{node}`")]
    UnknownSource { source_name: String, node: NodeId },

    #[error("unknown sink `{sink}` referenced by output `{alias}`")]
    UnknownSink { sink: String, alias: String },

    #[error("graph has cycle (node `{node}` revisited)")]
    Cycle { node: NodeId },

    #[error("op `{op}` factory rejected params: {source}")]
    OpFactory {
        op: String,
        #[source]
        source: OpError,
    },

    #[error("source `{source_name}` factory rejected params: {error}")]
    SourceFactory { source_name: String, error: SourceError },

    #[error("sink `{sink}` factory rejected params: {error}")]
    SinkFactory { sink: String, error: SinkError },

    #[error("wire version mismatch: graph={got}, executor={want} — rebuild the plugin")]
    WireVersion { got: u32, want: u32 },
}

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("source `{source_name}` failed on row {row_idx}: {error}")]
    Source {
        source_name: String,
        row_idx: usize,
        error: SourceError,
    },

    #[error("op `{op}` failed on row {row_idx}: {error}")]
    Op {
        op: String,
        row_idx: usize,
        error: OpError,
    },

    #[error("sink `{sink}` failed on row {row_idx}: {error}")]
    Sink {
        sink: String,
        row_idx: usize,
        error: SinkError,
    },

    #[error("op `{op}` panicked on row {row_idx}: {message}")]
    OpPanicked {
        op: String,
        row_idx: usize,
        message: String,
    },

    #[error("missing upstream node `{node}` for op `{op}` on row {row_idx}")]
    MissingUpstream {
        op: String,
        node: NodeId,
        row_idx: usize,
    },

    #[error("missing source `{source_name}` factory for node `{node}`")]
    MissingSourceFactory { source_name: String, node: NodeId },
}

impl CompiledGraph {
    /// Compile a graph: validate wire version, look up factories, toposort.
    pub fn compile(graph: Graph) -> Result<Self, CompileError> {
        if graph.wire_version != crate::WIRE_VERSION {
            return Err(CompileError::WireVersion {
                got: graph.wire_version,
                want: crate::WIRE_VERSION,
            });
        }

        let order = toposort(&graph)?;
        let mut sources = HashMap::new();
        let mut ops = HashMap::new();

        for node_id in &order {
            let node = graph
                .nodes
                .get(node_id)
                .expect("toposort returns only known node ids");
            match node {
                Node::Source { spec } => {
                    let reg = find_source(&spec.name).ok_or_else(|| {
                        CompileError::UnknownSource {
                            source_name: spec.name.clone(),
                            node: node_id.clone(),
                        }
                    })?;
                    let params = spec
                        .params
                        .try_into_literal_map()
                        .unwrap_or_else(ParamMap::new);
                    let source = (reg.factory)(&params).map_err(|error| {
                        CompileError::SourceFactory {
                            source_name: spec.name.clone(),
                            error,
                        }
                    })?;
                    sources.insert(node_id.clone(), source);
                }
                Node::Op { op_id, params, .. } => {
                    let reg = find_op(op_id).ok_or_else(|| CompileError::UnknownOp {
                        op: op_id.clone(),
                        node: node_id.clone(),
                    })?;
                    // Build the op once if all params are literal; otherwise
                    // defer to per-row construction.
                    let pre_built = if let Some(literals) = params.try_into_literal_map() {
                        let op =
                            (reg.factory)(&literals).map_err(|source| CompileError::OpFactory {
                                op: op_id.clone(),
                                source,
                            })?;
                        Some(op)
                    } else {
                        None
                    };
                    ops.insert(node_id.clone(), pre_built);
                }
            }
        }

        let mut sinks = Vec::with_capacity(graph.outputs.len());
        for binding in &graph.outputs {
            let reg = find_sink(&binding.sink.name).ok_or_else(|| CompileError::UnknownSink {
                sink: binding.sink.name.clone(),
                alias: binding.alias.clone(),
            })?;
            let params = binding
                .sink
                .params
                .try_into_literal_map()
                .unwrap_or_else(ParamMap::new);
            let sink = (reg.factory)(&params).map_err(|error| CompileError::SinkFactory {
                sink: binding.sink.name.clone(),
                error,
            })?;
            sinks.push(sink);
        }

        Ok(Self {
            graph,
            order,
            sources,
            ops,
            sinks,
        })
    }

    /// Convenience for tests: execute a row where every source is given the
    /// same byte slice. Real execution comes through the polars-cv bridge.
    pub fn execute_row_simple(
        &self,
        row_idx: usize,
        source_bytes: &[u8],
    ) -> Result<Vec<SinkRowOutput>, ExecError> {
        let mut outputs: HashMap<NodeId, NodeOutput> = HashMap::with_capacity(self.order.len());
        let ctx = ExecCtx::new(row_idx);

        for node_id in &self.order {
            let node = &self.graph.nodes[node_id];
            match node {
                Node::Source { spec } => {
                    let source = self.sources.get(node_id).ok_or_else(|| {
                        ExecError::MissingSourceFactory {
                            source_name: spec.name.clone(),
                            node: node_id.clone(),
                        }
                    })?;
                    let inputs = SourceInputs::Bytes(source_bytes);
                    let out = source.produce(row_idx, &inputs).map_err(|error| {
                        ExecError::Source {
                            source_name: spec.name.clone(),
                            row_idx,
                            error,
                        }
                    })?;
                    outputs.insert(node_id.clone(), out);
                }
                Node::Op {
                    op_id,
                    params,
                    inputs,
                } => {
                    let op_handle = match self.ops.get(node_id).and_then(|h| h.clone()) {
                        Some(h) => h,
                        None => {
                            // Per-row factory construction; in tests params
                            // are literal so this branch is dead.
                            let literals = params
                                .try_into_literal_map()
                                .unwrap_or_else(ParamMap::new);
                            let reg =
                                find_op(op_id).ok_or_else(|| ExecError::Op {
                                    op: op_id.clone(),
                                    row_idx,
                                    error: OpError::Failed {
                                        op: "lookup",
                                        message: format!("op `{op_id}` not registered"),
                                    },
                                })?;
                            (reg.factory)(&literals).map_err(|error| ExecError::Op {
                                op: op_id.clone(),
                                row_idx,
                                error,
                            })?
                        }
                    };
                    let upstream_outputs: Vec<(&'static str, &NodeOutput)> = match inputs {
                        Inputs::Single { node } => {
                            let upstream = outputs.get(node).ok_or_else(|| {
                                ExecError::MissingUpstream {
                                    op: op_id.clone(),
                                    node: node.clone(),
                                    row_idx,
                                }
                            })?;
                            vec![(SINGLE_PORT, upstream)]
                        }
                        Inputs::Named { ports } => {
                            let mut v = Vec::with_capacity(ports.len());
                            for (name, n) in ports {
                                let upstream = outputs.get(n).ok_or_else(|| {
                                    ExecError::MissingUpstream {
                                        op: op_id.clone(),
                                        node: n.clone(),
                                        row_idx,
                                    }
                                })?;
                                // SAFETY: ports are owned by the IR for the
                                // lifetime of execution; the &'static str
                                // is a leaked clone of the port name. For
                                // tests this is fine; the bridge layer uses
                                // a lookup table to avoid the leak.
                                let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
                                v.push((leaked, upstream));
                            }
                            v
                        }
                    };
                    let op_inputs = match inputs {
                        Inputs::Single { .. } => OpInputs::single(upstream_outputs[0].1),
                        Inputs::Named { .. } => OpInputs::named(&upstream_outputs),
                    };
                    let op_id_owned = op_id.clone();
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        op_handle.execute(&ctx, &op_inputs)
                    }));
                    let out = match result {
                        Ok(Ok(out)) => out,
                        Ok(Err(error)) => {
                            return Err(ExecError::Op {
                                op: op_id_owned,
                                row_idx,
                                error,
                            })
                        }
                        Err(panic) => {
                            let message = panic_message(panic);
                            return Err(ExecError::OpPanicked {
                                op: op_id_owned,
                                row_idx,
                                message,
                            });
                        }
                    };
                    outputs.insert(node_id.clone(), out);
                }
            }
        }

        let mut sink_outputs = Vec::with_capacity(self.graph.outputs.len());
        for (binding, sink) in self.graph.outputs.iter().zip(&self.sinks) {
            let upstream = outputs.get(&binding.node).ok_or_else(|| ExecError::Sink {
                sink: binding.sink.name.clone(),
                row_idx,
                error: SinkError::Failed {
                    name: "lookup",
                    row_idx,
                    message: format!(
                        "sink `{}` references unknown node `{}`",
                        binding.sink.name, binding.node
                    ),
                },
            })?;
            let out = sink.consume(row_idx, upstream).map_err(|error| ExecError::Sink {
                sink: binding.sink.name.clone(),
                row_idx,
                error,
            })?;
            sink_outputs.push(out);
        }

        Ok(sink_outputs)
    }
}

const SINGLE_PORT: &str = "input";

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = panic.downcast_ref::<String>() {
        return s.clone();
    }
    "<non-string panic>".into()
}

fn toposort(graph: &Graph) -> Result<Vec<NodeId>, CompileError> {
    use std::collections::HashSet;

    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut on_stack: HashSet<NodeId> = HashSet::new();
    let mut order: Vec<NodeId> = Vec::with_capacity(graph.nodes.len());

    fn dfs(
        node_id: &NodeId,
        graph: &Graph,
        visited: &mut std::collections::HashSet<NodeId>,
        on_stack: &mut std::collections::HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> Result<(), CompileError> {
        if visited.contains(node_id) {
            return Ok(());
        }
        if !on_stack.insert(node_id.clone()) {
            return Err(CompileError::Cycle {
                node: node_id.clone(),
            });
        }
        let upstreams: Vec<NodeId> = match graph.nodes.get(node_id) {
            Some(Node::Op { inputs, .. }) => inputs.upstreams().into_iter().cloned().collect(),
            Some(Node::Source { .. }) | None => Vec::new(),
        };
        for u in upstreams {
            dfs(&u, graph, visited, on_stack, order)?;
        }
        on_stack.remove(node_id);
        visited.insert(node_id.clone());
        order.push(node_id.clone());
        Ok(())
    }

    for node_id in graph.nodes.keys() {
        dfs(node_id, graph, &mut visited, &mut on_stack, &mut order)?;
    }

    Ok(order)
}
