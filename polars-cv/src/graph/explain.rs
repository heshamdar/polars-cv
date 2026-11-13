//! A graph described as it will run: its nodes in execution order, each op
//! with the state the planner gives its result, the fused segments, the
//! outputs with their planned dtypes, and the engine passes.
//!
//! [`describe`] builds the one [`Description`]; `PipelineGraph.explain()`
//! prints its [`Display`](std::fmt::Display) form and `show_graph()` draws
//! its JSON, so the text and the drawing cannot disagree, and neither can
//! describe a graph other than the one that executes. Nothing in it is
//! computed for the description alone: the states are the planner's
//! ([`planned_node_states`]), the output dtypes the schema's
//! ([`dtype_for_output`](crate::graph::dtype_for_output)), the struct decision
//! the plugin's ([`UnifiedGraph::returns_struct`]), the fused segments the row
//! loop's grouping (a run of [`GraphStep::Buffer`] steps, fixed when the graph
//! compiles), and the engine passes the graph's `OptConfig`.
//!
//! One limit, stated where it is shown: the graph is planned without the
//! input columns' types, which polars supplies only when the query is
//! planned. A source that takes its element type or sizes from its column
//! shows them unknown here, and an output whose type the column decides says
//! so instead of guessing.

use std::fmt;

use polars::prelude::*;
use serde::Serialize;
use serde_json::Value;

use crate::graph::compiled::planned_node_states;
use crate::graph::step::GraphStep;
use crate::graph::types::{NodeInput, UnifiedGraph, ERROR_FIELD};
use crate::plan::{PlannedShape, State};

/// A graph as it will run.
#[derive(Debug, Serialize)]
pub(crate) struct Description {
    /// In execution order.
    pub nodes: Vec<NodeDesc>,
    /// Sorted by name, as the result's struct fields are.
    pub outputs: Vec<OutputDesc>,
    /// The result is a struct, a field per output (and `_error`), rather
    /// than the one output's column.
    pub returns_struct: bool,
    /// Non-default per-row policies, as `(name, value)`.
    pub policies: Vec<(&'static str, &'static str)>,
    /// The engine passes that run inside each fused segment.
    pub engine_passes: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct NodeDesc {
    /// The node's wire id (`n0`, …), which operands name it by.
    pub id: String,
    pub input: InputDesc,
    /// The state the node starts from.
    pub state: String,
    pub ops: Vec<OpDesc>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InputDesc {
    /// Decodes plugin input `column` (its expression) through `source`.
    Decode { column: String, source: String },
    /// Reads another node's output.
    Node(String),
}

#[derive(Debug, Serialize)]
pub(crate) struct OpDesc {
    /// `name(field=value, ...)`, a per-row value as its expression.
    pub label: String,
    /// Runs in a fused segment with the buffer ops beside it.
    pub fused: bool,
    /// The planned state of its result.
    pub state: String,
    /// The other nodes it reads.
    pub reads: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OutputDesc {
    /// Its struct field; `None` for a lone output that is the column itself.
    pub name: Option<String>,
    /// `None` for the `_error` field, which no node produces.
    pub node: Option<String>,
    pub sink: Option<String>,
    /// The planned Polars dtype; `None` when the input column decides it.
    pub dtype: Option<String>,
}

/// Describe `graph_json`. `inputs[i]` names plugin input `i`, the expression
/// a slot or a root's column reads.
pub(crate) fn describe(graph_json: &str, inputs: &[String]) -> PolarsResult<Description> {
    let compiled = crate::graph::get_or_compile(graph_json)?;
    let graph = compiled.graph();
    let names = Names(inputs);

    let mut nodes: Vec<NodeDesc> = Vec::new();
    planned_node_states(graph, &[], |node_id, op, state| {
        let node = &graph.nodes[node_id];
        match op {
            None => nodes.push(NodeDesc {
                id: node_id.to_string(),
                input: match &node.input {
                    NodeInput::Decode { column, source } => InputDesc::Decode {
                        column: names.input(*column),
                        source: call(&crate::plan::wire(source), "format", &names)?,
                    },
                    NodeInput::Node(up) => InputDesc::Node(up.clone()),
                },
                state: state_text(state),
                ops: Vec::with_capacity(node.ops.len()),
            }),
            Some(i) => {
                let op = &node.ops[i];
                let desc = nodes.last_mut().expect("a node's entry precedes its ops");
                desc.ops.push(OpDesc {
                    label: call(&crate::plan::wire(op), "op", &names)?,
                    // The row loop fuses a run of buffer steps
                    // (`PendingSegment`); any other step runs on its own.
                    fused: matches!(op, GraphStep::Buffer(_)),
                    state: state_text(state),
                    reads: op.operands().iter().map(|n| n.0.clone()).collect(),
                });
            }
        }
        Ok(())
    })?;

    let returns_struct = graph.returns_struct();
    let mut outputs = Vec::new();
    for (alias, spec) in crate::graph::resolved_output_specs(graph, &[])? {
        outputs.push(OutputDesc {
            name: returns_struct.then(|| alias.clone()),
            sink: Some(call(&crate::plan::wire(&spec.sink), "format", &names)?),
            dtype: output_dtype(graph, &spec)?,
            node: Some(spec.node),
        });
    }
    if graph.adds_error_field() {
        outputs.push(OutputDesc {
            name: Some(ERROR_FIELD.to_string()),
            node: None,
            sink: None,
            dtype: Some(DataType::String.to_string()),
        });
    }

    let mut policies = Vec::new();
    for (name, value) in [
        ("on_error", graph.on_error.spelling()),
        ("on_null_param", graph.on_null_param.spelling()),
    ] {
        if value != "raise" {
            policies.push((name, value));
        }
    }

    let toggles = serde_json::to_value(graph.opt).expect("OptConfig serializes");
    let engine_passes = view_buffer::ENGINE_PASSES
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| toggles.get(name) == Some(&Value::Bool(true)))
        .collect();

    Ok(Description {
        nodes,
        outputs,
        returns_struct,
        policies,
        engine_passes,
    })
}

/// An output's planned dtype, or `None` when its input column decides it.
///
/// The column only ever fills in what the plan leaves unknown, so a type the
/// schema can give from the plan alone is the type; when it cannot, and the
/// node's lineage takes facts from its column, the column decides. Any other
/// refusal is the graph's error (`check()` raised it already).
fn output_dtype(
    graph: &UnifiedGraph,
    spec: &crate::graph::types::OutputSpec,
) -> PolarsResult<Option<String>> {
    match crate::graph::dtype_for_output(spec) {
        Ok(dtype) => Ok(Some(dtype.to_string())),
        Err(_) if graph.column_facts_pending(&spec.node).is_some() => Ok(None),
        Err(e) => Err(e),
    }
}

impl fmt::Display for Description {
    /// One line per node, op and output, the planned states aligned.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows: Vec<(String, &str)> = Vec::new();
        for node in &self.nodes {
            let head = match &node.input {
                InputDesc::Decode { column, source } => {
                    format!("{}: decode {column} as {source}", node.id)
                }
                InputDesc::Node(up) => format!("{}: reads {up}", node.id),
            };
            rows.push((head, &node.state));
            for op in &node.ops {
                let mark = if op.fused { '|' } else { ' ' };
                rows.push((format!("  {mark} {}", op.label), &op.state));
            }
        }
        let width = rows
            .iter()
            .map(|(h, _)| h.chars().count())
            .max()
            .unwrap_or(0);
        for (head, state) in &rows {
            writeln!(f, "{head:<width$}  -> {state}")?;
        }

        for out in &self.outputs {
            let label = match &out.name {
                Some(name) => format!("output '{name}'"),
                None => "output".to_string(),
            };
            let source = match (&out.node, &out.sink) {
                (Some(node), Some(sink)) => format!(": {node} as {sink}"),
                _ => " (each failing row's message)".to_string(),
            };
            let dtype = out.dtype.as_deref().unwrap_or("set by the input column");
            writeln!(f, "{label}{source} :: {dtype}")?;
        }
        if !self.policies.is_empty() {
            let policies: Vec<String> = self
                .policies
                .iter()
                .map(|(name, value)| format!("{name}='{value}'"))
                .collect();
            writeln!(f, "policies: {}", policies.join(", "))?;
        }
        // Only where there is a segment for them to run in.
        if self.nodes.iter().any(|n| n.ops.iter().any(|op| op.fused)) {
            let passes = if self.engine_passes.is_empty() {
                "none".to_string()
            } else {
                self.engine_passes.join(", ")
            };
            writeln!(
                f,
                "engine passes (on each '|' segment, planned per row layout): {passes}"
            )?;
        }
        Ok(())
    }
}

/// The plugin inputs' names.
struct Names<'a>(&'a [String]);

impl Names<'_> {
    fn input(&self, i: usize) -> String {
        self.0
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("input {i}"))
    }
}

/// `domain dtype [sizes]`, `?` for a size or rank not known at plan time.
fn state_text(s: &State) -> String {
    let sizes = |sizes: &[Option<usize>]| {
        sizes
            .iter()
            .map(|d| d.map_or("?".to_string(), |d| d.to_string()))
            .collect::<Vec<_>>()
    };
    let shape = match &s.shape {
        PlannedShape::Ranked(dims) => format!("[{}]", sizes(dims).join(", ")),
        PlannedShape::Unranked { .. } => {
            let mut parts = sizes(s.shape.sizes());
            parts.push("...".into());
            format!("[{}]", parts.join(", "))
        }
    };
    format!("{} {} {shape}", s.domain.name(), s.dtype.as_str())
}

/// A wire object as a call: its `name_key` field the name, the rest keyword
/// arguments. Every op, source and sink is written this way
/// ([`crate::plan::wire`]), so another shape is a bug, not a case.
fn call(value: &Value, name_key: &str, names: &Names<'_>) -> PolarsResult<String> {
    let name = value
        .as_object()
        .and_then(|fields| fields.get(name_key))
        .and_then(Value::as_str)
        .ok_or_else(
            || polars_err!(ComputeError: "internal: a wire object without {name_key:?}: {value}"),
        )?;
    let fields = value.as_object().expect("checked above");
    let args = fields
        .iter()
        .filter(|(k, _)| k.as_str() != name_key)
        .map(|(k, v)| Ok(format!("{k}={}", literal(v, names)?)))
        .collect::<PolarsResult<Vec<String>>>()?;
    Ok(format!("{name}({})", args.join(", ")))
}

/// A wire value as Python would write it: a slot as the expression it reads,
/// `this` arithmetic as its own repr.
fn literal(value: &Value, names: &Names<'_>) -> PolarsResult<String> {
    let joined = |items: Vec<PolarsResult<String>>| -> PolarsResult<String> {
        Ok(items
            .into_iter()
            .collect::<PolarsResult<Vec<_>>>()?
            .join(", "))
    };
    Ok(match value {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => {
            format!(
                "[{}]",
                joined(items.iter().map(|v| literal(v, names)).collect())?
            )
        }
        Value::Object(fields) => {
            if let (1, Some(slot)) = (fields.len(), fields.get("$slot").and_then(Value::as_u64)) {
                return Ok(names.input(slot as usize));
            }
            if let (1, Some(tree)) = (fields.len(), fields.get("$this")) {
                // The graph compiled, so its `this` trees parsed; one that
                // does not is a bug, said as one.
                return view_buffer::mode::ThisExpr::from_json(tree)
                    .map(|e| e.to_string())
                    .map_err(|e| polars_err!(ComputeError: "internal: `this` tree: {e}"));
            }
            let items = fields
                .iter()
                .map(|(k, v)| Ok(format!("'{k}': {}", literal(v, names)?)))
                .collect();
            format!("{{{}}}", joined(items)?)
        }
    })
}
