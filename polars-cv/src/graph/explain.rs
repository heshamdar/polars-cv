//! A graph as text: what will run, in the order it runs, with the state the
//! planner gives each op (`PipelineGraph.explain()`, CR-148).
//!
//! Rendered here, from the compiled graph and the one planner
//! ([`planned_node_states`]), so the text cannot describe a graph other
//! than the one that executes. Nothing in it is computed for the text alone:
//! the states are the planner's, the output dtypes the schema's, the fused
//! segments the row loop's grouping (a run of [`GraphStep::Buffer`] steps,
//! fixed when the graph compiles), and the engine passes the graph's
//! `OptConfig`.

use std::fmt::Write as _;

use polars::prelude::*;
use serde_json::Value;

use crate::graph::compiled::planned_node_states;
use crate::graph::step::GraphStep;
use crate::graph::types::NodeInput;
use crate::plan::{PlannedShape, State};

/// The text for `graph_json`. `inputs[i]` names plugin input `i` (the
/// expression a slot or a root's column reads); an input without a name is
/// shown as `input i`.
pub(crate) fn explain(graph_json: &str, inputs: &[String]) -> PolarsResult<String> {
    let compiled = crate::graph::get_or_compile(graph_json)?;
    let graph = compiled.graph();
    let names = Names(inputs);

    // The planner's states, as it makes them: entering each node, then after
    // each op.
    let mut entering = std::collections::HashMap::<String, State>::new();
    let mut after = std::collections::HashMap::<(String, usize), State>::new();
    planned_node_states(graph, &[], |node, op, state| match op {
        None => {
            entering.insert(node.to_string(), state.clone());
        }
        Some(i) => {
            after.insert((node.to_string(), i), state.clone());
        }
    })?;

    // Lines as (head, state): the heads are padded to one width so the
    // states line up.
    let mut lines: Vec<(String, Option<String>)> = Vec::new();
    for node_id in graph.topological_order() {
        let node = &graph.nodes[node_id];
        let head = match &node.input {
            NodeInput::Decode { column, source } => format!(
                "{node_id}: decode {} as {}",
                names.input(*column),
                call(&crate::plan::wire(source), "format", &names)
            ),
            NodeInput::Node(up) => format!("{node_id}: reads {up}"),
        };
        lines.push((head, Some(state(&entering[node_id]))));
        for (i, op) in node.ops.iter().enumerate() {
            // A buffer step joins the fused segment the row loop forms
            // (`PendingSegment`); any other step runs on its own.
            let mark = if matches!(op, GraphStep::Buffer(_)) {
                "|"
            } else {
                " "
            };
            let text = call(&crate::plan::wire(op), "op", &names);
            lines.push((
                format!("  {mark} {text}"),
                Some(state(&after[&(node_id.clone(), i)])),
            ));
        }
    }

    for (alias, spec) in crate::graph::resolved_output_specs(graph, &[])? {
        let label = if graph.is_single_output() {
            "output".to_string()
        } else {
            format!("output '{alias}'")
        };
        let dtype = match graph.column_facts_pending(&spec.node) {
            Some(_) => "from the input column".to_string(),
            None => crate::graph::dtype_for_output(&spec)?.to_string(),
        };
        let sink = call(&crate::plan::wire(&spec.sink), "format", &names);
        lines.push((format!("{label}: {} as {sink} :: {dtype}", spec.node), None));
    }

    let mut policies = Vec::new();
    for (name, value) in [
        ("on_error", graph.on_error.spelling()),
        ("on_null_param", graph.on_null_param.spelling()),
    ] {
        if value != "raise" {
            policies.push(format!("{name}='{value}'"));
        }
    }
    if !policies.is_empty() {
        lines.push((format!("policies: {}", policies.join(", ")), None));
    }

    // The engine passes rewrite inside each fused segment as it is planned
    // for a row's layout, so their effect is not known here: name the ones
    // that will run.
    let toggles = serde_json::to_value(graph.opt).expect("OptConfig serializes");
    let enabled: Vec<&str> = view_buffer::ENGINE_PASSES
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| toggles.get(name) == Some(&Value::Bool(true)))
        .collect();
    let engine = if enabled.is_empty() {
        "none".to_string()
    } else {
        enabled.join(", ")
    };
    lines.push((
        format!("engine passes (on each '|' segment, planned per row layout): {engine}"),
        None,
    ));

    let width = lines
        .iter()
        .filter(|(_, s)| s.is_some())
        .map(|(h, _)| h.chars().count())
        .max()
        .unwrap_or(0);
    let mut text = String::new();
    for (head, state) in lines {
        match state {
            Some(state) => {
                let _ = writeln!(text, "{head:<width$}  -> {state}");
            }
            None => {
                let _ = writeln!(text, "{head}");
            }
        }
    }
    Ok(text)
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
fn state(s: &State) -> String {
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
/// arguments.
fn call(value: &Value, name_key: &str, names: &Names<'_>) -> String {
    let Value::Object(fields) = value else {
        return literal(value, names);
    };
    let name = fields.get(name_key).and_then(Value::as_str).unwrap_or("?");
    let args: Vec<String> = fields
        .iter()
        .filter(|(k, _)| k.as_str() != name_key)
        .map(|(k, v)| format!("{k}={}", literal(v, names)))
        .collect();
    format!("{name}({})", args.join(", "))
}

/// A wire value as Python would write it: a slot as the expression it reads,
/// `this` arithmetic as its own repr.
fn literal(value: &Value, names: &Names<'_>) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|v| literal(v, names))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(fields) => {
            if let (1, Some(slot)) = (fields.len(), fields.get("$slot").and_then(Value::as_u64)) {
                return names.input(slot as usize);
            }
            if let (1, Some(tree)) = (fields.len(), fields.get("$this")) {
                return view_buffer::mode::ThisExpr::from_json(tree)
                    .map(|e| e.to_string())
                    .unwrap_or_else(|_| tree.to_string());
            }
            format!(
                "{{{}}}",
                fields
                    .iter()
                    .map(|(k, v)| format!("'{k}': {}", literal(v, names)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}
