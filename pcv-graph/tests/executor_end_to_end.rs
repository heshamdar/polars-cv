//! End-to-end executor test: source → identity op → sink, no Polars.
//!
//! Wires together everything in pcv-graph that the polars-cv bridge layer
//! will exercise: IR construction, JSON wire round-trip, compile, run a row,
//! verify the sink output. Once this passes we have a working Polars-free
//! pipeline runtime that the bridge can plug into.

use indexmap::IndexMap;
use pcv_graph::ir::{
    Graph, Inputs, Node, OutputBinding, SerializedParams, SinkSpec, SourceSpec,
};
use pcv_graph::sink::SinkRowOutput;
use pcv_graph::wire;
use pcv_graph::CompiledGraph;

fn build_test_graph() -> Graph {
    let mut g = Graph::new();
    g.nodes.insert(
        "src".into(),
        Node::Source {
            spec: SourceSpec {
                name: "bytes_passthrough".into(),
                params: SerializedParams::new(),
                input_column: Some("img".into()),
            },
        },
    );
    g.nodes.insert(
        "id".into(),
        Node::Op {
            op_id: "identity".into(),
            params: SerializedParams::new(),
            inputs: Inputs::Single { node: "src".into() },
        },
    );
    g.outputs.push(OutputBinding {
        alias: "_output".into(),
        node: "id".into(),
        sink: SinkSpec {
            name: "buffer_bytes".into(),
            params: SerializedParams::new(),
        },
        planned: None,
    });
    g
}

#[test]
fn source_identity_sink_roundtrips_bytes() {
    let graph = build_test_graph();
    let compiled = CompiledGraph::compile(graph).expect("compile should succeed");

    let payload: Vec<u8> = (0u8..32).collect();
    let outputs = compiled
        .execute_row_simple(0, &payload)
        .expect("execute_row should succeed");

    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        SinkRowOutput::Bytes(bytes) => {
            assert_eq!(bytes, &payload, "sink should round-trip the source bytes");
        }
        other => panic!("expected SinkRowOutput::Bytes, got {other:?}"),
    }
}

#[test]
fn graph_survives_json_roundtrip_then_executes() {
    let graph = build_test_graph();
    let bytes = wire::encode(&graph, wire::WireFormat::Json).expect("encode");
    let decoded = wire::decode(&bytes, wire::WireFormat::Json).expect("decode");

    let compiled = CompiledGraph::compile(decoded).expect("compile decoded graph");
    let payload = vec![7u8; 5];
    let outputs = compiled
        .execute_row_simple(0, &payload)
        .expect("execute decoded graph");
    match &outputs[0] {
        SinkRowOutput::Bytes(bytes) => assert_eq!(bytes, &payload),
        other => panic!("expected SinkRowOutput::Bytes, got {other:?}"),
    }
}

#[test]
fn compile_rejects_unknown_op() {
    let mut graph = Graph::new();
    graph.nodes.insert(
        "src".into(),
        Node::Source {
            spec: SourceSpec {
                name: "bytes_passthrough".into(),
                params: SerializedParams::new(),
                input_column: None,
            },
        },
    );
    graph.nodes.insert(
        "ghost".into(),
        Node::Op {
            op_id: "no_such_op".into(),
            params: SerializedParams::new(),
            inputs: Inputs::Single { node: "src".into() },
        },
    );
    graph.outputs.push(OutputBinding {
        alias: "_output".into(),
        node: "ghost".into(),
        sink: SinkSpec {
            name: "buffer_bytes".into(),
            params: SerializedParams::new(),
        },
        planned: None,
    });

    let err = match CompiledGraph::compile(graph) {
        Ok(_) => panic!("compile should reject unknown op"),
        Err(e) => e,
    };
    assert!(
        format!("{err}").contains("no_such_op"),
        "error should name the missing op: {err}"
    );
}

#[test]
fn executor_sees_planned_wire_version() {
    // Ensure the wire version travels through the graph and matches the crate
    // constant — this is the one assertion the Python side uses to detect a
    // stale plugin / regenerated ops.py mismatch.
    let g = build_test_graph();
    assert_eq!(g.wire_version, pcv_graph::WIRE_VERSION);
}

// Suppress unused-import warning when `IndexMap` isn't referenced by a
// particular test config; we keep it imported because tests below add named
// inputs in follow-up commits.
#[allow(dead_code)]
fn _silence_imports(_: IndexMap<String, String>) {}
