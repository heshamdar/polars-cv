//! End-to-end tests of compiling, caching and running graphs.

use polars::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

use crate::graph::node_output::NodeOutput;
use view_buffer::ops::Domain;
use view_buffer::{ViewBuffer, ViewDto};

use crate::catalog::TypedOp;
use crate::params::ParamCtx;

use crate::graph::step::GraphStep;

use super::cache::*;
use super::compile::*;
use super::run::*;

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

use crate::catalog::graph::Role;
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
/// does in `namespaces/exec.rs`, and the coverage assertion at the end of
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
            Role::NormalizePercentileMasked { .. } => "NormalizePercentileMasked",
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
    let src = include_str!("tests.rs");
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
    for op in crate::catalog::TypedOp::samples() {
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

    // NormalizePercentileMasked: a buffer's window from its own set
    // pixels (all 255): a constant selection, the empty window's zeros.
    let out = exec(
        r#"{
                "version": 2,
                "nodes": {
                    "n0": {"input": {"decode": {"column": 0, "source": {"format": "blob"}}}},
                    "n1": {"input": {"node": "n0"},
                           "ops": [{"op": "normalize_percentile_masked", "mask": "n0",
                                    "low": 2.0, "high": 98.0}]}
                },
                "outputs": {"_output": {"node": "n1", "sink": {"format": "blob"}}}
            }"#,
        &[Series::new("b".into(), &[mask_blob()])],
    );
    let normalized = ViewBuffer::from_blob(out.binary().unwrap().get(0).unwrap()).unwrap();
    assert_eq!(normalized.shape(), &[4, 4, 1]);
    assert_eq!(normalized.to_contiguous().as_slice::<f32>(), &[0.0; 16]);

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
    let contour_av = crate::namespaces::contour::contour_to_anyvalue(&contour);
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
/// call, not once per row, and replanned when any of them changes.
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
        let baseline = copies_in_call(&graph(ops, r#"{"format": "numpy"}"#), &input, image_bytes);
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
/// single-chunk frame on the in-memory engine is not single-threaded,
/// and the rows still come back in order.
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

// --- One operand path ---

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
        let expected = format!("node 'n1': {name}() reads node 'ghost', which is not in the graph");
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
        &[crate::namespaces::contour::contour_to_anyvalue(&contour)],
        false,
    )
    .unwrap();
    let contours = Series::from_any_values("c".into(), &[AnyValue::List(inner)], false).unwrap();
    let column = |domain: Domain, null: bool| -> (Series, &'static str) {
        match (domain, null) {
            (Domain::Buffer, false) => (Series::new("b".into(), &[mask_blob()]), "blob"),
            (Domain::Buffer, true) => (Series::full_null("b".into(), 1, &DataType::Binary), "blob"),
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
