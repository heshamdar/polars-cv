//! The logical optimisation passes that rewrite one node's op list.
//!
//! Each returns the node's new op order as indices into the old list (a
//! subset for a deletion, a permutation for a reorder), or `None` when nothing
//! changes. Python applies it with `Pipeline._replay`, which appends the kept
//! ops again from the node's first entering state, so a pass decides *what*
//! the ops become and never maintains per-position state itself.
//!
//! The decisions read only the op contracts (`GraphStep`) and the per-boundary
//! states the planner recorded; there is no op name in here.

use pyo3::prelude::*;
use view_buffer::{IdentityRule, SpatialDependency};

use crate::graph::step::GraphStep;
use crate::plan::State;
use crate::py_value_error;

/// Every logical optimisation pass, by the name `OptFlags` knows it by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalPass {
    /// Share a common leading op run across sibling pipelines (graph scope;
    /// applied by the Python graph, which holds the expression identities).
    CommonSubexpressionElimination,
    /// Delete ops that provably change nothing.
    IdentityElimination,
    /// Hoist a crop earlier past the pointwise ops before it.
    SpatialWindowPushdown,
}

view_buffer::naming::named_variants!(LogicalPass: "A logical optimisation pass (``OptFlags`` field names)." {
    "common_subexpression_elimination" => CommonSubexpressionElimination,
    "identity_elimination" => IdentityElimination,
    "spatial_window_pushdown" => SpatialWindowPushdown,
});

impl LogicalPass {
    /// One line on what the pass does (the catalogue's `summary`).
    pub fn summary(self) -> &'static str {
        match self {
            LogicalPass::CommonSubexpressionElimination => {
                "Share a common leading op run across sibling pipelines that read the same source column into one upstream node."
            }
            LogicalPass::IdentityElimination => {
                "Delete no-op operations — a zero pad, a same-dtype cast, a full-frame crop — that preserve their input byte for byte."
            }
            LogicalPass::SpatialWindowPushdown => {
                "Hoist a spatial window (a crop/ROI) earlier past ops it commutes with, so upstream ops process fewer pixels."
            }
        }
    }
}

/// One optimisation pass, as the catalogue describes it.
#[derive(serde::Serialize)]
struct PassDesc {
    name: &'static str,
    /// `logical` (rewrites the graph before serialization) or `engine` (a
    /// per-row lowering in view-buffer, toggled through `OptConfig`).
    tier: &'static str,
    summary: &'static str,
}

/// Every optimisation pass, in the order they apply: the logical ones
/// (`LogicalPass`), then the engine's (`view_buffer::ENGINE_PASSES`). The
/// Python `OptFlags` and `OPTIMIZATION_PASSES` are generated from it.
pub fn pass_catalog_json() -> String {
    let logical = LogicalPass::NAMED.iter().map(|(name, pass)| PassDesc {
        name,
        tier: "logical",
        summary: pass.summary(),
    });
    let engine = view_buffer::ENGINE_PASSES
        .iter()
        .map(|(name, summary)| PassDesc {
            name,
            tier: "engine",
            summary,
        });
    let passes: Vec<PassDesc> = logical.chain(engine).collect();
    let mut text = serde_json::to_string_pretty(&passes).expect("the catalogue serializes");
    text.push('\n');
    text
}

/// The pass catalogue (see [`pass_catalog_json`]).
#[pyfunction]
pub(crate) fn pass_catalog() -> String {
    pass_catalog_json()
}

/// One node's ops and what the planner knows about them.
pub(crate) struct Node<'a> {
    pub ops: &'a [String],
    /// `ops.len() + 1` states: entering each op, then the final one.
    pub states: &'a [State],
    /// Op boundaries carrying a shape declaration.
    pub assertions: &'a [usize],
}

/// Run a node-scope pass. `Err` for a graph-scope pass.
pub(crate) fn run(pass: LogicalPass, node: &Node<'_>) -> Result<Option<Vec<usize>>, String> {
    if node.states.len() != node.ops.len() + 1 {
        return Err(format!(
            "{} ops need {} boundary states, got {}",
            node.ops.len(),
            node.ops.len() + 1,
            node.states.len()
        ));
    }
    let order = match pass {
        LogicalPass::CommonSubexpressionElimination => {
            return Err(
                "common_subexpression_elimination rewrites the graph, not one node".to_string(),
            )
        }
        LogicalPass::IdentityElimination => eliminate_identities(node)?,
        LogicalPass::SpatialWindowPushdown => hoist_spatial_windows(node)?,
    };
    let unchanged = order.len() == node.ops.len() && order.iter().enumerate().all(|(i, &o)| i == o);
    Ok((!unchanged).then_some(order))
}

/// The ops that are not provable no-ops, in order.
///
/// A node carrying any assertion is left alone, so no assertion key has to be
/// re-derived across a deletion.
fn eliminate_identities(node: &Node<'_>) -> Result<Vec<usize>, String> {
    let all: Vec<usize> = (0..node.ops.len()).collect();
    if !node.assertions.is_empty() {
        return Ok(all);
    }
    let mut kept = Vec::with_capacity(all.len());
    for i in all {
        if !is_identity(node, i)? {
            kept.push(i);
        }
    }
    Ok(kept)
}

/// Whether op `i` is removable: its identity rule, evaluated against the
/// state entering it and the one it leaves. Any unknown keeps the op.
fn is_identity(node: &Node<'_>, i: usize) -> Result<bool, String> {
    let op_json = &node.ops[i];
    let op: crate::ops::TypedOp = serde_json::from_str(op_json).map_err(|e| e.to_string())?;
    let rule = crate::resolve_op_from_json(op_json)?.identity_rule();
    // A verdict resting on a parameter's literal value cannot be proven when
    // that parameter is per-row: the op keeps computing on rows where it is
    // not the identity value.
    let mut per_row = Vec::new();
    op.visit_slots(&mut |name, _| per_row.push(name));
    if rule.deciding_params().iter().any(|p| per_row.contains(p)) {
        return Ok(false);
    }
    let (entering, leaving) = (&node.states[i], &node.states[i + 1]);
    Ok(match rule {
        IdentityRule::Never => false,
        IdentityRule::Always { .. } => true,
        IdentityRule::WhenDtypePreserved => {
            entering.dtype != "auto" && leaving.dtype == entering.dtype
        }
        IdentityRule::WhenShapePreserved { .. } => {
            // Hints that may rest on a declaration are a claim, not a fact.
            // A declaration anywhere in the node's lineage makes its sizes
            // possibly a claim; the final state records whether one did.
            let declared = node.states.last().is_some_and(|s| s.declared);
            let Some(ndim) = entering.ndim.filter(|_| !declared) else {
                return Ok(false);
            };
            let dims: Vec<Option<i64>> = (0..ndim)
                .map(|axis| if axis < 2 { entering.dims[axis] } else { None })
                .collect();
            match crate::infer_shape(op_json, &dims)? {
                Some(out) => shape_preserved(&out, &dims),
                None => false,
            }
        }
    })
}

/// Whether an inferred output shape equals the shape entering the op.
///
/// A negative output dim is `infer_shape`'s "the unknown input axis,
/// unchanged", so it matches; otherwise a dim must be known and equal. Any
/// unproven dimension keeps the op.
fn shape_preserved(out: &[Option<i64>], entering: &[Option<i64>]) -> bool {
    out.len() == entering.len()
        && out.iter().zip(entering).all(|(o, e)| match o {
            Some(o) if *o < 0 => true,
            Some(o) => Some(*o) == *e,
            None => false,
        })
}

/// Each crop hoisted to the front of the pointwise run before it.
///
/// The run stops at the first op the window may not cross (its spatial rule
/// is not `Pointwise`, or it reads another node's buffer, which the window
/// would leave full-size). A crop whose run holds an assertion boundary stays
/// put: moving it would change a shape the user pinned. Two crops never
/// contend — a crop is itself not `Pointwise` — so, left to right, a crop's
/// run is always the tail of the order built so far.
fn hoist_spatial_windows(node: &Node<'_>) -> Result<Vec<usize>, String> {
    let steps = node
        .ops
        .iter()
        .map(|op| crate::resolve_op_from_json(op))
        .collect::<Result<Vec<GraphStep>, String>>()?;
    let crossable = |step: &GraphStep| {
        !step.reads_other_nodes() && step.spatial_dependency() == SpatialDependency::Pointwise
    };
    let mut order: Vec<usize> = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        if !step.is_spatial_window() {
            order.push(i);
            continue;
        }
        let mut j = i;
        while j > 0 && crossable(&steps[j - 1]) {
            j -= 1;
        }
        if node.assertions.iter().any(|b| (j + 1..=i).contains(b)) {
            order.push(i);
            continue;
        }
        order.insert(j, i);
    }
    Ok(order)
}

/// Python entry point: run the node-scope pass `pass_name`.
///
/// `states` are the planner's `PlanState` at each boundary (`len(ops) + 1`).
/// Returns the new order, or `None` when the pass changes nothing.
#[pyfunction]
pub(crate) fn node_pass(
    pass_name: &str,
    ops: Vec<String>,
    states: Vec<State>,
    assertions: Vec<usize>,
) -> PyResult<Option<Vec<usize>>> {
    let pass = view_buffer::naming::lookup(LogicalPass::NAMED, pass_name)
        .ok_or_else(|| py_value_error(format!("unknown pass {pass_name:?}")))?;
    run(
        pass,
        &Node {
            ops: &ops,
            states: &states,
            assertions: &assertions,
        },
    )
    .map_err(py_value_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn image(h: i64, w: i64) -> State {
        State {
            dims: [Some(h), Some(w), Some(3)],
            ..State::new("buffer", "u8", Some(3))
        }
    }

    fn ops(values: &[serde_json::Value]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    fn node<'a>(ops: &'a [String], states: &'a [State], assertions: &'a [usize]) -> Node<'a> {
        Node {
            ops,
            states,
            assertions,
        }
    }

    const CROP: fn() -> serde_json::Value =
        || json!({"op": "crop", "top": 1, "left": 1, "height": 2, "width": 2});

    #[test]
    fn a_zero_pad_and_a_full_frame_crop_are_eliminated() {
        let o = ops(&[
            json!({"op": "pad", "top": 0, "bottom": 0, "left": 0, "right": 0, "value": 0, "mode": "constant"}),
            json!({"op": "crop", "top": 0, "left": 0, "height": 4, "width": 5}),
            json!({"op": "grayscale"}),
        ]);
        let s = [image(4, 5), image(4, 5), image(4, 5), image(4, 5)];
        let order = run(LogicalPass::IdentityElimination, &node(&o, &s, &[])).unwrap();
        assert_eq!(order, Some(vec![2]));
    }

    #[test]
    fn a_partial_crop_or_an_asserting_node_is_kept() {
        let o = ops(&[CROP()]);
        let s = [image(4, 5), image(2, 2)];
        assert_eq!(
            run(LogicalPass::IdentityElimination, &node(&o, &s, &[])).unwrap(),
            None
        );
        let o = ops(&[
            json!({"op": "pad", "top": 0, "bottom": 0, "left": 0, "right": 0, "value": 0, "mode": "constant"}),
        ]);
        let s = [image(4, 5), image(4, 5)];
        assert_eq!(
            run(LogicalPass::IdentityElimination, &node(&o, &s, &[1])).unwrap(),
            None
        );
    }

    #[test]
    fn a_crop_hoists_past_pointwise_ops_but_not_a_barrier() {
        let o = ops(&[
            json!({"op": "blur", "sigma": 1.0}),
            json!({"op": "grayscale"}),
            json!({"op": "invert"}),
            CROP(),
        ]);
        let s = vec![image(4, 5); 5];
        let order = run(LogicalPass::SpatialWindowPushdown, &node(&o, &s, &[])).unwrap();
        // blur is a neighbourhood op: the crop stops after it.
        assert_eq!(order, Some(vec![0, 3, 1, 2]));
    }

    #[test]
    fn a_crop_does_not_cross_an_op_reading_another_node_or_an_assertion() {
        let o = ops(&[json!({"op": "add", "other": "n0"}), CROP()]);
        let s = vec![image(4, 5); 3];
        assert_eq!(
            run(LogicalPass::SpatialWindowPushdown, &node(&o, &s, &[])).unwrap(),
            None
        );
        let o = ops(&[json!({"op": "invert"}), CROP()]);
        assert_eq!(
            run(LogicalPass::SpatialWindowPushdown, &node(&o, &s, &[1])).unwrap(),
            None
        );
    }

    /// The committed pass catalogue is what `scripts/gen_ops.py` generates
    /// `OptFlags` from. Regenerate with `POLARS_CV_BLESS=1 cargo test -p
    /// polars-cv catalog_matches`.
    #[test]
    fn pass_catalog_matches_the_committed_file() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/pass_catalog.json"
        );
        let current = pass_catalog_json();
        if std::env::var_os("POLARS_CV_BLESS").is_some() {
            std::fs::write(path, &current).unwrap();
        }
        let committed = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            committed == current,
            "tests/golden/pass_catalog.json is stale; regenerate with \
             POLARS_CV_BLESS=1 cargo test -p polars-cv catalog_matches, then \
             python scripts/gen_ops.py"
        );
    }

    #[test]
    fn a_graph_pass_is_refused_on_a_node() {
        let err = run(
            LogicalPass::CommonSubexpressionElimination,
            &node(&[], &[image(1, 1)], &[]),
        )
        .unwrap_err();
        assert!(err.contains("rewrites the graph"), "{err}");
    }

    /// A verdict resting on a parameter's literal value is not provable when
    /// that parameter is per-row; a per-row parameter the verdict does not
    /// read (a pad's fill behind zero amounts) leaves it provable.
    #[test]
    fn only_a_per_row_deciding_param_keeps_an_identity() {
        let pad = |top: serde_json::Value, value: serde_json::Value| json!({"op": "pad", "top": top, "bottom": 0, "left": 0, "right": 0, "value": value, "mode": "constant"});
        let s = [image(4, 5), image(4, 5)];
        let eliminated = |op: serde_json::Value| {
            let o = ops(&[op]);
            run(LogicalPass::IdentityElimination, &node(&o, &s, &[])).unwrap() == Some(vec![])
        };
        assert!(!eliminated(pad(json!({"$slot": 0}), json!(0.0))));
        assert!(eliminated(pad(json!(0), json!({"$slot": 0}))));
        // A crop is a candidate only at a literal (0, 0) origin.
        let crop = |top: serde_json::Value, left: serde_json::Value| json!({"op": "crop", "top": top, "left": left, "height": 4, "width": 5});
        assert!(eliminated(crop(json!(0), json!(0))));
        assert!(!eliminated(crop(json!(1), json!(0))));
        assert!(!eliminated(crop(json!(0), json!(1))));
        assert!(!eliminated(crop(json!({"$slot": 0}), json!(0))));
        assert!(!eliminated(crop(json!(0), json!({"$slot": 0}))));
    }

    /// The spatial rule of one representative op per class, through the typed
    /// op's resolution (what the pushdown reads).
    #[test]
    fn representative_ops_have_their_true_spatial_rule() {
        let rule = |op: serde_json::Value| -> String {
            match crate::resolve_op_from_json(&op.to_string())
                .unwrap()
                .spatial_dependency()
            {
                SpatialDependency::Pointwise => "pointwise".into(),
                SpatialDependency::Neighborhood(support) => {
                    format!("neighborhood:{}", support.radius)
                }
                SpatialDependency::Global => "global".into(),
                SpatialDependency::Geometric(_) => "geometric".into(),
            }
        };
        let cases = [
            (json!({"op": "cast", "dtype": "f32"}), "pointwise"),
            (json!({"op": "invert"}), "pointwise"),
            (json!({"op": "grayscale"}), "pointwise"),
            (json!({"op": "threshold", "value": 128.0}), "pointwise"),
            (
                json!({"op": "convolve2d", "kernel": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "ksize": 3, "normalize": false, "border": "replicate"}),
                "neighborhood:1",
            ),
            // ceil(3 * sigma)
            (json!({"op": "blur", "sigma": 1.0}), "neighborhood:3"),
            (json!({"op": "blur", "sigma": 2.0}), "neighborhood:6"),
            (json!({"op": "adjust_contrast", "factor": 1.5}), "global"),
            (json!({"op": "equalize_histogram"}), "global"),
            // Hysteresis links edges non-locally.
            (
                json!({"op": "canny", "low_threshold": 50.0, "high_threshold": 150.0}),
                "global",
            ),
            (
                json!({"op": "perceptual_hash", "algorithm": "perceptual", "hash_size": 64}),
                "global",
            ),
            (json!({"op": "reduce_sum"}), "global"),
            (
                json!({"op": "resize", "height": 8, "width": 8, "filter": "bilinear"}),
                "geometric",
            ),
            (
                json!({"op": "rotate", "angle": 30.0, "expand": false, "interpolation": "nearest", "border_value": 0.0}),
                "geometric",
            ),
            (
                json!({"op": "pad", "top": 1, "bottom": 0, "left": 0, "right": 0, "value": 0.0, "mode": "constant"}),
                "geometric",
            ),
            (CROP(), "geometric"),
            (json!({"op": "flip", "axes": [0]}), "geometric"),
            (json!({"op": "transpose", "axes": [1, 0, 2]}), "geometric"),
        ];
        for (op, expected) in cases {
            assert_eq!(rule(op.clone()), expected, "{op}");
        }
    }
}
