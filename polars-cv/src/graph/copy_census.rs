//! What a view costs each op: the copies a crop, flip or transpose makes an
//! op pay, counted for every op in the typed catalogue.
//!
//! A view is free to make (`crop`, `flip`, `transpose` move no bytes), but an
//! op that cannot read its layout packs it first, and that copy is invisible
//! in every output. This census makes it visible. Each op runs on a view of a
//! base image and on a contiguous image of the view's exact shape, both from a
//! blob read in place. Every allocation at least the view's size is counted,
//! and the difference between the two runs is the copies the *view* cost. The
//! op's own working buffers appear in both runs and cancel.
//!
//! **Registry-driven, both ways.** The ops are `TypedOp::samples()`, so a new
//! op is measured the day it is added. An op the census cannot run is listed in
//! [`EXEMPT`] with the reason, and an exemption for an op that does run, or that
//! no longer exists, fails. A view-caused copy must be listed in [`KNOWN`] with
//! the reason it is paid. An unlisted one fails, and so does a listed one that
//! no longer happens, so the table cannot go stale in either direction.
//!
//! Limits: one dtype (u8), one base size, and the samples' own parameters. An
//! op whose copies depend on a parameter the sample does not exercise is not
//! seen through that parameter.

use polars::prelude::*;
use view_buffer::ViewBuffer;

use super::compiled::CompiledGraph;
use crate::ops::TypedOp;

/// The base image the views are cut from: `[H, W, C]`.
const H: usize = 64;
const W: usize = 48;

/// The views, by name: the ops that make one from the base, and its shape.
///
/// `crop` and `vflip` keep each row's elements packed (a row stride wider
/// than, or the negative of, a row), `transpose` and `hflip` do not, and
/// `rows` (whole rows from an offset) is contiguous: the control, which no op
/// may pay for.
fn layouts(c: usize) -> [(&'static str, String, [usize; 3]); 5] {
    [
        (
            "crop",
            r#"{"op": "crop", "top": 4, "left": 4, "height": 56, "width": 40}"#.into(),
            [56, 40, c],
        ),
        ("vflip", r#"{"op": "flip", "axes": [0]}"#.into(), [H, W, c]),
        (
            "rows",
            r#"{"op": "crop", "top": 4, "left": 0, "height": 56, "width": 48}"#.into(),
            [56, W, c],
        ),
        (
            "transpose",
            r#"{"op": "transpose", "axes": [1, 0, 2]}"#.into(),
            [W, H, c],
        ),
        ("hflip", r#"{"op": "flip", "axes": [1]}"#.into(), [H, W, c]),
    ]
}

/// The layouts a [`KNOWN`] row counts, in its order. `rows` is not one of
/// them: it is contiguous, and every op must read it for free.
const COUNTED: [&str; 4] = ["crop", "vflip", "transpose", "hflip"];

/// An op that declares `RequiresContiguous`: the planner packs any input that
/// is not contiguous before the kernel runs (`materialize_if_needed`).
const PACKED_FIRST: &str =
    "declares RequiresContiguous, so the planner packs any non-contiguous input";

/// A binary op: the planner packs the left operand, and the kernel packs both
/// operands again (`ops/binary.rs`, `ops/mask.rs`).
const BOTH_OPERANDS: &str =
    "the planner packs the left operand and the kernel packs both operands again";

/// fast_image_resize reads packed rows (`interop/fir.rs`): a crop or a
/// vertical flip is read in place, a transpose or a horizontal flip is not.
const PACKED_ROWS_ONLY: &str =
    "fast_image_resize reads packed rows only; a transpose or horizontal flip has none";

/// View-caused copies an op is known to pay, per [`COUNTED`] layout, and why.
///
/// Each is a debt or a limit of what reads the input, not a target. Removing
/// a copy means changing its row here, and the census fails until you do.
const KNOWN: &[(&str, [usize; 4], &str)] = &[
    ("add", [2, 2, 2, 2], BOTH_OPERANDS),
    ("apply_mask", [2, 2, 2, 2], BOTH_OPERANDS),
    ("bitwise_and", [2, 2, 2, 2], BOTH_OPERANDS),
    ("bitwise_or", [2, 2, 2, 2], BOTH_OPERANDS),
    ("bitwise_xor", [2, 2, 2, 2], BOTH_OPERANDS),
    ("blend", [2, 2, 2, 2], BOTH_OPERANDS),
    ("divide", [2, 2, 2, 2], BOTH_OPERANDS),
    ("maximum", [2, 2, 2, 2], BOTH_OPERANDS),
    ("minimum", [2, 2, 2, 2], BOTH_OPERANDS),
    ("multiply", [2, 2, 2, 2], BOTH_OPERANDS),
    ("subtract", [2, 2, 2, 2], BOTH_OPERANDS),
    ("blur", [1, 1, 1, 1], PACKED_FIRST),
    (
        "canny",
        [1, 1, 1, 2],
        "declares RequiresContiguous, and packs a horizontally flipped input once more",
    ),
    ("channel_swap", [1, 1, 1, 1], PACKED_FIRST),
    ("convolve2d", [1, 1, 1, 1], PACKED_FIRST),
    ("cvt_color", [1, 1, 1, 1], PACKED_FIRST),
    ("dilate", [1, 1, 1, 1], PACKED_FIRST),
    ("equalize_histogram", [1, 1, 1, 1], PACKED_FIRST),
    ("erode", [1, 1, 1, 1], PACKED_FIRST),
    ("extract_contours", [1, 1, 1, 1], PACKED_FIRST),
    ("histogram", [1, 1, 1, 1], PACKED_FIRST),
    ("morphology_gradient", [1, 1, 1, 1], PACKED_FIRST),
    ("pad", [1, 1, 1, 1], PACKED_FIRST),
    ("pad_to_size", [1, 1, 1, 1], PACKED_FIRST),
    ("perceptual_hash", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_argmax", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_argmin", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_max", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_mean", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_min", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_percentile", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_popcount", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_std", [1, 1, 1, 1], PACKED_FIRST),
    ("reduce_sum", [1, 1, 1, 1], PACKED_FIRST),
    ("rotate", [1, 1, 1, 1], PACKED_FIRST),
    ("rotate_and_scale", [1, 1, 1, 1], PACKED_FIRST),
    ("warp_affine", [1, 1, 1, 1], PACKED_FIRST),
    ("letterbox", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("resize", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("resize_max", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("resize_min", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("resize_scale", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("resize_to_height", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("resize_to_width", [0, 0, 1, 1], PACKED_ROWS_ONLY),
];

/// Catalogue ops the census does not run, and why.
const EXEMPT: &[(&str, &str)] = &[
    (
        "assert_shape",
        "its sample asserts a [8, ?, 2] shape no census input has; it moves no data",
    ),
    (
        "reshape",
        "its sample reshapes to 4 elements, which no census input holds",
    ),
    (
        "channel_merge",
        "its sample merges nodes n0 and n1, and the census builds one viewed node",
    ),
    (
        "label_reduce",
        "it reads a second input column (the labels), which the census does not supply",
    ),
    (
        "rasterize",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_area",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_bounding_box",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_centroid",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_convex_hull",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_largest",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_perimeter",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_scale",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_simplify",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
    (
        "contour_translate",
        "it reads a contour, not a buffer: no view of one reaches it",
    ),
];

/// A u8 blob of `shape` with a recognisable pattern.
fn blob(shape: [usize; 3]) -> Series {
    let n = shape.iter().product::<usize>();
    let data: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
    let blob = ViewBuffer::from_vec_with_shape(data, shape.to_vec()).to_blob();
    Series::new("b".into(), std::slice::from_ref(&blob))
}

/// The graph running `op` after the `prefix` view ops, into `sink`. A binary
/// op (one reading node `n0`) reads the viewed node as both operands.
fn graph(op: &str, prefix: &str, sink: &str) -> String {
    let source = r#"{"format": "blob", "dtype": "u8"}"#;
    if op.contains(r#""n0""#) {
        format!(
            r#"{{"nodes": {{"n0": {{"source": {source}, "ops": [{prefix}]}},
                 "n1": {{"source": {{"format": "blob"}}, "upstream": ["n0"], "ops": [{op}]}}}},
               "outputs": {{"_output": {{"node": "n1", "sink": {sink}}}}},
               "column_bindings": {{"n0": 0}}}}"#
        )
    } else {
        let ops = if prefix.is_empty() {
            op.to_string()
        } else {
            format!("{prefix}, {op}")
        };
        format!(
            r#"{{"nodes": {{"n0": {{"source": {source}, "ops": [{ops}]}}}},
               "outputs": {{"_output": {{"node": "n0", "sink": {sink}}}}},
               "column_bindings": {{"n0": 0}}}}"#
        )
    }
}

/// Allocations of at least `threshold` bytes one row of `graph` makes, or
/// why it did not run. The row runs inline, on this thread.
fn copies(graph: &str, input: &Series, threshold: usize) -> Result<usize, String> {
    let compiled = CompiledGraph::compile(graph).map_err(|e| e.to_string())?;
    let (out, count) = crate::test_alloc::large_allocations(threshold, || {
        compiled.execute(std::slice::from_ref(input))
    });
    match out {
        Ok(s) if s.null_count() == 0 => Ok(count),
        Ok(_) => Err("a null row".into()),
        Err(e) => Err(e.to_string()),
    }
}

/// The sink and channel count an op runs with: three channels unless it only
/// takes one, and the numpy sink unless its output is not a buffer. `None` if
/// it runs with none of them.
fn runnable(op: &str) -> Option<(usize, &'static str)> {
    for c in [3, 1] {
        for sink in [r#"{"format": "numpy"}"#, r#"{"format": "native"}"#] {
            let shape = [H, W, c];
            let threshold = shape.iter().product();
            if copies(&graph(op, "", sink), &blob(shape), threshold).is_ok() {
                return Some((c, sink));
            }
        }
    }
    None
}

/// Every runnable op's view-caused copies per layout, and the ops that ran.
fn census() -> (Vec<(String, &'static str, usize)>, Vec<String>) {
    let mut measured = Vec::new();
    let mut ran = Vec::new();
    for sample in TypedOp::samples() {
        let name = sample.name().to_string();
        let op = serde_json::to_string(&sample).unwrap();
        let Some((c, sink)) = runnable(&op) else {
            continue;
        };
        ran.push(name.clone());
        let base = blob([H, W, c]);
        for (layout, prefix, shape) in layouts(c) {
            let threshold = shape.iter().product::<usize>();
            let viewed = copies(&graph(&op, &prefix, sink), &base, threshold)
                .unwrap_or_else(|e| panic!("{name} on a {layout} view: {e}"));
            let dense = copies(&graph(&op, "", sink), &blob(shape), threshold)
                .unwrap_or_else(|e| panic!("{name} on a contiguous {shape:?}: {e}"));
            measured.push((name.clone(), layout, viewed.saturating_sub(dense)));
        }
    }
    (measured, ran)
}

#[test]
fn a_view_costs_each_op_only_its_known_copies() {
    let (measured, ran) = census();
    assert!(
        ran.len() > 40,
        "only {} ops ran; the census is not measuring the catalogue",
        ran.len()
    );

    let mut problems = Vec::new();
    for (op, layout, extra) in &measured {
        let known = match COUNTED.iter().position(|l| l == layout) {
            None => 0,
            Some(i) => KNOWN.iter().find(|(o, _, _)| o == op).map_or(0, |k| k.1[i]),
        };
        if *extra != known {
            problems.push(format!(
                "{op} on a {layout} view: {extra} extra copies, KNOWN says {known}"
            ));
        }
    }
    for (op, copies, why) in KNOWN {
        assert!(why.len() > 20, "{op}: a KNOWN row needs its reason");
        if copies.iter().all(|&c| c == 0) {
            problems.push(format!("KNOWN lists {op} with no copies; drop the row"));
        }
        if !ran.iter().any(|o| o == op) {
            problems.push(format!("KNOWN lists {op}, which the census did not run"));
        }
    }
    let names: Vec<String> = TypedOp::samples()
        .iter()
        .map(|s| s.name().to_string())
        .collect();
    for name in &names {
        let exempt = EXEMPT.iter().any(|(e, _)| e == name);
        match (ran.contains(name), exempt) {
            (false, false) => problems.push(format!("{name} did not run and is not EXEMPT")),
            (true, true) => problems.push(format!("{name} is EXEMPT but runs; measure it")),
            _ => {}
        }
    }
    for (name, why) in EXEMPT {
        assert!(why.len() > 20, "{name}: an exemption needs its reason");
        if !names.iter().any(|n| n == name) {
            problems.push(format!("EXEMPT lists {name}, which is not a catalogue op"));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}
