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
//! Limits: the exact table ([`KNOWN`]) is u8 only (every dtype is held to its
//! op's declared memory effect instead), one base size, and the samples' own
//! parameters. An op whose copies depend on a parameter the sample does not
//! exercise is not seen through that parameter.

use polars::prelude::*;
use view_buffer::ops::MemoryEffect;
use view_buffer::{DType, ViewBuffer};

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

/// An op that reads packed rows (`RequiresDenseRows`, or `PacksOwnRows` for
/// the resizes' own row readers): a crop or a vertical flip is read in place,
/// a transpose or a horizontal flip, which has no packed rows, is not.
const PACKED_ROWS_ONLY: &str =
    "reads packed rows only; a transpose or horizontal flip has none and is packed";

/// A binary op reads each operand's packed rows in place and packs an operand
/// without them (`zip_with`). The census reads one view as both operands, so
/// a transpose or horizontal flip is packed twice.
const BOTH_PACKED: &str =
    "packs each operand without packed rows; walking one element by element measured ~4x slower";

/// View-caused copies an op is known to pay, per [`COUNTED`] layout, and why.
///
/// Each is a debt or a limit of what reads the input, not a target. Removing
/// a copy means changing its row here, and the census fails until you do.
const KNOWN: &[(&str, [usize; 4], &str)] = &[
    ("add", [0, 0, 2, 2], BOTH_PACKED),
    (
        "apply_mask",
        [0, 0, 1, 1],
        "packs an image without packed rows; the inverted mask is written packed already",
    ),
    ("bitwise_and", [0, 0, 2, 2], BOTH_PACKED),
    ("bitwise_or", [0, 0, 2, 2], BOTH_PACKED),
    ("bitwise_xor", [0, 0, 2, 2], BOTH_PACKED),
    ("blend", [0, 0, 2, 2], BOTH_PACKED),
    ("divide", [0, 0, 2, 2], BOTH_PACKED),
    ("maximum", [0, 0, 2, 2], BOTH_PACKED),
    ("minimum", [0, 0, 2, 2], BOTH_PACKED),
    ("multiply", [0, 0, 2, 2], BOTH_PACKED),
    ("subtract", [0, 0, 2, 2], BOTH_PACKED),
    ("blur", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    (
        "canny",
        [1, 1, 1, 2],
        "declares RequiresContiguous, and packs a horizontally flipped input once more",
    ),
    ("channel_swap", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("dilate", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("equalize_histogram", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("erode", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("extract_contours", [1, 1, 1, 1], PACKED_FIRST),
    ("histogram", [1, 1, 1, 1], PACKED_FIRST),
    ("morphology_gradient", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("pad", [0, 0, 1, 1], PACKED_ROWS_ONLY),
    ("pad_to_size", [0, 0, 1, 1], PACKED_ROWS_ONLY),
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

/// A `dtype` blob of `shape` with a recognisable pattern.
fn blob(shape: [usize; 3], dtype: DType) -> Series {
    let n = shape.iter().product::<usize>();
    let data: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
    let blob = ViewBuffer::from_vec_with_shape(data, shape.to_vec())
        .cast(dtype)
        .to_blob();
    Series::new("b".into(), std::slice::from_ref(&blob))
}

/// The graph running `op` after the `prefix` view ops, into `sink`. A binary
/// op (one reading node `n0`) reads the viewed node as both operands.
fn graph(op: &str, prefix: &str, sink: &str, dtype: DType) -> String {
    let source = format!(r#"{{"format": "blob", "dtype": "{}"}}"#, dtype.short_name());
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
fn runnable(op: &str, dtype: DType) -> Option<(usize, &'static str)> {
    for c in [3, 1] {
        for sink in [r#"{"format": "numpy"}"#, r#"{"format": "native"}"#] {
            let shape = [H, W, c];
            let threshold = shape.iter().product::<usize>() * dtype.size_of();
            if copies(&graph(op, "", sink, dtype), &blob(shape, dtype), threshold).is_ok() {
                return Some((c, sink));
            }
        }
    }
    None
}

/// Every op's view-caused copies per layout on `dtype` input, and the ops that
/// ran (an op refusing `dtype` does not).
fn census(dtype: DType) -> (Vec<(String, &'static str, usize)>, Vec<String>) {
    let mut measured = Vec::new();
    let mut ran = Vec::new();
    for sample in TypedOp::samples() {
        let name = sample.name().to_string();
        let op = serde_json::to_string(&sample).unwrap();
        let Some((c, sink)) = runnable(&op, dtype) else {
            continue;
        };
        ran.push(name.clone());
        let base = blob([H, W, c], dtype);
        for (layout, prefix, shape) in layouts(c) {
            let threshold = shape.iter().product::<usize>() * dtype.size_of();
            let viewed = copies(&graph(&op, &prefix, sink, dtype), &base, threshold)
                .unwrap_or_else(|e| panic!("{name} on a {layout} view: {e}"));
            let dense = copies(&graph(&op, "", sink, dtype), &blob(shape, dtype), threshold)
                .unwrap_or_else(|e| panic!("{name} on a contiguous {shape:?}: {e}"));
            measured.push((name.clone(), layout, viewed.saturating_sub(dense)));
        }
    }
    (measured, ran)
}

#[test]
fn a_view_costs_each_op_only_its_known_copies() {
    let (measured, ran) = census(DType::U8);
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

/// Ops that declare no [`MemoryEffect`] (graph ops other than the binary
/// ones, which run outside the engine's planner), so only [`KNOWN`] holds
/// their copies, and on u8 only. Listing one is checked both ways, like
/// [`EXEMPT`].
const UNDECLARED: &[(&str, &str)] = &[
    (
        "apply_mask",
        "a graph op run by the executor itself; KNOWN holds its u8 copies",
    ),
    (
        "extract_shape",
        "a graph op reading only the shape; KNOWN would list any copy",
    ),
];

/// Whether an op declaring `effect` may pay a view-caused copy on `layout`,
/// and whether it must: nothing on a layout it reads in place, at least one
/// on a layout it packs. `(may, must)`.
fn bounds(effect: MemoryEffect, layout: &str) -> (bool, bool) {
    let dense_rows = matches!(layout, "crop" | "vflip");
    match effect {
        MemoryEffect::View | MemoryEffect::StridePreserving => (false, false),
        MemoryEffect::RequiresDenseRows | MemoryEffect::PacksOwnRows => (!dense_rows, !dense_rows),
        MemoryEffect::RequiresContiguous | MemoryEffect::ViewOfContiguous => (true, true),
    }
}

/// An op's `memory_effect` is its account of which input layouts cost it a
/// copy. Held on every dtype an op accepts, since a kernel's paths split by
/// dtype (resize takes fast_image_resize for u8/u16/f32, a cast for i8/i16,
/// an f64 resampler for the rest):
///
/// - on every dtype, an op pays no copy on a layout its declaration reads in
///   place (`StridePreserving` on a kernel that packs a crop fails);
/// - on u8, it also pays one where its declaration packs (`StridePreserving`
///   on a kernel that packs a transpose fails). Only on u8, because a dtype
///   path may read a layout in place that the declaration allows it to pack.
///
/// Who makes a copy (`RequiresDenseRows` vs `PacksOwnRows`) is the same count
/// either way, so this cannot tell those two apart.
#[test]
fn each_op_pays_the_copies_its_memory_effect_declares() {
    let samples = TypedOp::samples();
    let mut problems = Vec::new();
    let mut undeclared = Vec::new();
    for &dtype in DType::ALL {
        let (measured, ran) = census(dtype);
        let mut compared = 0;
        for sample in &samples {
            let name = sample.name();
            if !ran.iter().any(|r| r == name) {
                continue;
            }
            let Some(effect) = sample.declared_memory_effect() else {
                if !undeclared.contains(&name) {
                    undeclared.push(name);
                }
                continue;
            };
            for (op, layout, extra) in &measured {
                if op != name || !COUNTED.contains(layout) {
                    continue;
                }
                compared += 1;
                let (may, must) = bounds(effect, layout);
                let paid = *extra > 0;
                if (paid && !may) || (dtype == DType::U8 && must && !paid) {
                    problems.push(format!(
                        "{name} declares {effect:?} but pays {extra} copies on a {} {layout} view",
                        dtype.short_name()
                    ));
                }
            }
        }
        // Every dtype some ops accept: a census that compared nothing is a
        // census that stopped measuring, not a pass.
        assert!(
            compared >= 4 * 10,
            "{} compared only {compared} (op, layout) pairs",
            dtype.short_name()
        );
    }
    for name in &undeclared {
        if !UNDECLARED.iter().any(|(u, _)| u == name) {
            problems.push(format!(
                "{name} declares no memory effect and is not UNDECLARED"
            ));
        }
    }
    for (name, why) in UNDECLARED {
        assert!(why.len() > 20, "{name}: an UNDECLARED row needs its reason");
        if !undeclared.contains(name) {
            problems.push(format!(
                "UNDECLARED lists {name}, which declares an effect or never ran"
            ));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}
