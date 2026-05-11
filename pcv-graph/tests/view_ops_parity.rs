//! Byte-parity for the view-ops batch (step 5a).
//!
//! Each test builds the same `ViewDto` two ways:
//!  - directly through view-buffer's `ViewExpr` (mirrors v1's flow).
//!  - through the pcv-graph registry: `find_op(name).factory(params).execute(...)`.
//!
//! Then asserts the materialized buffers are bytewise identical. This is the
//! strongest possible parity proof short of running both paths through
//! Polars itself — the v2 path adds zero behavior, it just changes
//! dispatch.

use pcv_graph::op::{ExecCtx, OpInputs};
use pcv_graph::params::ParamMap;
use pcv_graph::registry::find_op;
use serde_json::json;
use view_buffer::expr::ViewExpr;
use view_buffer::ops::{Domain, NodeOutput, ViewDto, ViewOp};
use view_buffer::ViewBuffer;

fn make_rgb_2x2() -> ViewBuffer {
    // 2x2 RGB image, [H, W, C] = [2, 2, 3], 12 bytes
    let data: Vec<u8> = vec![
        10, 20, 30, 40, 50, 60, // row 0
        70, 80, 90, 100, 110, 120, // row 1
    ];
    ViewBuffer::from_vec_with_shape(data, vec![2, 2, 3])
}

fn v1_apply(input: ViewBuffer, dto: ViewDto) -> ViewBuffer {
    ViewExpr::new_source(input).apply_op(dto).plan().execute()
}

fn v2_apply(op_name: &str, params: ParamMap, input: ViewBuffer) -> ViewBuffer {
    let reg = find_op(op_name).unwrap_or_else(|| panic!("op `{op_name}` not registered"));
    let op = (reg.factory)(&params).expect("factory");
    let input_node = NodeOutput::from_buffer(input);
    let ctx = ExecCtx::new(0);
    let out = op
        .execute(&ctx, &OpInputs::single(&input_node))
        .expect("execute");
    assert_eq!(out.domain(), Domain::Buffer);
    let arc = out.as_buffer().unwrap();
    (**arc).clone()
}

fn buffers_eq_bytes(a: &ViewBuffer, b: &ViewBuffer) -> bool {
    // View ops produce strided buffers; comparing raw bytes against the
    // original storage doesn't tell us anything semantic. Materialize both
    // to contiguous so we compare the actual element sequence.
    let ac = a.to_contiguous();
    let bc = b.to_contiguous();
    let (ptr_a, shape_a, _, dt_a) = ac.as_raw_parts();
    let (ptr_b, shape_b, _, dt_b) = bc.as_raw_parts();
    if shape_a != shape_b || dt_a != dt_b {
        eprintln!(
            "shape/dtype mismatch: v1={shape_a:?} {dt_a:?}, v2={shape_b:?} {dt_b:?}"
        );
        return false;
    }
    let total: usize = shape_a.iter().product::<usize>() * dt_a.size_of();
    let sa = unsafe { std::slice::from_raw_parts(ptr_a, total) };
    let sb = unsafe { std::slice::from_raw_parts(ptr_b, total) };
    sa == sb
}

#[test]
fn transpose_byte_parity() {
    let v1 = v1_apply(
        make_rgb_2x2(),
        ViewDto::View(ViewOp::Transpose(vec![1, 0, 2])),
    );
    let mut params = ParamMap::new();
    params.insert("axes".into(), json!([1, 0, 2]));
    let v2 = v2_apply("transpose", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn reshape_byte_parity() {
    let v1 = v1_apply(
        make_rgb_2x2(),
        ViewDto::View(ViewOp::Reshape(vec![4, 3])),
    );
    let mut params = ParamMap::new();
    params.insert("shape".into(), json!([4, 3]));
    let v2 = v2_apply("reshape", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn flip_byte_parity() {
    let v1 = v1_apply(make_rgb_2x2(), ViewDto::View(ViewOp::Flip(vec![0])));
    let mut params = ParamMap::new();
    params.insert("axes".into(), json!([0]));
    let v2 = v2_apply("flip", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn crop_byte_parity() {
    // Crop semantics in v1's resolve_op: when either height OR width is
    // omitted, BOTH dims extend to the buffer's end. So we have to specify
    // both height and width on the v2 side to match an end of
    // `[top+h, left+w, usize::MAX]` on the v1 side — see comments at
    // polars-cv/src/execute.rs:394.
    let v1 = v1_apply(
        make_rgb_2x2(),
        ViewDto::View(ViewOp::Crop {
            start: vec![0, 0, 0],
            end: vec![1, 2, usize::MAX],
        }),
    );
    let mut params = ParamMap::new();
    params.insert("top".into(), json!(0));
    params.insert("left".into(), json!(0));
    params.insert("height".into(), json!(1));
    params.insert("width".into(), json!(2));
    let v2 = v2_apply("crop", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn channel_select_byte_parity() {
    let v1 = v1_apply(
        make_rgb_2x2(),
        ViewDto::View(ViewOp::ChannelSelect { index: 1 }),
    );
    let mut params = ParamMap::new();
    params.insert("index".into(), json!(1));
    let v2 = v2_apply("channel_select", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn rotate_90_byte_parity() {
    let v1 = v1_apply(make_rgb_2x2(), ViewDto::View(ViewOp::Rotate90));
    let mut params = ParamMap::new();
    params.insert("degrees".into(), json!(90));
    let v2 = v2_apply("rotate", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn rotate_180_byte_parity() {
    let v1 = v1_apply(make_rgb_2x2(), ViewDto::View(ViewOp::Rotate180));
    let mut params = ParamMap::new();
    params.insert("degrees".into(), json!(180));
    let v2 = v2_apply("rotate", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn rotate_270_byte_parity() {
    let v1 = v1_apply(make_rgb_2x2(), ViewDto::View(ViewOp::Rotate270));
    let mut params = ParamMap::new();
    params.insert("degrees".into(), json!(270));
    let v2 = v2_apply("rotate", params, make_rgb_2x2());
    assert!(buffers_eq_bytes(&v1, &v2));
}

#[test]
fn all_view_ops_registered() {
    for name in [
        "transpose",
        "reshape",
        "flip",
        "crop",
        "channel_select",
        "rotate",
    ] {
        assert!(find_op(name).is_some(), "{name} should be registered");
    }
}
