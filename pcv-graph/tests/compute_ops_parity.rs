//! Byte-parity for the compute-ops batch (step 5b).
//!
//! Same pattern as `view_ops_parity.rs`: build the same ViewDto two ways
//! (v1: direct ViewExpr; v2: registry factory + Operation::execute) and
//! assert the materialized buffers are bytewise identical after
//! `.to_contiguous()`.

use pcv_graph::op::{ExecCtx, OpInputs};
use pcv_graph::params::ParamMap;
use pcv_graph::registry::find_op;
use serde_json::json;
use view_buffer::expr::ViewExpr;
use view_buffer::ops::compute::{ComputeOp, NormalizeMethod};
use view_buffer::ops::{Domain, NodeOutput, ViewDto};
use view_buffer::ViewBuffer;

fn make_u8_3x3() -> ViewBuffer {
    let data: Vec<u8> = (0u8..9u8).collect();
    ViewBuffer::from_vec_with_shape(data, vec![3, 3])
}

fn make_f32_2x2() -> ViewBuffer {
    let data: Vec<f32> = vec![-2.0, -0.5, 0.5, 2.0];
    ViewBuffer::from_vec_with_shape(data, vec![2, 2])
}

fn v1_apply(input: ViewBuffer, dto: ViewDto) -> ViewBuffer {
    ViewExpr::new_source(input).apply_op(dto).plan().execute()
}

fn v2_apply(op_name: &str, params: ParamMap, input: ViewBuffer) -> ViewBuffer {
    let reg = find_op(op_name).unwrap();
    let op = (reg.factory)(&params).expect("factory");
    let input_node = NodeOutput::from_buffer(input);
    let ctx = ExecCtx::new(0);
    let out = op
        .execute(&ctx, &OpInputs::single(&input_node))
        .expect("execute");
    assert_eq!(out.domain(), Domain::Buffer);
    (**out.as_buffer().unwrap()).clone()
}

fn buffers_eq(a: &ViewBuffer, b: &ViewBuffer) -> bool {
    let ac = a.to_contiguous();
    let bc = b.to_contiguous();
    let (pa, sa, _, da) = ac.as_raw_parts();
    let (pb, sb, _, db) = bc.as_raw_parts();
    if sa != sb || da != db {
        return false;
    }
    let total: usize = sa.iter().product::<usize>() * da.size_of();
    let bya = unsafe { std::slice::from_raw_parts(pa, total) };
    let byb = unsafe { std::slice::from_raw_parts(pb, total) };
    bya == byb
}

#[test]
fn cast_byte_parity() {
    let v1 = v1_apply(
        make_u8_3x3(),
        ViewDto::Compute(ComputeOp::Cast(view_buffer::core::dtype::DType::F32)),
    );
    let mut params = ParamMap::new();
    params.insert("dtype".into(), json!("f32"));
    let v2 = v2_apply("cast", params, make_u8_3x3());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn scale_byte_parity() {
    let v1 = v1_apply(make_u8_3x3(), ViewDto::Compute(ComputeOp::Scale(2.5)));
    let mut params = ParamMap::new();
    params.insert("factor".into(), json!(2.5));
    let v2 = v2_apply("scale", params, make_u8_3x3());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn clamp_byte_parity() {
    let v1 = v1_apply(
        make_f32_2x2(),
        ViewDto::Compute(ComputeOp::Clamp { min: -1.0, max: 1.0 }),
    );
    let mut params = ParamMap::new();
    params.insert("min".into(), json!(-1.0));
    params.insert("max".into(), json!(1.0));
    let v2 = v2_apply("clamp", params, make_f32_2x2());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn relu_byte_parity() {
    let v1 = v1_apply(make_f32_2x2(), ViewDto::Compute(ComputeOp::Relu));
    let v2 = v2_apply("relu", ParamMap::new(), make_f32_2x2());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn invert_byte_parity() {
    let v1 = v1_apply(make_u8_3x3(), ViewDto::Compute(ComputeOp::Invert));
    let v2 = v2_apply("invert", ParamMap::new(), make_u8_3x3());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn adjust_contrast_byte_parity() {
    let v1 = v1_apply(
        make_f32_2x2(),
        ViewDto::Compute(ComputeOp::AdjustContrast(1.5)),
    );
    let mut params = ParamMap::new();
    params.insert("factor".into(), json!(1.5));
    let v2 = v2_apply("adjust_contrast", params, make_f32_2x2());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn adjust_gamma_byte_parity() {
    let v1 = v1_apply(
        make_f32_2x2(),
        ViewDto::Compute(ComputeOp::AdjustGamma(2.2)),
    );
    let mut params = ParamMap::new();
    params.insert("gamma".into(), json!(2.2));
    let v2 = v2_apply("adjust_gamma", params, make_f32_2x2());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn normalize_minmax_byte_parity() {
    let v1 = v1_apply(
        make_f32_2x2(),
        ViewDto::Compute(ComputeOp::Normalize(NormalizeMethod::MinMax)),
    );
    let mut params = ParamMap::new();
    params.insert("method".into(), json!("minmax"));
    let v2 = v2_apply("normalize", params, make_f32_2x2());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn normalize_zscore_byte_parity() {
    let v1 = v1_apply(
        make_f32_2x2(),
        ViewDto::Compute(ComputeOp::Normalize(NormalizeMethod::ZScore)),
    );
    let mut params = ParamMap::new();
    params.insert("method".into(), json!("zscore"));
    let v2 = v2_apply("normalize", params, make_f32_2x2());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn normalize_preset_byte_parity() {
    let v1 = v1_apply(
        make_f32_2x2(),
        ViewDto::Compute(ComputeOp::Normalize(NormalizeMethod::Preset {
            mean: vec![0.5],
            std: vec![0.25],
        })),
    );
    let mut params = ParamMap::new();
    params.insert("method".into(), json!("preset"));
    params.insert("mean".into(), json!([0.5]));
    params.insert("std".into(), json!([0.25]));
    let v2 = v2_apply("normalize", params, make_f32_2x2());
    assert!(buffers_eq(&v1, &v2));
}

#[test]
fn all_compute_ops_registered() {
    for name in [
        "cast",
        "scale",
        "clamp",
        "relu",
        "invert",
        "adjust_contrast",
        "adjust_gamma",
        "normalize",
    ] {
        assert!(find_op(name).is_some(), "{name} should be registered");
    }
}
