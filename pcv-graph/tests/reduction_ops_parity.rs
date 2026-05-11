//! Byte-parity for the reduction-ops batch (step 5d).
//!
//! Reductions don't go through `ViewExpr` — view-buffer's expr.rs panics for
//! `ViewDto::Reduction`. They go directly through `reduction_op.execute(buf)`
//! and pcv-graph's `common::apply_reduction` mirrors that flow.

use pcv_graph::op::{ExecCtx, OpInputs};
use pcv_graph::params::ParamMap;
use pcv_graph::registry::find_op;
use serde_json::json;
use view_buffer::ops::reduction::ReductionOp;
use view_buffer::ops::NodeOutput;
use view_buffer::ViewBuffer;

fn make_buf() -> ViewBuffer {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ViewBuffer::from_vec_with_shape(data, vec![2, 3])
}

fn v1_run(op: ReductionOp, buf: &ViewBuffer) -> NodeOutput {
    let result = op.execute(buf);
    if result.shape() == [1] {
        use view_buffer::core::dtype::DType;
        let scalar = match result.dtype() {
            DType::F32 => result.as_slice::<f32>()[0] as f64,
            DType::F64 => result.as_slice::<f64>()[0],
            DType::I64 => result.as_slice::<i64>()[0] as f64,
            DType::U64 => result.as_slice::<u64>()[0] as f64,
            _ => panic!("unexpected dtype"),
        };
        NodeOutput::Scalar(scalar)
    } else {
        NodeOutput::from_buffer(result)
    }
}

fn v2_run(name: &str, params: ParamMap, buf: ViewBuffer) -> NodeOutput {
    let reg = find_op(name).unwrap();
    let op = (reg.factory)(&params).expect("factory");
    let input = NodeOutput::from_buffer(buf);
    op.execute(&ExecCtx::new(0), &OpInputs::single(&input))
        .expect("execute")
}

fn scalars_equal(a: &NodeOutput, b: &NodeOutput) -> bool {
    match (a.as_scalar(), b.as_scalar()) {
        (Some(x), Some(y)) => (x - y).abs() < 1e-9 || (x.is_nan() && y.is_nan()),
        _ => false,
    }
}

#[test]
fn reduce_sum_parity() {
    let buf = make_buf();
    let v1 = v1_run(ReductionOp::Sum { axis: None }, &buf);
    let v2 = v2_run("reduce_sum", ParamMap::new(), make_buf());
    assert!(scalars_equal(&v1, &v2), "v1={v1:?} v2={v2:?}");
}

#[test]
fn reduce_mean_parity() {
    let buf = make_buf();
    let v1 = v1_run(ReductionOp::Mean { axis: None }, &buf);
    let v2 = v2_run("reduce_mean", ParamMap::new(), make_buf());
    assert!(scalars_equal(&v1, &v2), "v1={v1:?} v2={v2:?}");
}

#[test]
fn reduce_max_parity() {
    let buf = make_buf();
    let v1 = v1_run(ReductionOp::Max { axis: None }, &buf);
    let v2 = v2_run("reduce_max", ParamMap::new(), make_buf());
    assert!(scalars_equal(&v1, &v2));
}

#[test]
fn reduce_min_parity() {
    let buf = make_buf();
    let v1 = v1_run(ReductionOp::Min { axis: None }, &buf);
    let v2 = v2_run("reduce_min", ParamMap::new(), make_buf());
    assert!(scalars_equal(&v1, &v2));
}

#[test]
fn reduce_std_parity() {
    let buf = make_buf();
    let v1 = v1_run(ReductionOp::Std { axis: None, ddof: 0 }, &buf);
    let v2 = v2_run("reduce_std", ParamMap::new(), make_buf());
    assert!(scalars_equal(&v1, &v2));
}

#[test]
fn reduce_popcount_parity() {
    let data: Vec<u8> = vec![0b1010_1010, 0b0101_0101, 0b1111_0000];
    let buf = ViewBuffer::from_vec_with_shape(data.clone(), vec![3]);
    let v1 = v1_run(ReductionOp::PopCount, &buf);
    let v2_buf = ViewBuffer::from_vec_with_shape(data, vec![3]);
    let v2 = v2_run("reduce_popcount", ParamMap::new(), v2_buf);
    assert!(scalars_equal(&v1, &v2));
}

#[test]
fn reduce_percentile_parity() {
    let buf = make_buf();
    let v1 = v1_run(ReductionOp::Percentile { q: 50.0 }, &buf);
    let mut params = ParamMap::new();
    params.insert("q".into(), json!(50.0));
    let v2 = v2_run("reduce_percentile", params, make_buf());
    assert!(scalars_equal(&v1, &v2));
}

#[test]
fn all_reduction_ops_registered() {
    for name in [
        "reduce_sum",
        "reduce_mean",
        "reduce_max",
        "reduce_min",
        "reduce_std",
        "reduce_popcount",
        "reduce_percentile",
        "reduce_argmax",
        "reduce_argmin",
    ] {
        assert!(find_op(name).is_some(), "{name} should be registered");
    }
}
