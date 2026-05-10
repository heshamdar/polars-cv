//! Integration tests for the inventory-based op registry.
//!
//! These prove that an op can be discovered by name, instantiated through its
//! factory, and executed through the trait surface — without any of the
//! higher-level IR/executor layers that arrive in later steps.

use std::sync::Arc;

use pcv_graph::contract::{AlphaMode, DTypeEffect, NdimEffect};
use pcv_graph::op::{ExecCtx, OpInputs};
use pcv_graph::params::ParamMap;
use pcv_graph::registry::find_op;
use view_buffer::ops::NodeOutput;
use view_buffer::ViewBuffer;

#[test]
fn identity_is_registered_and_returns_input() {
    let reg = find_op("identity").expect("identity op should be registered");
    assert_eq!(reg.contract.dtype_effect, DTypeEffect::Preserve);
    assert_eq!(reg.contract.ndim_effect, NdimEffect::Preserve);
    assert_eq!(reg.contract.alpha_mode, AlphaMode::Passthrough);

    let op = (reg.factory)(&ParamMap::new()).expect("identity factory should never fail");

    let buf = ViewBuffer::from_vec_with_shape(vec![1u8, 2, 3, 4], vec![2, 2]);
    let input = NodeOutput::from_buffer(buf);
    let ctx = ExecCtx::new(0);
    let out = op
        .execute(&ctx, &OpInputs::single(&input))
        .expect("identity should never fail");

    let buf_out = out.as_buffer().expect("identity should return a buffer");
    let buf_in = input.as_buffer().unwrap();
    // Identity returns its input by clone — the underlying Arc must be the
    // same allocation, proving the no-copy contract.
    assert!(Arc::ptr_eq(buf_out, buf_in));
}

#[test]
fn registry_iteration_includes_identity() {
    let names: Vec<&'static str> = pcv_graph::registry::iter_ops()
        .map(|reg| reg.name)
        .collect();
    assert!(names.contains(&"identity"), "registered ops were: {names:?}");
}
