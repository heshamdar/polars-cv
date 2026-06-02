//! `reduce_argmax` op — index of the maximum along an axis.

use std::sync::Arc;

use view_buffer::ops::reduction::ReductionOp;
use view_buffer::ops::{Domain, NodeOutput};

use crate::contract::{DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_reduction;
use crate::params::{require_i64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ReduceArgmaxOp { axis: usize }

impl Operation for ReduceArgmaxOp {
    fn name(&self) -> &'static str { "reduce_argmax" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Scalar }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_reduction("reduce_argmax", inputs, ReductionOp::ArgMax { axis: self.axis })
    }
}

const CONTRACT: OpContract = OpContract::non_image(DTypeEffect::FixedI64, NdimEffect::ToZero);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "reduce_argmax",
    doc: "Index of the maximum along `axis`. Output dtype is i64.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let axis = require_i64(params, "reduce_argmax", "axis")?.max(0) as usize;
    Ok(Arc::new(ReduceArgmaxOp { axis }))
}

inventory::submit! {
    OpRegistration { name: "reduce_argmax", contract: &CONTRACT, schema: || &SCHEMA, factory }
}
