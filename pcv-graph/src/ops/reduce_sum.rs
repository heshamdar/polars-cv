//! `reduce_sum` op — sum, global or along an axis.

use std::sync::Arc;

use view_buffer::ops::reduction::ReductionOp;
use view_buffer::ops::{Domain, NodeOutput};

use crate::contract::{DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_reduction;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ReduceSumOp { axis: Option<usize> }

impl Operation for ReduceSumOp {
    fn name(&self) -> &'static str { "reduce_sum" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Scalar }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_reduction("reduce_sum", inputs, ReductionOp::Sum { axis: self.axis })
    }
}

const CONTRACT: OpContract = OpContract::non_image(DTypeEffect::FixedF64, NdimEffect::ToZero);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "reduce_sum",
    doc: "Sum the buffer. Default: global (scalar output). Pass `axis: int` to \
          reduce along one axis only — output buffer has one fewer dim.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let axis = params.get("axis").and_then(|v| v.as_i64()).map(|i| i.max(0) as usize);
    Ok(Arc::new(ReduceSumOp { axis }))
}

inventory::submit! {
    OpRegistration { name: "reduce_sum", contract: &CONTRACT, schema: || &SCHEMA, factory }
}
