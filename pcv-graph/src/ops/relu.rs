//! `relu` op — element-wise `max(x, 0)`.

use std::sync::Arc;

use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ReluOp;

impl Operation for ReluOp {
    fn name(&self) -> &'static str { "relu" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto("relu", inputs, ViewDto::Compute(ComputeOp::Relu))
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::PromoteToFloat,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "relu",
    doc: "Element-wise `max(x, 0)`. Integers are promoted to f32.",
    params: &[],
};

fn factory(_params: &ParamMap) -> Result<OpHandle, OpError> {
    Ok(Arc::new(ReluOp))
}

inventory::submit! {
    OpRegistration {
        name: "relu",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
