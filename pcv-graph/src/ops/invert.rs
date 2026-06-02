//! `invert` op — `x' = max(dtype) - x`.

use std::sync::Arc;

use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct InvertOp;

impl Operation for InvertOp {
    fn name(&self) -> &'static str { "invert" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto("invert", inputs, ViewDto::Compute(ComputeOp::Invert))
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "invert",
    doc: "Photonegative: `x' = max(dtype) - x`. For floats, `1.0 - x`.",
    params: &[],
};

fn factory(_params: &ParamMap) -> Result<OpHandle, OpError> {
    Ok(Arc::new(InvertOp))
}

inventory::submit! {
    OpRegistration {
        name: "invert",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
