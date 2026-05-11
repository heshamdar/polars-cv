//! `scale` op — multiply every element by a constant factor.

use std::sync::Arc;

use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ScaleOp {
    factor: f32,
}

impl Operation for ScaleOp {
    fn name(&self) -> &'static str { "scale" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto("scale", inputs, ViewDto::Compute(ComputeOp::Scale(self.factor)))
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::PromoteToFloat,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "scale",
    doc: "Multiply every element by `factor` (a float). Integers are promoted to f32.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let factor = require_f64(params, "scale", "factor")? as f32;
    Ok(Arc::new(ScaleOp { factor }))
}

inventory::submit! {
    OpRegistration {
        name: "scale",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
