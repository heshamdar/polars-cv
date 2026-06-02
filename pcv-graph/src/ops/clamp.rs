//! `clamp` op — element-wise min/max.

use std::sync::Arc;

use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ClampOp {
    min: f32,
    max: f32,
}

impl Operation for ClampOp {
    fn name(&self) -> &'static str { "clamp" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "clamp",
            inputs,
            ViewDto::Compute(ComputeOp::Clamp { min: self.min, max: self.max }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::PromoteToFloat,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "clamp",
    doc: "Clamp each element to `[min, max]`. Integers are promoted to f32.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let min = require_f64(params, "clamp", "min")? as f32;
    let max = require_f64(params, "clamp", "max")? as f32;
    Ok(Arc::new(ClampOp { min, max }))
}

inventory::submit! {
    OpRegistration {
        name: "clamp",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
