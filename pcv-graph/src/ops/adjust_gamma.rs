//! `adjust_gamma` op — power-law transformation `x' = x^gamma`.

use std::sync::Arc;

use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct AdjustGammaOp { gamma: f32 }

impl Operation for AdjustGammaOp {
    fn name(&self) -> &'static str { "adjust_gamma" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "adjust_gamma",
            inputs,
            ViewDto::Compute(ComputeOp::AdjustGamma(self.gamma)),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::PromoteToFloat,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "adjust_gamma",
    doc: "Power-law transformation `x' = x^gamma`. Gamma > 1 darkens; < 1 brightens.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let gamma = require_f64(params, "adjust_gamma", "gamma")? as f32;
    Ok(Arc::new(AdjustGammaOp { gamma }))
}

inventory::submit! {
    OpRegistration {
        name: "adjust_gamma",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
