//! `adjust_contrast` op — multiplicative contrast adjustment around the mid-point.

use std::sync::Arc;

use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct AdjustContrastOp { factor: f32 }

impl Operation for AdjustContrastOp {
    fn name(&self) -> &'static str { "adjust_contrast" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "adjust_contrast",
            inputs,
            ViewDto::Compute(ComputeOp::AdjustContrast(self.factor)),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::PromoteToFloat,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "adjust_contrast",
    doc: "Multiplicative contrast adjustment around the mid-point. `factor` > 1 \
          increases contrast; < 1 decreases.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let factor = require_f64(params, "adjust_contrast", "factor")? as f32;
    Ok(Arc::new(AdjustContrastOp { factor }))
}

inventory::submit! {
    OpRegistration {
        name: "adjust_contrast",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
