//! `threshold` op — binarize a single-channel image at `value`.

use std::sync::Arc;

use view_buffer::ops::{Domain, ImageOp, ImageOpKind, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ThresholdOp { value: f64 }

impl Operation for ThresholdOp {
    fn name(&self) -> &'static str { "threshold" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "threshold",
            inputs,
            ViewDto::Image(ImageOp { kind: ImageOpKind::Threshold(self.value) }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::FixedU8,
    NdimEffect::Preserve,
    AlphaMode::Drop,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "threshold",
    doc: "Binarize a single-channel image: output = 255 if input >= value else 0.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let value = require_f64(params, "threshold", "value")?;
    Ok(Arc::new(ThresholdOp { value }))
}

inventory::submit! {
    OpRegistration {
        name: "threshold",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
