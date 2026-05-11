//! `canny` op — fused edge detection.

use std::sync::Arc;

use view_buffer::ops::{Domain, ImageOp, ImageOpKind, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct CannyOp { low: f32, high: f32 }

impl Operation for CannyOp {
    fn name(&self) -> &'static str { "canny" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "canny",
            inputs,
            ViewDto::Image(ImageOp {
                kind: ImageOpKind::Canny { low_threshold: self.low, high_threshold: self.high },
            }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::FixedU8,
    NdimEffect::Preserve,
    AlphaMode::Drop,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "canny",
    doc: "Canny edge detection: Gaussian blur + Sobel + non-max suppression + hysteresis. \
          Output is a single-channel u8 binary edge map.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let low = require_f64(params, "canny", "low_threshold")? as f32;
    let high = require_f64(params, "canny", "high_threshold")? as f32;
    Ok(Arc::new(CannyOp { low, high }))
}

inventory::submit! {
    OpRegistration {
        name: "canny",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
