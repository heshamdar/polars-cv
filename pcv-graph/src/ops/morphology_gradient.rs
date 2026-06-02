//! `morphology_gradient` op — dilate minus erode (edge outline).

use std::sync::Arc;

use view_buffer::ops::{Domain, ImageOp, ImageOpKind, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_i64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct MorphologyGradientOp { ksize: u32 }

impl Operation for MorphologyGradientOp {
    fn name(&self) -> &'static str { "morphology_gradient" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "morphology_gradient",
            inputs,
            ViewDto::Image(ImageOp { kind: ImageOpKind::MorphGradient { ksize: self.ksize } }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Drop,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "morphology_gradient",
    doc: "Morphological gradient: `dilate(x) - erode(x)`. Edge outline. \
          Requires single-channel input.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let ksize = require_i64(params, "morphology_gradient", "ksize")?.max(1) as u32;
    Ok(Arc::new(MorphologyGradientOp { ksize }))
}

inventory::submit! {
    OpRegistration {
        name: "morphology_gradient",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
