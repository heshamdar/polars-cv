//! `equalize_histogram` op — per-channel histogram equalization.

use std::sync::Arc;

use view_buffer::ops::{Domain, ImageOp, ImageOpKind, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct EqualizeHistogramOp;

impl Operation for EqualizeHistogramOp {
    fn name(&self) -> &'static str { "equalize_histogram" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "equalize_histogram",
            inputs,
            ViewDto::Image(ImageOp { kind: ImageOpKind::HistogramEqualize }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::FixedU8,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "equalize_histogram",
    doc: "Per-channel histogram equalization. Output is u8 regardless of input dtype.",
    params: &[],
};

fn factory(_params: &ParamMap) -> Result<OpHandle, OpError> {
    Ok(Arc::new(EqualizeHistogramOp))
}

inventory::submit! {
    OpRegistration {
        name: "equalize_histogram",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
