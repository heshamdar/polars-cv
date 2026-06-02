//! `blur` op — Gaussian blur with `sigma`.

use std::sync::Arc;

use view_buffer::ops::{Domain, ImageOp, ImageOpKind, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct BlurOp { sigma: f32 }

impl Operation for BlurOp {
    fn name(&self) -> &'static str { "blur" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "blur",
            inputs,
            ViewDto::Image(ImageOp { kind: ImageOpKind::Blur { sigma: self.sigma } }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::FixedU8,
    NdimEffect::Preserve,
    AlphaMode::StripProcessRestore,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "blur",
    doc: "Gaussian blur with the given `sigma`. Alpha is stripped, blur applied, alpha restored.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let sigma = require_f64(params, "blur", "sigma")? as f32;
    Ok(Arc::new(BlurOp { sigma }))
}

inventory::submit! {
    OpRegistration {
        name: "blur",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
