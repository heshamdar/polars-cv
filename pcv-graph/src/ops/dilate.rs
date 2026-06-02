//! `dilate` op — morphological dilation (local maximum filter).

use std::sync::Arc;

use view_buffer::ops::{Domain, ImageOp, ImageOpKind, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{opt_i64, require_i64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct DilateOp { ksize: u32, iterations: u32 }

impl Operation for DilateOp {
    fn name(&self) -> &'static str { "dilate" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "dilate",
            inputs,
            ViewDto::Image(ImageOp {
                kind: ImageOpKind::Dilate { ksize: self.ksize, iterations: self.iterations },
            }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Drop,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "dilate",
    doc: "Morphological dilation: local maximum over a ksize×ksize neighborhood. \
          Requires single-channel input.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let ksize = require_i64(params, "dilate", "ksize")?.max(1) as u32;
    let iterations = opt_i64(params, "iterations", 1).max(1) as u32;
    Ok(Arc::new(DilateOp { ksize, iterations }))
}

inventory::submit! {
    OpRegistration {
        name: "dilate",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
