//! `flip` op — reverse the buffer along the given axes.

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput, ViewDto, ViewOp};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct FlipOp {
    axes: Vec<usize>,
}

impl Operation for FlipOp {
    fn name(&self) -> &'static str {
        "flip"
    }
    fn input_arity(&self) -> InputArity {
        InputArity::Unary
    }
    fn input_domain(&self, _port: &str) -> Domain {
        Domain::Buffer
    }
    fn output_domain(&self) -> Domain {
        Domain::Buffer
    }
    fn contract(&self) -> &'static OpContract {
        &CONTRACT
    }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "flip",
            inputs,
            ViewDto::View(ViewOp::Flip(self.axes.clone())),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "flip",
    doc: "Reverse the buffer along the listed axes. Common usage: \
          `flip(axes=[0])` for vertical flip, `flip(axes=[1])` for horizontal.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let axes = crate::params::require_usize_array(params, "flip", "axes")?;
    Ok(Arc::new(FlipOp { axes }))
}

inventory::submit! {
    OpRegistration {
        name: "flip",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
