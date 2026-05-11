//! `reshape` op — reinterpret the buffer with a new shape.

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput, ViewDto, ViewOp};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ReshapeOp {
    shape: Vec<usize>,
}

impl Operation for ReshapeOp {
    fn name(&self) -> &'static str {
        "reshape"
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
            "reshape",
            inputs,
            ViewDto::View(ViewOp::Reshape(self.shape.clone())),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "reshape",
    doc: "Reinterpret the buffer as a new shape. Element count must match.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let shape = crate::params::require_usize_array(params, "reshape", "shape")?;
    Ok(Arc::new(ReshapeOp { shape }))
}

inventory::submit! {
    OpRegistration {
        name: "reshape",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
