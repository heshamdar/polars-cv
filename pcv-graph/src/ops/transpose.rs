//! `transpose` op — permute the buffer's axes.

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput, ViewDto, ViewOp};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct TransposeOp {
    axes: Vec<usize>,
}

impl Operation for TransposeOp {
    fn name(&self) -> &'static str {
        "transpose"
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
            "transpose",
            inputs,
            ViewDto::View(ViewOp::Transpose(self.axes.clone())),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "transpose",
    doc: "Permute the buffer's axes according to `axes`. Shape and strides \
          update; data is not copied unless a downstream op requires \
          contiguous layout.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let axes = crate::params::require_usize_array(params, "transpose", "axes")?;
    Ok(Arc::new(TransposeOp { axes }))
}

inventory::submit! {
    OpRegistration {
        name: "transpose",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
