//! Identity op — passes its input through unchanged.
//!
//! Lives in the registry so the inventory machinery and the `dump_schema`
//! codegen are both exercised by something concrete from the moment they
//! land. Real ops (resize, grayscale, …) follow the same pattern.

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct IdentityOp;

impl Operation for IdentityOp {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn input_arity(&self) -> InputArity {
        InputArity::Unary
    }

    fn input_domain(&self, _port: &str) -> Domain {
        Domain::Any
    }

    fn output_domain(&self) -> Domain {
        Domain::Any
    }

    fn contract(&self) -> &'static OpContract {
        &CONTRACT
    }

    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        let input = inputs.require_single()?;
        Ok(input.clone())
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "identity",
    doc: "Pass the input through unchanged. Useful as a placeholder or for \
          materializing a node into a sink without any transformation.",
    params: &[],
};

fn factory(_params: &ParamMap) -> Result<OpHandle, OpError> {
    Ok(Arc::new(IdentityOp))
}

inventory::submit! {
    OpRegistration {
        name: "identity",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
