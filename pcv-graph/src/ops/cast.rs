//! `cast` op — change the buffer's element dtype.

use std::sync::Arc;

use view_buffer::core::dtype::DType;
use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::{apply_view_dto, parse_dtype};
use crate::params::{require_str, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct CastOp {
    dtype: DType,
}

impl Operation for CastOp {
    fn name(&self) -> &'static str {
        "cast"
    }
    fn input_arity(&self) -> InputArity {
        InputArity::Unary
    }
    fn input_domain(&self, _p: &str) -> Domain {
        Domain::Buffer
    }
    fn output_domain(&self) -> Domain {
        Domain::Buffer
    }
    fn contract(&self) -> &'static OpContract {
        &CONTRACT
    }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto("cast", inputs, ViewDto::Compute(ComputeOp::Cast(self.dtype)))
    }
}

// FixedU8 is a placeholder — the Python contract table at _types.py:482-486
// notes that `cast`'s dtype is param-dependent. Plan-time resolution will
// inspect the `dtype` param; the static contract here is the conservative
// "concrete dtype known once params are resolved" version.
const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::FixedU8,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "cast",
    doc: "Cast the buffer to a different element dtype. Param `dtype` is one \
          of `u8|i8|u16|i16|u32|i32|u64|i64|f32|f64`.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let dtype_str = require_str(params, "cast", "dtype")?;
    let dtype = parse_dtype("cast", dtype_str)?;
    Ok(Arc::new(CastOp { dtype }))
}

inventory::submit! {
    OpRegistration {
        name: "cast",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
