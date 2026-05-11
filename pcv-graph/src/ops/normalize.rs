//! `normalize` op — `minmax`, `zscore`, or `preset(mean, std)`.

use std::sync::Arc;

use view_buffer::ops::compute::{ComputeOp, NormalizeMethod};
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_str, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct NormalizeOp {
    method: NormalizeMethod,
}

impl Operation for NormalizeOp {
    fn name(&self) -> &'static str { "normalize" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Buffer }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_view_dto(
            "normalize",
            inputs,
            ViewDto::Compute(ComputeOp::Normalize(self.method.clone())),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::ConfigurableF32,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "normalize",
    doc: "Normalize the buffer. `method` is one of: `minmax` (rescale to [0,1]), \
          `zscore` (subtract mean, divide by std), or `preset` (subtract `mean`, \
          divide by `std` — both arrays of floats, one per channel).",
    params: &[],
};

fn parse_float_array(
    params: &ParamMap,
    name: &'static str,
) -> Result<Vec<f32>, OpError> {
    let v = params.get(name).ok_or_else(|| OpError::Failed {
        op: "normalize",
        message: format!("preset requires `{name}`"),
    })?;
    let arr = v.as_array().ok_or_else(|| OpError::Failed {
        op: "normalize",
        message: format!("`{name}` must be a list of floats"),
    })?;
    arr.iter()
        .map(|e| {
            e.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| OpError::Failed {
                    op: "normalize",
                    message: format!("`{name}` entries must be floats"),
                })
        })
        .collect()
}

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let method_str = require_str(params, "normalize", "method")?;
    let method = match method_str {
        "minmax" => NormalizeMethod::MinMax,
        "zscore" => NormalizeMethod::ZScore,
        "preset" => {
            let mean = parse_float_array(params, "mean")?;
            let std = parse_float_array(params, "std")?;
            NormalizeMethod::Preset { mean, std }
        }
        other => {
            return Err(OpError::Failed {
                op: "normalize",
                message: format!("unknown method `{other}`"),
            })
        }
    };
    Ok(Arc::new(NormalizeOp { method }))
}

inventory::submit! {
    OpRegistration {
        name: "normalize",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
