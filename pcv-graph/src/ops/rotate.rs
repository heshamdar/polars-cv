//! `rotate` op — fixed 90/180/270° rotations or general affine.
//!
//! The v1 API takes a single `degrees` parameter. The cardinal angles
//! (90/180/270) map to view-buffer's free `ViewOp::Rotate90/180/270` (which
//! are transpose+flip combos and do not allocate a new buffer if downstream
//! ops can consume non-contiguous data). Arbitrary angles fall through to
//! [`AffineRotateOp`], which builds a `ComputeOp::RotateAffine` and goes
//! through the bilinear sampler. Matches v1 at
//! `polars-cv/src/execute.rs:602-680`.

use std::sync::Arc;

use view_buffer::ops::compute::ComputeOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto, ViewOp};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

enum RotateKind {
    Cardinal(ViewOp),
    Affine { degrees: f64 },
}

pub struct RotateOp(RotateKind);

impl Operation for RotateOp {
    fn name(&self) -> &'static str {
        "rotate"
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
        match &self.0 {
            RotateKind::Cardinal(view_op) => {
                apply_view_dto("rotate", inputs, ViewDto::View(view_op.clone()))
            }
            RotateKind::Affine { degrees } => apply_view_dto(
                "rotate",
                inputs,
                ViewDto::Compute(ComputeOp::RotateAffine {
                    angle_deg: *degrees as f32,
                    expand: false,
                    interpolation: view_buffer::ops::affine::InterpolationType::Bilinear,
                    border_value: 0.0,
                }),
            ),
        }
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "rotate",
    doc: "Rotate the buffer by `degrees`. 90/180/270 use free transpose/flip; \
          arbitrary angles use bilinear sampling with zero-fill.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let degrees = require_f64(params, "rotate", "degrees")?;
    let kind = match degrees.round() as i64 {
        d if d == 0 || (d - 360).abs() == 0 || (d + 360).abs() == 0 => {
            // No rotation — preserve via the cheapest path (identity-equivalent
            // transpose is too noisy; just use a 0-axes flip which is a no-op).
            RotateKind::Cardinal(ViewOp::Flip(Vec::new()))
        }
        d if d == 90 || d == -270 => RotateKind::Cardinal(ViewOp::Rotate90),
        d if d == 180 || d == -180 => RotateKind::Cardinal(ViewOp::Rotate180),
        d if d == 270 || d == -90 => RotateKind::Cardinal(ViewOp::Rotate270),
        _ => RotateKind::Affine { degrees },
    };
    Ok(Arc::new(RotateOp(kind)))
}

inventory::submit! {
    OpRegistration {
        name: "rotate",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
