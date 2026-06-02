//! `Operation` trait and helper types.
//!
//! Wraps view-buffer's [`Op`](view_buffer::ops::traits::Op) and
//! [`DomainOp`](view_buffer::ops::traits::DomainOp) traits to add the things
//! pcv-graph cares about that pure tensor ops don't:
//!
//! - **Named multi-input ports** (e.g. binary ops, `apply_mask`) — view-buffer
//!   ops are single-input by design.
//! - **Per-row expression-bound parameters** — values can come from a Polars
//!   expression rather than a literal.
//! - **Plan-time contract** — dtype/ndim/alpha effects live next to the op
//!   definition, not in a Python table.

use std::sync::Arc;

use thiserror::Error;
use view_buffer::ops::{Domain, NodeOutput};

use crate::contract::OpContract;

/// Number and shape of inputs an operation consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputArity {
    /// Single unnamed input — the common case.
    Unary,
    /// Named ports (e.g. `["lhs", "rhs"]`, `["image", "mask"]`).
    NamedMulti(&'static [&'static str]),
}

/// Resolved inputs handed to [`Operation::execute`].
///
/// For [`InputArity::Unary`] use [`OpInputs::single`]; for
/// [`InputArity::NamedMulti`] use [`OpInputs::named`]. The executor builds
/// these from the IR and the upstream node outputs.
pub enum OpInputs<'a> {
    Single(&'a NodeOutput),
    Named(&'a [(&'static str, &'a NodeOutput)]),
}

impl<'a> OpInputs<'a> {
    pub fn single(input: &'a NodeOutput) -> Self {
        OpInputs::Single(input)
    }

    pub fn named(inputs: &'a [(&'static str, &'a NodeOutput)]) -> Self {
        OpInputs::Named(inputs)
    }

    /// Get the unique input for [`OpInputs::Single`], or fail.
    pub fn require_single(&self) -> Result<&NodeOutput, OpError> {
        match self {
            OpInputs::Single(out) => Ok(out),
            OpInputs::Named(_) => Err(OpError::ArityMismatch {
                expected: "single input",
                got: "named multi-input",
            }),
        }
    }

    /// Look up a named port, or fail with a clear error.
    pub fn require_named(&self, port: &'static str) -> Result<&NodeOutput, OpError> {
        match self {
            OpInputs::Named(pairs) => pairs
                .iter()
                .find(|(name, _)| *name == port)
                .map(|(_, out)| *out)
                .ok_or(OpError::MissingInputPort { port }),
            OpInputs::Single(_) => Err(OpError::ArityMismatch {
                expected: "named multi-input",
                got: "single input",
            }),
        }
    }
}

/// Per-row execution context handed to ops.
///
/// Initially carries only the row index so panic-isolation messages can name
/// the offending row. Later steps will add a `SecurityPolicy` and shared
/// caches here without changing op signatures.
#[derive(Debug, Clone, Copy)]
pub struct ExecCtx {
    pub row_idx: usize,
}

impl ExecCtx {
    pub fn new(row_idx: usize) -> Self {
        Self { row_idx }
    }
}

/// Errors returned from [`Operation::execute`].
///
/// All variants are caller-visible (no `Other(String)` catch-all that hides
/// structure); the executor maps these into Polars `ComputeError` with the
/// row index attached.
#[derive(Debug, Error)]
pub enum OpError {
    #[error("op `{op}` rejected input dtype {dtype:?}")]
    UnsupportedInputDtype {
        op: &'static str,
        dtype: view_buffer::DType,
    },

    #[error("op `{op}` rejected input shape {shape:?}: {reason}")]
    InvalidInputShape {
        op: &'static str,
        shape: Vec<usize>,
        reason: &'static str,
    },

    #[error("op missing required input port `{port}`")]
    MissingInputPort { port: &'static str },

    #[error("input arity mismatch: expected {expected}, got {got}")]
    ArityMismatch {
        expected: &'static str,
        got: &'static str,
    },

    #[error("op `{op}` got unexpected input domain {got:?}, expected {expected:?}")]
    DomainMismatch {
        op: &'static str,
        expected: Domain,
        got: Domain,
    },

    #[error("op `{op}` failed: {message}")]
    Failed {
        op: &'static str,
        message: String,
    },
}

/// Convenience: build a `Failed` error from a string.
pub fn fail(op: &'static str, message: impl Into<String>) -> OpError {
    OpError::Failed {
        op,
        message: message.into(),
    }
}

impl From<crate::params::ParamError> for OpError {
    fn from(err: crate::params::ParamError) -> Self {
        use crate::params::ParamError;
        match err {
            ParamError::Missing { op, name } => OpError::Failed {
                op,
                message: format!("missing required parameter `{name}`"),
            },
            ParamError::WrongType {
                op, name, expected, got,
            } => OpError::Failed {
                op,
                message: format!("parameter `{name}`: expected {expected}, got {got}"),
            },
            ParamError::OutOfRange { op, name, message } => OpError::Failed {
                op,
                message: format!("parameter `{name}` out of range: {message}"),
            },
        }
    }
}

/// The operation trait — implemented once per built-in or extension op.
///
/// Implementations are constructed by [`OpRegistration::factory`] from a
/// resolved parameter map (literals only — expression-bound params are
/// resolved per row by the executor before the factory is called).
pub trait Operation: Send + Sync + 'static {
    /// Stable identifier (matches the Python op name and the registry key).
    fn name(&self) -> &'static str;

    /// What inputs this op consumes.
    fn input_arity(&self) -> InputArity;

    /// Domain expected on a given input port (or the sole input for `Unary`).
    fn input_domain(&self, port: &str) -> Domain;

    /// Domain produced on the output.
    fn output_domain(&self) -> Domain;

    /// Plan-time contract — drives dtype/ndim/alpha inference and the codegen.
    fn contract(&self) -> &'static OpContract;

    /// Execute the op on a single row.
    fn execute(&self, ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError>;
}

/// Trait object alias — the registry stores ops as `Arc<dyn Operation>` so
/// the executor can clone cheaply between rows.
pub type OpHandle = Arc<dyn Operation>;
