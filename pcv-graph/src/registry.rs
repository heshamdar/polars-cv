//! Inventory-based op registry.
//!
//! Each op submits an [`OpRegistration`] via `inventory::submit!` from its own
//! file; the executor and the codegen iterate over [`iter_ops`] to discover
//! all built-in ops at link time. Adding a built-in op is a single-file change
//! and a single `submit!` call.

use crate::contract::OpContract;
use crate::op::{OpError, OpHandle};
use crate::params::ParamMap;

/// Compile-time registration of a built-in operation.
///
/// Constructed via `inventory::submit!` from each op module.
pub struct OpRegistration {
    /// String identifier (matches the Python op name).
    pub name: &'static str,
    /// Plan-time contract (dtype/ndim/alpha effects).
    pub contract: &'static OpContract,
    /// Returns the static schema descriptor for codegen.
    pub schema: fn() -> &'static OpSchemaDescriptor,
    /// Build an instance from a literal-resolved parameter map.
    ///
    /// Expression-bound params are materialized to literals by the bridge
    /// layer in `polars-cv` before this is called.
    pub factory: fn(&ParamMap) -> Result<OpHandle, OpError>,
}

inventory::collect!(OpRegistration);

/// Iterate registered ops in link order.
pub fn iter_ops() -> impl Iterator<Item = &'static OpRegistration> {
    inventory::iter::<OpRegistration>.into_iter()
}

/// Look up a registration by name (linear scan; the registry is small and
/// the lookup is plan-time only).
pub fn find_op(name: &str) -> Option<&'static OpRegistration> {
    iter_ops().find(|reg| reg.name == name)
}

/// Static schema for an op's parameters.
///
/// Drives codegen of the Python `OpContract` table and the `Pipeline` builder
/// methods. The values here are static so codegen can run without
/// instantiating ops.
pub struct OpSchemaDescriptor {
    pub name: &'static str,
    pub doc: &'static str,
    pub params: &'static [ParamSchema],
}

/// Single-parameter schema entry.
pub struct ParamSchema {
    pub name: &'static str,
    pub kind: ParamKind,
    pub required: bool,
    pub doc: &'static str,
}

/// Coarse parameter kind, sufficient for codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    Bool,
    Int,
    Float,
    String,
    /// Enum-like string; choices documented in `doc`.
    Enum,
    /// List of ints (e.g. shape, kernel size).
    IntList,
    /// List of floats (e.g. mean, std).
    FloatList,
    /// Free-form (struct, nested list, etc.).
    Any,
}

impl ParamKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ParamKind::Bool => "bool",
            ParamKind::Int => "int",
            ParamKind::Float => "float",
            ParamKind::String => "str",
            ParamKind::Enum => "enum",
            ParamKind::IntList => "list[int]",
            ParamKind::FloatList => "list[float]",
            ParamKind::Any => "Any",
        }
    }
}
