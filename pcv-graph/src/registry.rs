//! Inventory-based op registry.
//!
//! Each op submits an [`OpRegistration`] via `inventory::submit!` from its own
//! file; the executor and the codegen iterate over `inventory::iter` to
//! discover all built-in ops at link time. Adding a built-in op is a single-file
//! change and a single `submit!` call.
//!
//! Schemas exposed through [`OpSchemaDescriptor`] are consumed by the
//! `dump_schema` binary to regenerate the Python contract table.

use crate::contract::OpContract;

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
}

inventory::collect!(OpRegistration);

/// Iterate registered ops in link order.
pub fn iter_ops() -> impl Iterator<Item = &'static OpRegistration> {
    inventory::iter::<OpRegistration>.into_iter()
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
