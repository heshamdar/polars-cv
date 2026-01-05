//! Data type definitions for view-buffer.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Supported data types for buffer elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
    U64,
    I64,
}

/// Categories of data types that operations can accept as input.
///
/// This enables operations to declare what types they can work with,
/// allowing the execution layer to handle automatic casting.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DTypeCategory {
    /// Accept all data types.
    #[default]
    Any,
    /// Accept all numeric types (all currently supported types).
    Numeric,
    /// Accept only integer types (u8, i8, u16, i16, u32, i32, u64, i64).
    Integer,
    /// Accept only floating-point types (f32, f64).
    Float,
    /// Accept only specific data types.
    Specific(Vec<DType>),
}

impl DTypeCategory {
    /// Check if a dtype is accepted by this category.
    pub fn accepts(&self, dtype: DType) -> bool {
        match self {
            DTypeCategory::Any => true,
            DTypeCategory::Numeric => true, // All current types are numeric
            DTypeCategory::Integer => matches!(
                dtype,
                DType::U8
                    | DType::I8
                    | DType::U16
                    | DType::I16
                    | DType::U32
                    | DType::I32
                    | DType::U64
                    | DType::I64
            ),
            DTypeCategory::Float => matches!(dtype, DType::F32 | DType::F64),
            DTypeCategory::Specific(allowed) => allowed.contains(&dtype),
        }
    }
}

/// Rules for determining output dtype of an operation.
///
/// This separates the semantic behavior of an operation from its
/// dtype mechanics, allowing for flexible and predictable pipelines.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OutputDTypeRule {
    /// Output dtype matches input dtype.
    #[default]
    PreserveInput,
    /// Output is always a fixed dtype (e.g., always F32).
    Fixed(DType),
    /// Default to a specific dtype, but can be overridden via out_dtype parameter.
    Configurable(DType),
    /// Promote integers to float32, preserve float types.
    PromoteToFloat,
    /// Force output to F64 (for reductions that need precision).
    ForceF64,
    /// Force output to I64 (for argmax/argmin).
    ForceI64,
    /// Force output to U64 (for count-based operations).
    ForceU64,
    /// Force output to U32 (for bin indices).
    ForceU32,
}

impl OutputDTypeRule {
    /// Resolve the output dtype given an input dtype and optional override.
    pub fn resolve(&self, input_dtype: DType, out_dtype_override: Option<DType>) -> DType {
        // If there's an explicit override, use it
        if let Some(override_dtype) = out_dtype_override {
            return override_dtype;
        }

        match self {
            OutputDTypeRule::PreserveInput => input_dtype,
            OutputDTypeRule::Fixed(dtype) => *dtype,
            OutputDTypeRule::Configurable(default) => *default,
            OutputDTypeRule::PromoteToFloat => {
                if matches!(input_dtype, DType::F32 | DType::F64) {
                    input_dtype
                } else {
                    DType::F32
                }
            }
            OutputDTypeRule::ForceF64 => DType::F64,
            OutputDTypeRule::ForceI64 => DType::I64,
            OutputDTypeRule::ForceU64 => DType::U64,
            OutputDTypeRule::ForceU32 => DType::U32,
        }
    }
}

impl DType {
    /// Returns the size in bytes of this data type.
    pub fn size_of(&self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }
}

/// Trait to map Rust types to DType enum.
pub trait ViewType: 'static + Copy + Send + Sync + std::fmt::Debug {
    /// The corresponding DType for this Rust type.
    const DTYPE: DType;
}

macro_rules! impl_view_type {
    ($rust_type:ty, $dtype:expr) => {
        impl ViewType for $rust_type {
            const DTYPE: DType = $dtype;
        }
    };
}

impl_view_type!(u8, DType::U8);
impl_view_type!(i8, DType::I8);
impl_view_type!(u16, DType::U16);
impl_view_type!(i16, DType::I16);
impl_view_type!(u32, DType::U32);
impl_view_type!(i32, DType::I32);
impl_view_type!(f32, DType::F32);
impl_view_type!(f64, DType::F64);
impl_view_type!(u64, DType::U64);
impl_view_type!(i64, DType::I64);
