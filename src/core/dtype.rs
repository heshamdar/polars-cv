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
