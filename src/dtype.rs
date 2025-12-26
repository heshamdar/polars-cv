
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub fn size_of(&self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }
}

/// Trait to map Rust types to DType enum
pub trait TensorType: 'static + Copy + Send + Sync + std::fmt::Debug {
    const DTYPE: DType;
}

macro_rules! impl_tensor_type {
    ($rust_type:ty, $dtype:expr) => {
        impl TensorType for $rust_type {
            const DTYPE: DType = $dtype;
        }
    };
}

impl_tensor_type!(u8, DType::U8);
impl_tensor_type!(i8, DType::I8);
impl_tensor_type!(u16, DType::U16);
impl_tensor_type!(i16, DType::I16);
impl_tensor_type!(u32, DType::U32);
impl_tensor_type!(i32, DType::I32);
impl_tensor_type!(f32, DType::F32);
impl_tensor_type!(f64, DType::F64);
impl_tensor_type!(u64, DType::U64);
impl_tensor_type!(i64, DType::I64);