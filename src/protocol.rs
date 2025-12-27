use crate::core::dtype::DType;
#[cfg(feature = "serde")]
use bytemuck::{Pod, Zeroable};

pub const MAGIC_BYTES: [u8; 4] = *b"VIEW";
pub const VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 64;

/// Fixed-size header for binary transport (64 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ViewHeader {
    pub magic: [u8; 4],     // "VIEW"
    pub version: u16,       // 1
    pub dtype: u8,          // Mapped from DType
    pub rank: u8,           // Number of dimensions
    pub data_offset: u64,   // Offset in bytes where raw data starts
    pub flags: u64,         // Reserved for future flags (e.g. compression, endianness)
    pub reserved: [u8; 40], // Padding to reach 64 bytes
}

#[cfg(feature = "serde")]
unsafe impl Zeroable for ViewHeader {}

#[cfg(feature = "serde")]
unsafe impl Pod for ViewHeader {}

impl Default for ViewHeader {
    fn default() -> Self {
        Self {
            magic: MAGIC_BYTES,
            version: VERSION,
            dtype: 0,
            rank: 0,
            data_offset: 0,
            flags: 0,
            reserved: [0; 40],
        }
    }
}

// Stable mapping for DType <-> u8 to ensure binary compatibility
pub fn dtype_to_u8(dt: DType) -> u8 {
    match dt {
        DType::U8 => 1,
        DType::I8 => 2,
        DType::U16 => 3,
        DType::I16 => 4,
        DType::U32 => 5,
        DType::I32 => 6,
        DType::F32 => 7,
        DType::F64 => 8,
        DType::U64 => 9,
        DType::I64 => 10,
    }
}

pub fn u8_to_dtype(code: u8) -> Option<DType> {
    match code {
        1 => Some(DType::U8),
        2 => Some(DType::I8),
        3 => Some(DType::U16),
        4 => Some(DType::I16),
        5 => Some(DType::U32),
        6 => Some(DType::I32),
        7 => Some(DType::F32),
        8 => Some(DType::F64),
        9 => Some(DType::U64),
        10 => Some(DType::I64),
        _ => None,
    }
}
