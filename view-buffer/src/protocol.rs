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

// Stable mapping for DType <-> u8 to ensure binary compatibility.
// The codes themselves live in `dtype_table!` (core/dtype.rs) alongside each
// dtype's other names; these remain as the protocol-facing spelling.
pub fn dtype_to_u8(dt: DType) -> u8 {
    dt.wire_code()
}

pub fn u8_to_dtype(code: u8) -> Option<DType> {
    DType::from_wire_code(code)
}
