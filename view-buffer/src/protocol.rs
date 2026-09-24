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

/// A blob's payload layout, validated against the blob it was read from.
///
/// Produced only by [`parse_blob`], so holding one means every header field
/// has been checked: `data_offset..data_offset + data_len` lies inside the
/// blob, every element `(shape, strides)` can address lies inside that window,
/// and the offset and every stride are whole multiples of the element size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobLayout {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte strides of a non-contiguous layout; `None` means contiguous.
    pub strides: Option<Vec<isize>>,
    /// Start of the payload, in bytes from the start of the blob.
    pub data_offset: usize,
    /// Bytes from `data_offset` the layout may address: exactly the element
    /// bytes when contiguous, the rest of the blob when strided (a padded
    /// layout may span more than its logical size).
    pub data_len: usize,
}

/// Parse and validate a VIEW blob header. **The one blob parser**: the
/// zero-copy plugin decode and [`ViewBuffer::from_blob`] both read through it.
///
/// Every header field is untrusted column data. Beyond bounds, the payload
/// must be *aligned*: a typed view over it is a `&[T]`, and a misaligned one is
/// undefined behaviour, not a wrong value. The offset and strides are checked
/// relative to the blob's first byte, so the caller must place the blob at an
/// address aligned to at least the largest element size (8).
///
/// [`ViewBuffer::from_blob`]: crate::ViewBuffer::from_blob
pub fn parse_blob(data: &[u8]) -> Result<BlobLayout, String> {
    let u64_at = |pos: usize, what: &str| -> Result<u64, String> {
        data.get(pos..pos + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8-byte slice")))
            .ok_or_else(|| format!("Blob truncated reading {what}"))
    };
    if data.len() < HEADER_SIZE {
        return Err("Blob data too short for header".into());
    }
    if data[0..4] != MAGIC_BYTES {
        return Err("Invalid blob magic bytes".into());
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != VERSION {
        return Err(format!("Unsupported blob version: {version}"));
    }
    let dtype = u8_to_dtype(data[6]).ok_or_else(|| format!("Unknown dtype code: {}", data[6]))?;
    let rank = data[7] as usize;
    let data_offset = usize::try_from(u64_at(8, "data offset")?)
        .map_err(|_| "Blob data offset overflow".to_string())?;
    let flags = u64_at(16, "flags")?;

    let shape = (0..rank)
        .map(|i| Ok(u64_at(HEADER_SIZE + i * 8, "shape")? as usize))
        .collect::<Result<Vec<usize>, String>>()?;
    let strides = (0..rank)
        .map(|i| Ok(u64_at(HEADER_SIZE + (rank + i) * 8, "strides")? as i64 as isize))
        .collect::<Result<Vec<isize>, String>>()?;

    let elem = dtype.size_of();
    let num_elements = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "Shape product overflow: dimensions too large".to_string())?;
    let element_bytes = num_elements
        .checked_mul(elem)
        .ok_or_else(|| "Data length overflow: buffer too large".to_string())?;
    // Checked: a near-usize::MAX offset must not wrap below the blob length.
    let data_end = data_offset
        .checked_add(element_bytes)
        .ok_or_else(|| "Blob data offset overflow".to_string())?;
    if data_end > data.len() {
        return Err(format!(
            "Blob data truncated: offset={data_offset}, expected={element_bytes}, total={}",
            data.len()
        ));
    }
    if !data_offset.is_multiple_of(elem) {
        return Err(format!(
            "Blob data offset {data_offset} is not aligned to its {dtype:?} element size ({elem} bytes)"
        ));
    }

    if flags == 1 || strides.is_empty() {
        return Ok(BlobLayout {
            dtype,
            shape,
            strides: None,
            data_offset,
            data_len: element_bytes,
        });
    }

    if let Some(bad) = strides.iter().find(|&&s| s % elem as isize != 0) {
        return Err(format!(
            "Blob stride {bad} is not aligned to its {dtype:?} element size ({elem} bytes): \
             strides={strides:?}"
        ));
    }
    // Every element the (shape, strides) pair can address must fall inside
    // the window from data_offset to the end of the blob.
    let window_len = data.len() - data_offset;
    if num_elements > 0 {
        let (mut min_reach, mut max_reach) = (0i128, 0i128);
        for (&dim, &stride) in shape.iter().zip(&strides) {
            let reach = (dim as i128 - 1) * stride as i128;
            if reach >= 0 {
                max_reach += reach;
            } else {
                min_reach += reach;
            }
        }
        if min_reach < 0 {
            return Err(format!(
                "Blob strides reach below the data start: shape={shape:?}, strides={strides:?}"
            ));
        }
        let span_end = max_reach + elem as i128;
        if span_end > window_len as i128 {
            return Err(format!(
                "Blob strides reach outside the data: shape={shape:?}, \
                 strides={strides:?}, span={span_end}, available={window_len}"
            ));
        }
    }
    Ok(BlobLayout {
        dtype,
        shape,
        strides: Some(strides),
        data_offset,
        data_len: window_len,
    })
}
