//! Buffer storage and view types.

use std::sync::Arc;

use num_traits::AsPrimitive;
use thiserror::Error;

use crate::core::dtype::{DType, ViewType};
use crate::core::layout::{ExternalLayout, Layout, LayoutFacts, LayoutReport};
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::protocol::{dtype_to_u8, u8_to_dtype, ViewHeader, HEADER_SIZE, MAGIC_BYTES, VERSION};

/// Errors that can occur during buffer operations.
#[derive(Error, Debug)]
pub enum BufferError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: DType, got: DType },
    #[error("Buffer is not contiguous")]
    NotContiguous,
    #[error("Layout incompatible with target: {target:?}")]
    IncompatibleLayout { target: ExternalLayout },
    #[error("Invalid binary protocol: {0}")]
    InvalidProtocol(String),
}

/// Storage backend for ViewBuffer data.
#[derive(Debug, Clone)]
pub enum BufferStorage {
    /// Owned Rust Vec wrapped in Arc for cheap cloning.
    Rust(Arc<Vec<u8>>),
    /// Arrow buffer for zero-copy interop.
    #[cfg(feature = "arrow_interop")]
    Arrow(arrow::buffer::Buffer),
    /// Polars-arrow buffer for zero-copy Polars integration.
    /// The offset field allows referencing a slice within the buffer.
    #[cfg(feature = "polars_interop")]
    PolarsArrow {
        /// The underlying polars-arrow buffer (Arc-backed, cheap to clone).
        buffer: polars_arrow::buffer::Buffer<u8>,
        /// Byte offset into the buffer where this view starts.
        offset: usize,
        /// Length of this view in bytes.
        len: usize,
    },
}

impl BufferStorage {
    /// Returns a raw pointer to the start of the buffer.
    ///
    /// For PolarsArrow storage, this returns a pointer to the start of the view
    /// (i.e., buffer start + offset).
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            BufferStorage::Rust(v) => v.as_ptr(),
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(b) => b.as_ptr(),
            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { buffer, offset, .. } => {
                // Safety: offset is validated at construction time to be within bounds
                unsafe { buffer.as_ptr().add(*offset) }
            }
        }
    }

    /// Returns the length of the underlying byte buffer.
    ///
    /// For PolarsArrow storage, this returns the length of the view, not the entire buffer.
    pub fn len(&self) -> usize {
        match self {
            BufferStorage::Rust(v) => v.len(),
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(b) => b.len(),
            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { len, .. } => *len,
        }
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A view over a contiguous or strided buffer of typed elements.
#[derive(Debug, Clone)]
pub struct ViewBuffer {
    pub(crate) data: BufferStorage,
    pub(crate) layout: Layout,
}

/// Default SIMD alignment (64 bytes for AVX-512 compatibility).
pub const SIMD_ALIGNMENT: usize = 64;

impl ViewBuffer {
    /// Creates a ViewBuffer from a Vec of typed elements.
    pub fn from_vec<T: ViewType>(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        let dtype = T::DTYPE;
        let layout = Layout::new_contiguous(shape, dtype);

        // SAFETY:
        // 1. T is Copy, so no Drop glue is needed.
        // 2. Alignment: The allocation is created by Vec<T>, so it is aligned for T.
        //    Converting to Vec<u8> (align 1) is safe.
        //    We must ensure we don't re-interpret these bytes as a type with higher
        //    alignment requirements than T later without checking (enforced by as_ptr check).
        let data_bytes = unsafe {
            let mut v_clone = std::mem::ManuallyDrop::new(data);
            let ptr = v_clone.as_mut_ptr() as *mut u8;
            let len = v_clone.len() * std::mem::size_of::<T>();
            let cap = v_clone.capacity() * std::mem::size_of::<T>();
            Vec::from_raw_parts(ptr, len, cap)
        };

        Self {
            data: BufferStorage::Rust(Arc::new(data_bytes)),
            layout,
        }
    }

    /// Creates a ViewBuffer from a slice of typed elements with SIMD-friendly alignment.
    ///
    /// The buffer is allocated with the specified alignment (default 64 bytes for AVX-512).
    /// This enables efficient SIMD processing in fused kernels.
    ///
    /// # Arguments
    /// * `data` - Slice of elements to copy into the aligned buffer.
    /// * `alignment` - Alignment in bytes (must be power of 2, typically 32 or 64).
    ///
    /// # Example
    /// ```
    /// use view_buffer::ViewBuffer;
    /// let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    /// let aligned_buf = ViewBuffer::from_slice_aligned(&data, 64);
    /// assert!(aligned_buf.is_aligned(64));
    /// ```
    pub fn from_slice_aligned<T: ViewType>(data: &[T], alignment: usize) -> Self {
        debug_assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        debug_assert!(
            alignment >= std::mem::align_of::<T>(),
            "Alignment must be >= type alignment"
        );

        let len_bytes = std::mem::size_of_val(data);
        let alloc_layout = std::alloc::Layout::from_size_align(len_bytes, alignment)
            .expect("Invalid layout parameters");

        // Allocate aligned memory
        let aligned_ptr = unsafe { std::alloc::alloc(alloc_layout) };
        if aligned_ptr.is_null() {
            std::alloc::handle_alloc_error(alloc_layout);
        }

        // Copy data to aligned buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, aligned_ptr, len_bytes);
        }

        // Create a Vec from the aligned allocation
        let aligned_vec = unsafe { Vec::from_raw_parts(aligned_ptr, len_bytes, len_bytes) };

        let shape = vec![data.len()];
        let dtype = T::DTYPE;
        let layout = Layout::new_contiguous(shape, dtype);

        Self {
            data: BufferStorage::Rust(Arc::new(aligned_vec)),
            layout,
        }
    }

    /// Creates a ViewBuffer with default SIMD alignment (64 bytes).
    pub fn from_slice_simd_aligned<T: ViewType>(data: &[T]) -> Self {
        Self::from_slice_aligned(data, SIMD_ALIGNMENT)
    }

    /// Creates a ViewBuffer from a Vec with a specific shape.
    ///
    /// # Arguments
    /// * `data` - Vector of elements.
    /// * `shape` - Shape of the resulting buffer.
    ///
    /// # Panics
    /// Panics if the data length doesn't match the shape product.
    pub fn from_vec_with_shape<T: ViewType>(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let dtype = T::DTYPE;
        let layout = Layout::new_contiguous(shape, dtype);

        let data_bytes = unsafe {
            let mut v_clone = std::mem::ManuallyDrop::new(data);
            let ptr = v_clone.as_mut_ptr() as *mut u8;
            let len = v_clone.len() * std::mem::size_of::<T>();
            let cap = v_clone.capacity() * std::mem::size_of::<T>();
            Vec::from_raw_parts(ptr, len, cap)
        };

        Self {
            data: BufferStorage::Rust(Arc::new(data_bytes)),
            layout,
        }
    }

    /// Creates a scalar ViewBuffer (shape [1]).
    pub fn from_scalar<T: ViewType>(value: T) -> Self {
        Self::from_vec_with_shape(vec![value], vec![1])
    }

    /// Cast buffer elements to a different dtype.
    ///
    /// This creates a new buffer with the converted values.
    pub fn cast_to(&self, target_dtype: DType) -> Self {
        if self.layout.dtype == target_dtype {
            return self.clone();
        }

        let contig = self.to_contiguous();
        let shape = contig.shape().to_vec();
        let _len: usize = shape.iter().product();

        // Macro to handle all dtype combinations
        macro_rules! cast_impl {
            ($src:ty, $dst_dtype:expr) => {{
                let src_data = contig.as_slice::<$src>();
                match $dst_dtype {
                    DType::U8 => {
                        let data: Vec<u8> = src_data.iter().map(|&x| x as u8).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::I8 => {
                        let data: Vec<i8> = src_data.iter().map(|&x| x as i8).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::U16 => {
                        let data: Vec<u16> = src_data.iter().map(|&x| x as u16).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::I16 => {
                        let data: Vec<i16> = src_data.iter().map(|&x| x as i16).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::U32 => {
                        let data: Vec<u32> = src_data.iter().map(|&x| x as u32).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::I32 => {
                        let data: Vec<i32> = src_data.iter().map(|&x| x as i32).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::U64 => {
                        let data: Vec<u64> = src_data.iter().map(|&x| x as u64).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::I64 => {
                        let data: Vec<i64> = src_data.iter().map(|&x| x as i64).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::F32 => {
                        let data: Vec<f32> = src_data.iter().map(|&x| x as f32).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                    DType::F64 => {
                        let data: Vec<f64> = src_data.iter().map(|&x| x as f64).collect();
                        Self::from_vec_with_shape(data, shape)
                    }
                }
            }};
        }

        match self.layout.dtype {
            DType::U8 => cast_impl!(u8, target_dtype),
            DType::I8 => cast_impl!(i8, target_dtype),
            DType::U16 => cast_impl!(u16, target_dtype),
            DType::I16 => cast_impl!(i16, target_dtype),
            DType::U32 => cast_impl!(u32, target_dtype),
            DType::I32 => cast_impl!(i32, target_dtype),
            DType::U64 => cast_impl!(u64, target_dtype),
            DType::I64 => cast_impl!(i64, target_dtype),
            DType::F32 => cast_impl!(f32, target_dtype),
            DType::F64 => cast_impl!(f64, target_dtype),
        }
    }

    /// Returns true if the buffer data is aligned to the specified boundary.
    ///
    /// # Arguments
    /// * `alignment` - Alignment to check in bytes (must be power of 2).
    pub fn is_aligned(&self, alignment: usize) -> bool {
        debug_assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        let ptr = self.data.as_ptr();
        (ptr as usize).is_multiple_of(alignment)
    }

    /// Returns true if the buffer is aligned for SIMD operations (64-byte alignment).
    pub fn is_simd_aligned(&self) -> bool {
        self.is_aligned(SIMD_ALIGNMENT)
    }

    /// Creates a ViewBuffer from an Arrow buffer (zero-copy).
    #[cfg(feature = "arrow_interop")]
    pub fn from_arrow_buffer(
        buffer: arrow::buffer::Buffer,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let layout = Layout::new_contiguous(shape, dtype);
        Self {
            data: BufferStorage::Arrow(buffer),
            layout,
        }
    }

    /// Creates a ViewBuffer from a Polars-arrow buffer (zero-copy).
    ///
    /// This enables zero-copy data ingestion from Polars columns.
    ///
    /// # Arguments
    /// * `buffer` - The polars-arrow buffer containing the data.
    /// * `offset` - Byte offset into the buffer where this view starts.
    /// * `shape` - Shape of the resulting tensor.
    /// * `dtype` - Data type of the elements.
    ///
    /// # Panics
    /// Panics if `offset + required_bytes > buffer.len()`.
    #[cfg(feature = "polars_interop")]
    pub fn from_polars_buffer(
        buffer: polars_arrow::buffer::Buffer<u8>,
        offset: usize,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let num_elements: usize = shape.iter().product();
        let required_bytes = num_elements * dtype.size_of();

        assert!(
            offset + required_bytes <= buffer.len(),
            "Polars buffer too small: offset={}, required={}, buffer_len={}",
            offset,
            required_bytes,
            buffer.len()
        );

        let layout = Layout::new_contiguous(shape, dtype);
        Self {
            data: BufferStorage::PolarsArrow {
                buffer,
                offset,
                len: required_bytes,
            },
            layout,
        }
    }

    /// Creates a ViewBuffer from a Polars-arrow buffer slice (zero-copy).
    ///
    /// This is a convenience method when you already know the exact byte length.
    ///
    /// # Arguments
    /// * `buffer` - The polars-arrow buffer containing the data.
    /// * `offset` - Byte offset into the buffer where this view starts.
    /// * `len` - Length of this view in bytes.
    /// * `shape` - Shape of the resulting tensor.
    /// * `dtype` - Data type of the elements.
    ///
    /// # Panics
    /// Panics if `offset + len > buffer.len()` or if `len` doesn't match
    /// `shape.product() * dtype.size_of()`.
    #[cfg(feature = "polars_interop")]
    pub fn from_polars_buffer_slice(
        buffer: polars_arrow::buffer::Buffer<u8>,
        offset: usize,
        len: usize,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let num_elements: usize = shape.iter().product();
        let expected_bytes = num_elements * dtype.size_of();

        assert!(
            offset + len <= buffer.len(),
            "Polars buffer slice out of bounds: offset={}, len={}, buffer_len={}",
            offset,
            len,
            buffer.len()
        );
        assert!(
            len == expected_bytes,
            "Byte length mismatch: provided={}, expected={} (shape={:?}, dtype={:?})",
            len,
            expected_bytes,
            shape,
            dtype
        );

        let layout = Layout::new_contiguous(shape, dtype);
        Self {
            data: BufferStorage::PolarsArrow {
                buffer,
                offset,
                len,
            },
            layout,
        }
    }

    /// Returns the data type of the buffer elements.
    pub fn dtype(&self) -> DType {
        self.layout.dtype
    }

    /// Returns the shape of the buffer.
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    /// Returns the strides in bytes.
    pub fn strides_bytes(&self) -> &[isize] {
        &self.layout.strides
    }

    /// Returns a raw pointer to the start of the view data.
    ///
    /// # Safety
    /// Caller must ensure that:
    /// 1. The resulting pointer is not accessed out of bounds.
    /// 2. The data at this pointer is valid for type T.
    pub unsafe fn as_ptr<T>(&self) -> *const T {
        let ptr = self.data.as_ptr().add(self.layout.offset);

        // Safety Recommendation 1: Alignment Check
        // We use debug_assert to catch this in testing/debug builds.
        debug_assert!(
            (ptr as usize).is_multiple_of(std::mem::align_of::<T>()),
            "ViewBuffer pointer is not aligned for type {}; address={:p}, align={}",
            std::any::type_name::<T>(),
            ptr,
            std::mem::align_of::<T>()
        );

        ptr as *const T
    }

    /// Returns raw parts of the buffer for low-level access.
    pub fn as_raw_parts(&self) -> (*const u8, &[usize], &[isize], DType) {
        (
            unsafe { self.data.as_ptr().add(self.layout.offset) },
            &self.layout.shape,
            &self.layout.strides,
            self.layout.dtype,
        )
    }

    /// Returns a typed slice of the buffer data.
    ///
    /// # Panics
    /// Panics if the buffer is not contiguous.
    ///
    /// # Safety Note
    /// The caller must ensure the type T matches the buffer's dtype.
    pub fn as_slice<T: ViewType>(&self) -> &[T] {
        assert!(
            self.layout.is_contiguous(),
            "Buffer must be contiguous to get a slice. Call to_contiguous() first."
        );

        let len: usize = self.layout.shape.iter().product();
        unsafe {
            let ptr = self.as_ptr::<T>();
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Returns a unique identifier for the underlying storage.
    /// Used for zero-copy verification in tests.
    pub fn storage_id(&self) -> usize {
        match &self.data {
            BufferStorage::Rust(arc) => Arc::as_ptr(arc) as usize,
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(buf) => buf.as_ptr() as usize,
            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { buffer, offset, .. } => {
                // Include offset in the ID to distinguish different views into the same buffer
                buffer.as_ptr() as usize + offset
            }
        }
    }

    // --- Zero-Copy Ownership Transfer ---

    /// Try to extract the underlying Vec without copying.
    ///
    /// This consumes the ViewBuffer and attempts to extract the owned data.
    /// Returns `Some(Vec<u8>)` if:
    /// - Storage is `Rust(Arc<Vec<u8>>)` with a single owner (refcount == 1)
    /// - Buffer is contiguous (no strided views)
    /// - Layout offset is 0 (full buffer, not a slice)
    ///
    /// Returns `None` if zero-copy extraction is not possible, in which case
    /// the caller should use `to_contiguous()` and copy the data.
    ///
    /// # Example
    /// ```
    /// use view_buffer::ViewBuffer;
    ///
    /// let buf = ViewBuffer::from_vec(vec![1u8, 2, 3, 4]);
    /// if let Some(owned) = buf.try_into_owned_bytes() {
    ///     // Zero-copy: we now own the Vec
    ///     assert_eq!(owned, vec![1, 2, 3, 4]);
    /// }
    /// ```
    pub fn try_into_owned_bytes(self) -> Option<Vec<u8>> {
        // Must be contiguous with no offset (check before moving data)
        if !self.layout.is_contiguous() || self.layout.offset != 0 {
            return None;
        }

        // Only Rust storage can be unwrapped
        match self.data {
            BufferStorage::Rust(arc) => {
                // Try to unwrap the Arc - only succeeds if refcount == 1
                Arc::try_unwrap(arc).ok()
            }
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => unreachable!("try_into_owned_bytes called on Arrow buffer"),
            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { .. } => unreachable!("try_into_owned_bytes called on PolarsArrow buffer"),
        }
    }

    /// Convert to a polars-arrow Buffer, zero-copy when possible.
    ///
    /// This consumes the ViewBuffer and returns a polars Buffer suitable
    /// for constructing Polars Series/ChunkedArrays.
    ///
    /// Zero-copy occurs when:
    /// - Storage is `Rust(Arc<Vec<u8>>)` with a single owner
    /// - Buffer is contiguous with no offset
    ///
    /// Falls back to copying the data otherwise.
    ///
    /// # Returns
    /// A tuple of `(Buffer<u8>, shape, dtype)` for use in output encoding.
    #[cfg(feature = "polars_interop")]
    pub fn into_polars_buffer(self) -> (polars_arrow::buffer::Buffer<u8>, Vec<usize>, DType) {
        let shape = self.layout.shape.clone();
        let dtype = self.layout.dtype;

        // Check if zero-copy is possible before consuming self
        let can_zero_copy = self.can_zero_copy_transfer();

        if can_zero_copy {
            // Safe to try zero-copy - we know it will succeed
            if let Some(owned_vec) = self.try_into_owned_bytes() {
                let buffer = polars_arrow::buffer::Buffer::from(owned_vec);
                return (buffer, shape, dtype);
            }
            // Shouldn't reach here if can_zero_copy_transfer is correct
            unreachable!("can_zero_copy_transfer returned true but try_into_owned_bytes failed");
        }

        // Fallback: ensure contiguous and copy
        let contig = self.to_contiguous();
        let num_elements: usize = contig.layout.shape.iter().product();
        let data_len = num_elements * contig.layout.dtype.size_of();

        let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
        let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());

        (buffer, shape, dtype)
    }

    /// Check if this buffer can be zero-copy transferred.
    ///
    /// Returns `true` if `try_into_owned_bytes()` would succeed.
    /// Useful for testing and debugging zero-copy behavior.
    pub fn can_zero_copy_transfer(&self) -> bool {
        match &self.data {
            BufferStorage::Rust(arc) => {
                // Check if we're the sole owner and layout allows extraction
                Arc::strong_count(arc) == 1
                    && self.layout.is_contiguous()
                    && self.layout.offset == 0
            }
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => false,
            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { .. } => false,
        }
    }

    /// Returns layout facts for this buffer.
    pub fn layout_facts(&self) -> LayoutFacts {
        LayoutFacts::from(&self.layout)
    }

    /// Returns true if the buffer is compatible with the target external layout.
    pub fn is_compatible_with(&self, target: ExternalLayout) -> bool {
        self.layout_facts().compatible_with(target)
    }

    /// Returns a layout report for inspection.
    pub fn layout_report(&self) -> LayoutReport {
        let facts = self.layout_facts();
        LayoutReport {
            shape: facts.shape.clone(),
            strides: facts.strides.clone(),
            dtype: facts.dtype,
            contiguous: facts.is_contiguous(),
            image_compatible: facts.compatible_with(ExternalLayout::ImageCrate),
            ndarray_compatible: facts.compatible_with(ExternalLayout::NdArray),
        }
    }

    // --- Serialization (Protocol) ---

    /// Serializes the view to a binary blob (ViewBlob format).
    /// Always forces materialization to contiguous layout for transport efficiency.
    pub fn to_blob(&self) -> Vec<u8> {
        // 1. Ensure Contiguous
        let buffer = self.to_contiguous();
        let shape = buffer.shape();
        let strides = buffer.strides_bytes();
        let dtype = buffer.dtype();
        let rank = shape.len();

        // 2. Prepare Metadata
        let shape_bytes_len = rank * 8; // u64 per dim
        let stride_bytes_len = rank * 8; // i64 per dim
        let data_offset = (HEADER_SIZE + shape_bytes_len + stride_bytes_len) as u64;

        let header = ViewHeader {
            magic: MAGIC_BYTES,
            version: VERSION,
            dtype: dtype_to_u8(dtype),
            rank: rank as u8,
            data_offset,
            flags: 1, // 1 = Contiguous
            reserved: [0; 40],
        };

        // 3. Allocate Output Vector
        // Size = Header + ShapeArr + StrideArr + Data
        let data_len = buffer.data.len();
        let total_size = (data_offset as usize) + data_len;
        let mut blob = Vec::with_capacity(total_size);

        // 4. Write Parts
        // Header
        // Use unsafe copy to bytes for the #[repr(C)] struct
        let header_slice = unsafe {
            std::slice::from_raw_parts(&header as *const ViewHeader as *const u8, HEADER_SIZE)
        };
        blob.extend_from_slice(header_slice);

        // Shape (u64)
        for &dim in shape {
            blob.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        // Strides (i64)
        for &stride in strides {
            blob.extend_from_slice(&(stride as i64).to_le_bytes());
        }

        // Data
        // Since we called to_contiguous, the data is just the raw buffer content.
        // We use the pointer to copy the bytes.
        let raw_ptr = unsafe { buffer.as_ptr::<u8>() };
        let raw_slice = unsafe { std::slice::from_raw_parts(raw_ptr, data_len) };
        blob.extend_from_slice(raw_slice);

        blob
    }

    /// Deserializes a ViewBuffer from a binary blob.
    /// Currently performs a copy of the data payload into a new Vec<u8>.
    pub fn from_blob(data: &[u8]) -> Result<ViewBuffer, BufferError> {
        if data.len() < HEADER_SIZE {
            return Err(BufferError::InvalidProtocol(
                "Data too short for header".into(),
            ));
        }

        // 1. Read Header
        // Unsafe cast from bytes to struct (valid due to #[repr(C)] and POD nature)
        let header = unsafe { &*(data.as_ptr() as *const ViewHeader) };

        // Validate Magic
        if header.magic != MAGIC_BYTES {
            return Err(BufferError::InvalidProtocol("Invalid magic bytes".into()));
        }
        if header.version != VERSION {
            return Err(BufferError::InvalidProtocol(format!(
                "Unsupported version: {}",
                header.version
            )));
        }

        let rank = header.rank as usize;
        let dtype = u8_to_dtype(header.dtype).ok_or_else(|| {
            BufferError::InvalidProtocol(format!("Unknown dtype code: {}", header.dtype))
        })?;
        let data_offset = header.data_offset as usize;

        // 2. Read Shape & Strides
        let shape_start = HEADER_SIZE;
        let stride_start = shape_start + (rank * 8);

        if data_offset > data.len() {
            return Err(BufferError::InvalidProtocol(
                "Data offset out of bounds".into(),
            ));
        }

        let mut shape = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);

        let mut pos = shape_start;
        for _ in 0..rank {
            if pos + 8 > data.len() {
                return Err(BufferError::InvalidProtocol("Truncated shape data".into()));
            }
            let bytes: [u8; 8] = data[pos..pos + 8].try_into().unwrap();
            shape.push(u64::from_le_bytes(bytes) as usize);
            pos += 8;
        }

        pos = stride_start;
        for _ in 0..rank {
            if pos + 8 > data.len() {
                return Err(BufferError::InvalidProtocol("Truncated stride data".into()));
            }
            let bytes: [u8; 8] = data[pos..pos + 8].try_into().unwrap();
            strides.push(i64::from_le_bytes(bytes) as isize);
            pos += 8;
        }

        // 3. Extract Data
        // Safe Baseline: Copy into new Vec
        let raw_data = &data[data_offset..];

        // Validate size against shape/dtype
        let expected_elements: usize = shape.iter().product();
        let expected_bytes = expected_elements * dtype.size_of();

        if raw_data.len() < expected_bytes {
            return Err(BufferError::InvalidProtocol(format!(
                "Data payload too short. Expected {} bytes, got {}",
                expected_bytes,
                raw_data.len()
            )));
        }

        // Create owned buffer
        let vec_data = raw_data[0..expected_bytes].to_vec();

        // 4. Construct ViewBuffer
        let layout = Layout::new_contiguous(shape, dtype);

        Ok(ViewBuffer {
            data: BufferStorage::Rust(Arc::new(vec_data)),
            layout,
        })
    }

    // --- Views ---

    /// Permutes the dimensions of the buffer.
    pub fn permute(&self, dims: &[usize]) -> Self {
        let mut new_shape = vec![0; self.layout.shape.len()];
        let mut new_strides = vec![0; self.layout.strides.len()];

        for (i, &p) in dims.iter().enumerate() {
            new_shape[i] = self.layout.shape[p];
            new_strides[i] = self.layout.strides[p];
        }

        Self {
            data: self.data.clone(),
            layout: Layout {
                shape: new_shape,
                strides: new_strides,
                offset: self.layout.offset,
                dtype: self.layout.dtype,
            },
        }
    }

    /// Slices the buffer along all dimensions.
    ///
    /// Start and end indices are clamped to valid ranges. If an end index
    /// exceeds the dimension size, it is clamped to the dimension size.
    /// If a start index exceeds the dimension size, it is clamped and
    /// the resulting dimension will have size 0.
    pub fn slice(&self, start: &[usize], end: &[usize]) -> Self {
        let mut new_offset = self.layout.offset as isize;
        let mut new_shape = Vec::new();

        for i in 0..self.layout.shape.len() {
            let dim_size = self.layout.shape[i];
            // Clamp start and end to valid bounds
            let s = start[i].min(dim_size);
            let e = end[i].min(dim_size);
            // Ensure end >= start to avoid underflow
            let dim_len = e.saturating_sub(s);

            new_offset += (s as isize) * self.layout.strides[i];
            new_shape.push(dim_len);
        }

        Self {
            data: self.data.clone(),
            layout: Layout {
                shape: new_shape,
                strides: self.layout.strides.clone(),
                offset: new_offset as usize,
                dtype: self.layout.dtype,
            },
        }
    }

    /// Flips the buffer along the specified axes.
    pub fn flip(&self, axes: &[usize]) -> Self {
        let mut new_strides = self.layout.strides.clone();
        let mut new_offset = self.layout.offset as isize;

        for &axis in axes {
            let dim_len = self.layout.shape[axis];
            let stride = self.layout.strides[axis];
            new_offset += (dim_len as isize - 1) * stride;
            new_strides[axis] = -stride;
        }

        Self {
            data: self.data.clone(),
            layout: Layout {
                shape: self.layout.shape.clone(),
                strides: new_strides,
                offset: new_offset as usize,
                dtype: self.layout.dtype,
            },
        }
    }

    // --- Compute / Materialization ---

    /// Converts the buffer to a contiguous layout, copying if necessary.
    ///
    /// # Panics
    /// Panics if the total allocation size would overflow `usize`.
    pub fn to_contiguous(&self) -> Self {
        if self.layout.is_contiguous() {
            return self.clone();
        }

        // Use checked arithmetic to detect overflow early with a clear error message
        let total_elems: usize = self
            .layout
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .expect("shape product overflow: buffer dimensions are too large");

        let dtype_size = self.dtype().size_of();
        let total_bytes = total_elems
            .checked_mul(dtype_size)
            .expect("allocation size overflow: buffer is too large to materialize");

        let mut new_data = Vec::with_capacity(total_bytes);

        let mut indices = vec![0; self.layout.shape.len()];
        let shape = &self.layout.shape;
        let strides = &self.layout.strides;
        let ptr = self.data.as_ptr();
        let base_offset = self.layout.offset;
        let data_len = self.data.len();

        for _ in 0..total_elems {
            let mut offset = base_offset as isize;
            for (dim, &idx) in indices.iter().enumerate() {
                offset += (idx as isize) * strides[dim];
            }

            // Safety Recommendation 2: Bounds Checking in Debug
            debug_assert!(offset >= 0, "Negative offset calculation");
            debug_assert!(
                (offset as usize) < data_len,
                "Offset out of bounds: {offset} vs len {data_len}"
            );

            unsafe {
                let src = ptr.offset(offset);
                // Ensure we don't read past end when reading the scalar value
                debug_assert!(
                    (offset as usize) + dtype_size <= data_len,
                    "Read overrun during compaction"
                );

                for k in 0..dtype_size {
                    new_data.push(*src.add(k));
                }
            }

            for dim in (0..shape.len()).rev() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }

        let new_layout = Layout::new_contiguous(self.layout.shape.clone(), self.dtype());
        Self {
            data: BufferStorage::Rust(Arc::new(new_data)),
            layout: new_layout,
        }
    }

    /// Applies a fused kernel of scalar operations element-wise.
    ///
    /// This function is optimized for SIMD processing when the buffer is contiguous.
    /// For non-contiguous buffers, it falls back to a strided loop.
    pub fn apply_fused_kernel(&self, kernel: &FusedKernel) -> ViewBuffer {
        if self.dtype() != DType::F32 {
            panic!("FusedKernel currently only supports F32 views");
        }

        let total_elems: usize = self.layout.shape.iter().product();

        // Fast path: contiguous buffer - use SIMD-friendly processing
        if self.layout.is_contiguous() {
            return self.apply_fused_kernel_contiguous(kernel, total_elems);
        }

        // Slow path: strided buffer
        self.apply_fused_kernel_strided(kernel, total_elems)
    }

    /// SIMD-optimized fused kernel for contiguous buffers.
    #[inline]
    fn apply_fused_kernel_contiguous(
        &self,
        kernel: &FusedKernel,
        total_elems: usize,
    ) -> ViewBuffer {
        let mut output = Vec::with_capacity(total_elems);

        let src_ptr = unsafe { self.data.as_ptr().add(self.layout.offset) as *const f32 };
        let src = unsafe { std::slice::from_raw_parts(src_ptr, total_elems) };

        // Process in chunks of 8 for better vectorization (f32 x 8 = 256 bits = AVX)
        const CHUNK_SIZE: usize = 8;
        let chunks = total_elems / CHUNK_SIZE;
        let remainder = total_elems % CHUNK_SIZE;

        // Process main chunks - compiler can auto-vectorize this
        for chunk_idx in 0..chunks {
            let base = chunk_idx * CHUNK_SIZE;

            // Read chunk (hint for SIMD)
            let mut acc = [0.0f32; CHUNK_SIZE];
            acc.copy_from_slice(&src[base..base + CHUNK_SIZE]);

            // Apply all operations to the chunk
            for op in &kernel.ops {
                match op {
                    ScalarOp::Add(c) => {
                        for v in &mut acc {
                            *v += c;
                        }
                    }
                    ScalarOp::Mul(c) => {
                        for v in &mut acc {
                            *v *= c;
                        }
                    }
                    ScalarOp::Relu => {
                        for v in &mut acc {
                            *v = v.max(0.0);
                        }
                    }
                    ScalarOp::Clamp(min, max) => {
                        for v in &mut acc {
                            *v = v.clamp(*min, *max);
                        }
                    }
                }
            }

            // Write results
            output.extend_from_slice(&acc);
        }

        // Handle remainder elements
        let remainder_start = chunks * CHUNK_SIZE;
        for i in 0..remainder {
            let mut acc = src[remainder_start + i];
            for op in &kernel.ops {
                match op {
                    ScalarOp::Add(c) => acc += c,
                    ScalarOp::Mul(c) => acc *= c,
                    ScalarOp::Relu => acc = acc.max(0.0),
                    ScalarOp::Clamp(min, max) => acc = acc.clamp(*min, *max),
                }
            }
            output.push(acc);
        }

        // Convert f32 vec to bytes
        let byte_data = unsafe {
            let mut output = std::mem::ManuallyDrop::new(output);
            let ptr = output.as_mut_ptr() as *mut u8;
            let len = output.len() * 4;
            let cap = output.capacity() * 4;
            Vec::from_raw_parts(ptr, len, cap)
        };

        let new_layout = Layout::new_contiguous(self.layout.shape.clone(), DType::F32);
        Self {
            data: BufferStorage::Rust(Arc::new(byte_data)),
            layout: new_layout,
        }
    }

    /// Strided fused kernel for non-contiguous buffers.
    fn apply_fused_kernel_strided(&self, kernel: &FusedKernel, total_elems: usize) -> ViewBuffer {
        let mut new_data = Vec::with_capacity(total_elems * 4); // F32 = 4 bytes

        let mut indices = vec![0; self.layout.shape.len()];
        let shape = &self.layout.shape;
        let strides = &self.layout.strides;
        let ptr = self.data.as_ptr();
        let base_offset = self.layout.offset;
        let data_len = self.data.len();

        for _ in 0..total_elems {
            let mut offset = base_offset as isize;
            for (dim, &idx) in indices.iter().enumerate() {
                offset += (idx as isize) * strides[dim];
            }

            debug_assert!(
                offset >= 0 && (offset as usize) + 4 <= data_len,
                "Fused kernel read OOB"
            );

            unsafe {
                let src_ptr = ptr.offset(offset) as *const f32;
                let mut acc = *src_ptr;

                for op in &kernel.ops {
                    match op {
                        ScalarOp::Add(c) => acc += c,
                        ScalarOp::Mul(c) => acc *= c,
                        ScalarOp::Relu => acc = acc.max(0.0),
                        ScalarOp::Clamp(min, max) => acc = acc.clamp(*min, *max),
                    }
                }

                let val_bytes = acc.to_ne_bytes();
                new_data.extend_from_slice(&val_bytes);
            }

            for dim in (0..shape.len()).rev() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }

        let new_layout = Layout::new_contiguous(self.layout.shape.clone(), DType::F32);
        Self {
            data: BufferStorage::Rust(Arc::new(new_data)),
            layout: new_layout,
        }
    }

    /// Casts the buffer to a different data type.
    pub fn cast(&self, target: DType) -> Self {
        if self.dtype() == target {
            return self.clone();
        }
        let contig = self.to_contiguous();

        match (contig.dtype(), target) {
            (DType::U8, DType::F32) => contig.cast_impl::<u8, f32>(),
            (DType::F32, DType::U8) => contig.cast_impl::<f32, u8>(),
            (DType::I32, DType::F32) => contig.cast_impl::<i32, f32>(),
            (DType::F32, DType::I32) => contig.cast_impl::<f32, i32>(),
            _ => unimplemented!(
                "Cast pair {:?} -> {:?} not implemented",
                self.dtype(),
                target
            ),
        }
    }

    fn cast_impl<S, D>(&self) -> Self
    where
        S: ViewType + AsPrimitive<D>,
        D: ViewType + Copy + 'static,
    {
        let elem_count = self.layout.shape.iter().product();
        let src_slice = unsafe { std::slice::from_raw_parts(self.as_ptr::<S>(), elem_count) };

        let new_data: Vec<D> = src_slice.iter().map(|&x| x.as_()).collect();
        Self::from_vec(new_data).reshape(self.layout.shape.clone())
    }

    /// Reshapes the buffer to a new shape.
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        self.layout.shape = shape;
        self.layout = Layout::new_contiguous(self.layout.shape, self.layout.dtype);
        self
    }
}
