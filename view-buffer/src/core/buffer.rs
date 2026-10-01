//! Buffer storage and view types.
//!
//! This module provides the core [`ViewBuffer`] type for zero-copy tensor operations
//! and efficient interoperability with Polars.
//!
//! # Zero-Copy Transfer
//!
//! When transferring data back to Polars, the library supports zero-copy transfer
//! via [`ViewBuffer::into_polars_buffer`] and [`ViewBuffer::into_polars_buffer_with_policy`].
//!
//! ## Slice Policy
//!
//! When a buffer has a non-zero offset (e.g., from a slice or crop operation),
//! the [`SlicePolicy`] controls whether to:
//!
//! - **Zero-copy slice**: Use `Buffer::sliced()` to create a view (keeps full buffer alive)
//! - **Copy**: Copy just the slice (releases unused memory)
//!
//! This is a memory vs. performance trade-off:
//!
//! | Policy | Behavior | Memory | Performance |
//! |--------|----------|--------|-------------|
//! | `AlwaysZeroCopy` | Always use `Buffer::sliced()` | May waste memory | Fast |
//! | `AlwaysCopy` | Always copy sliced data | Efficient | Slower |
//! | `Heuristic(0.5)` | Zero-copy if slice >= 50% of buffer | Balanced | Balanced |
//!
//! ## Storage Types
//!
//! - **Rust storage** (`Arc<AlignedBytes>`): Zero-copy requires sole ownership (refcount == 1)
//! - **PolarsArrow storage**: Always zero-copy via `Buffer::sliced()`
//! - **Arrow storage**: Not currently supported for zero-copy transfer

use std::sync::Arc;

use thiserror::Error;

use crate::core::bytes::AlignedBytes;
use crate::core::convert::convert_view;
use crate::core::dtype::{with_dtype, DType, ViewType};
use crate::core::layout::{Dims, ExternalLayout, Layout, LayoutFacts, Strides};
use crate::ops::scalar::FusedKernel;
use crate::protocol::{dtype_to_u8, ViewHeader, HEADER_SIZE, MAGIC_BYTES, VERSION};

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

/// Policy for handling sliced buffers during zero-copy transfer.
///
/// When a `ViewBuffer` has a non-zero offset (e.g., from a slice or crop operation),
/// this policy determines whether to:
/// - Use zero-copy slicing (keeps the entire underlying buffer alive)
/// - Copy just the slice (releases unused memory)
///
/// # Memory Trade-offs
///
/// - **AlwaysZeroCopy**: Maximum performance, but may keep large buffers alive
///   even when only a small slice is needed.
/// - **AlwaysCopy**: Predictable memory usage, always releases unused portions,
///   but incurs copy overhead.
/// - **Heuristic**: Balanced approach - uses zero-copy for slices that are a
///   significant portion of the buffer, copies smaller slices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlicePolicy {
    /// Always use zero-copy, even for small slices.
    ///
    /// This maximizes performance but may waste memory if a small slice
    /// keeps a large underlying buffer alive.
    AlwaysZeroCopy,

    /// Always copy sliced data to release unused memory.
    ///
    /// This ensures predictable memory usage but incurs copy overhead
    /// for all sliced buffers.
    AlwaysCopy,

    /// Use heuristic: zero-copy if slice >= threshold of total buffer.
    ///
    /// The threshold is a ratio (0.0 to 1.0). For example, with threshold 0.5:
    /// - A 60% slice uses zero-copy (60% >= 50%)
    /// - A 30% slice is copied (30% < 50%)
    Heuristic {
        /// Minimum ratio of slice size to buffer size for zero-copy.
        /// Value should be between 0.0 and 1.0.
        threshold: f64,
    },
}

impl Default for SlicePolicy {
    fn default() -> Self {
        // Default to heuristic with 50% threshold - balanced approach
        SlicePolicy::Heuristic { threshold: 0.5 }
        // SlicePolicy::AlwaysZeroCopy
    }
}

/// Storage backend for ViewBuffer data.
#[derive(Debug, Clone)]
pub enum BufferStorage {
    /// Owned bytes wrapped in Arc for cheap cloning. The storage records
    /// the alignment of the original (typically typed-Vec) allocation so
    /// dropping deallocates with the layout it was allocated with.
    Rust(Arc<AlignedBytes>),
    /// Arrow buffer for zero-copy interop.
    #[cfg(feature = "arrow_interop")]
    Arrow(arrow::buffer::Buffer),
    /// Polars-arrow buffer for zero-copy Polars integration.
    /// The offset field allows referencing a slice within the buffer.
    #[cfg(feature = "polars_interop")]
    PolarsArrow {
        /// The underlying polars-arrow buffer (Arc-backed, cheap to clone).
        buffer: polars_buffer::Buffer<u8>,
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
///
/// The named argument for [`ViewBuffer::from_slice_aligned`] and
/// [`ViewBuffer::is_aligned`], which together are this crate's alignment API.
/// Zero-argument wrappers hardcoding it (`from_slice_simd_aligned`,
/// `is_simd_aligned`) had no caller and were removed; the parameterised pair
/// answers strictly more.
pub const SIMD_ALIGNMENT: usize = 64;

impl ViewBuffer {
    /// Creates a ViewBuffer from a Vec of typed elements.
    pub fn from_vec<T: ViewType>(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        let dtype = T::DTYPE;
        let layout = Layout::new_contiguous(shape, dtype);

        // AlignedBytes takes over the typed allocation and deallocates it
        // with T's alignment — reinterpreting it as a Vec<u8> (align 1)
        // would be dealloc-layout UB.
        Self {
            data: BufferStorage::Rust(Arc::new(AlignedBytes::from_typed_vec(data))),
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
    /// use view_buffer::{ViewBuffer, SIMD_ALIGNMENT};
    /// let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    /// let buf = ViewBuffer::from_slice_aligned(&data, SIMD_ALIGNMENT);
    /// assert_eq!(buf.shape(), &[4]);
    /// assert!(buf.is_aligned(SIMD_ALIGNMENT));
    /// ```
    pub fn from_slice_aligned<T: ViewType>(data: &[T], alignment: usize) -> Self {
        debug_assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        debug_assert!(
            alignment >= std::mem::align_of::<T>(),
            "Alignment must be >= type alignment"
        );

        // SAFETY: any T: ViewType is plain-old-data, so its bytes can be
        // viewed as a u8 slice.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        // AlignedBytes records the custom alignment and deallocates with
        // the same layout, so the aligned allocation is kept as-is — the
        // returned buffer really is `alignment`-aligned (the previous
        // implementation had to copy into a plain Vec and lost it).
        let aligned = AlignedBytes::copy_from_slice_aligned(bytes, alignment);

        let shape = vec![data.len()];
        let dtype = T::DTYPE;
        let layout = Layout::new_contiguous(shape, dtype);

        Self {
            data: BufferStorage::Rust(Arc::new(aligned)),
            layout,
        }
    }

    /// Creates a ViewBuffer from a Vec with a specific shape.
    ///
    /// # Arguments
    /// * `data` - Vector of elements.
    /// * `shape` - Shape of the resulting buffer.
    ///
    /// # Panics
    /// Panics if the data length doesn't match the shape product.
    pub fn from_vec_with_shape<T: ViewType>(data: Vec<T>, shape: impl Into<Dims>) -> Self {
        let shape: Dims = shape.into();
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

        Self {
            data: BufferStorage::Rust(Arc::new(AlignedBytes::from_typed_vec(data))),
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

        // The conversion rule (integer sources `as`; float → integer rounds
        // to nearest then saturates; float → float `as`) lives once, in
        // `convert::CastFrom`, and the bulk loop is its dispatched kernel,
        // reading a view's runs where they lie (no packed copy first).
        with_dtype!(self.dtype(), S => with_dtype!(target_dtype, D => {
            convert_view::<S, D>(self)
        }))
    }

    /// The elements as IEEE-754 half precision (binary16), stored as their
    /// bit patterns in a new packed `U16` buffer of this shape: the engine
    /// has no f16 dtype, and this is the half-precision tensor sink's
    /// conversion. Each element is read as f32 by the conversion rule (as
    /// [`cast_to`](Self::cast_to) `F32` does), then rounded to nearest-even,
    /// so NaN stays NaN, values beyond f16's range become infinite and tiny
    /// ones subnormal or zero. Any view is read where it lies.
    pub fn to_f16_bits(&self) -> Self {
        with_dtype!(self.dtype(), S => crate::core::convert::f16_bits::<S>(self))
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
        buffer: polars_buffer::Buffer<u8>,
        offset: usize,
        shape: impl Into<Dims>,
        dtype: DType,
    ) -> Self {
        let shape: Dims = shape.into();
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
        buffer: polars_buffer::Buffer<u8>,
        offset: usize,
        len: usize,
        shape: impl Into<Dims>,
        dtype: DType,
    ) -> Self {
        let shape: Dims = shape.into();
        let num_elements: usize = shape.iter().product();
        let expected_bytes = num_elements * dtype.size_of();

        assert!(
            offset + len <= buffer.len(),
            "Polars buffer slice out of bounds: offset={offset}, len={len}, buffer_len={}",
            buffer.len()
        );
        assert!(
            len == expected_bytes,
            "Byte length mismatch: provided={len}, expected={expected_bytes} (shape={shape:?}, dtype={dtype:?})"
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

    /// Creates a ViewBuffer from a Polars Arrow buffer with explicit strides.
    ///
    /// This is the strided variant of `from_polars_buffer_slice`, used when
    /// the blob format stores a non-contiguous layout.
    ///
    /// # Arguments
    /// * `buffer` - The underlying Polars Arrow buffer (reference-counted).
    /// * `offset` - Byte offset into the buffer where data starts.
    /// * `len` - Total byte length of the data region.
    /// * `shape` - Shape of the buffer.
    /// * `strides` - Byte strides for each dimension.
    /// * `dtype` - Element data type.
    #[cfg(feature = "polars_interop")]
    pub fn from_polars_buffer_slice_with_strides(
        buffer: polars_buffer::Buffer<u8>,
        offset: usize,
        len: usize,
        shape: impl Into<Dims>,
        strides: impl Into<Strides>,
        dtype: DType,
    ) -> Self {
        let shape: Dims = shape.into();
        let strides: Strides = strides.into();
        assert!(
            offset + len <= buffer.len(),
            "Polars buffer slice out of bounds: offset={offset}, len={len}, buffer_len={}",
            buffer.len()
        );
        assert_eq!(
            shape.len(),
            strides.len(),
            "Shape and strides must have same rank"
        );

        let layout = Layout {
            shape,
            strides,
            offset: 0, // offset is handled by BufferStorage
            dtype,
        };
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

        // Checked in every build, not only debug: callers turn this pointer
        // into a `&[T]`, and a misaligned slice is undefined behaviour rather
        // than a wrong value. A panic here is caught per row by the plugin and
        // reported as an engine bug; the constructors (e.g. `parse_blob`) are
        // what keep untrusted data from reaching it (CR-41).
        assert!(
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
        assert_eq!(
            T::DTYPE,
            self.layout.dtype,
            "Type mismatch: requested {:?} but buffer has {:?}",
            T::DTYPE,
            self.layout.dtype
        );

        let len: usize = self.layout.shape.iter().product();
        unsafe {
            let ptr = self.as_ptr::<T>();
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Read a single-element buffer's value as `f64`, whatever its dtype.
    ///
    /// The dtype-dispatched counterpart of `as_slice::<T>()[0]` for callers
    /// that treat one-element results as scalars (e.g. global reductions).
    /// Returns `None` when the buffer does not hold exactly one element.
    pub fn scalar_f64(&self) -> Option<f64> {
        if self.layout.shape.iter().product::<usize>() != 1 {
            return None;
        }
        let contig = self.to_contiguous();
        Some(match contig.dtype() {
            DType::U8 => contig.as_slice::<u8>()[0] as f64,
            DType::I8 => contig.as_slice::<i8>()[0] as f64,
            DType::U16 => contig.as_slice::<u16>()[0] as f64,
            DType::I16 => contig.as_slice::<i16>()[0] as f64,
            DType::U32 => contig.as_slice::<u32>()[0] as f64,
            DType::I32 => contig.as_slice::<i32>()[0] as f64,
            DType::U64 => contig.as_slice::<u64>()[0] as f64,
            DType::I64 => contig.as_slice::<i64>()[0] as f64,
            DType::F32 => contig.as_slice::<f32>()[0] as f64,
            DType::F64 => contig.as_slice::<f64>()[0],
        })
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
    /// # Note
    ///
    /// This method has strict requirements (offset == 0). For more flexible
    /// zero-copy transfer to Polars that supports sliced buffers, use
    /// [`into_polars_buffer_with_policy`] with an appropriate [`SlicePolicy`].
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
                // Try to unwrap the Arc - only succeeds if refcount == 1.
                // into_vec is zero-copy for byte-aligned allocations and
                // copies otherwise (a Vec<u8> may not own an allocation
                // with a different alignment).
                Arc::try_unwrap(arc).ok().map(AlignedBytes::into_vec)
            }
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => None,
            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { .. } => None,
        }
    }

    /// Convert to a polars-arrow Buffer, zero-copy when possible.
    ///
    /// This consumes the ViewBuffer and returns a polars Buffer suitable
    /// for constructing Polars Series/ChunkedArrays.
    ///
    /// Uses the default [`SlicePolicy`] (Heuristic with 50% threshold).
    /// For custom control over zero-copy behavior, use [`into_polars_buffer_with_policy`].
    ///
    /// # Returns
    /// A tuple of `(Buffer<u8>, shape, dtype)` for use in output encoding.
    #[cfg(feature = "polars_interop")]
    pub fn into_polars_buffer(self) -> (polars_buffer::Buffer<u8>, Vec<usize>, DType) {
        self.into_polars_buffer_with_policy(SlicePolicy::default())
    }

    /// Convert to a polars-arrow Buffer with configurable slice policy.
    ///
    /// This consumes the ViewBuffer and returns a polars Buffer suitable
    /// for constructing Polars Series/ChunkedArrays.
    ///
    /// # Zero-Copy Conditions
    ///
    /// Zero-copy transfer occurs when:
    /// - Buffer is contiguous (strides match C-order layout)
    /// - For `Rust` storage: sole owner (Arc refcount == 1) and policy allows it
    /// - For `PolarsArrow` storage: always zero-copy via buffer slicing
    ///
    /// # Slice Handling
    ///
    /// When a buffer has a non-zero offset (from slice/crop operations):
    /// - **AlwaysZeroCopy**: Uses `Buffer::sliced()` to create a view (keeps full buffer alive)
    /// - **AlwaysCopy**: Copies just the slice (releases unused memory)
    /// - **Heuristic**: Zero-copy if slice >= threshold of buffer, else copy
    ///
    /// # Arguments
    /// * `policy` - Controls how sliced buffers are handled
    ///
    /// # Returns
    /// A tuple of `(Buffer<u8>, shape, dtype)` for use in output encoding.
    #[cfg(feature = "polars_interop")]
    pub fn into_polars_buffer_with_policy(
        self,
        policy: SlicePolicy,
    ) -> (polars_buffer::Buffer<u8>, Vec<usize>, DType) {
        let shape = self.layout.shape.to_vec();
        let dtype = self.layout.dtype;
        let offset = self.layout.offset;
        let required_bytes: usize = shape.iter().product::<usize>() * dtype.size_of();
        let is_contiguous = self.layout.is_contiguous();

        match self.data {
            // Handle PolarsArrow storage - always zero-copy via slicing
            BufferStorage::PolarsArrow {
                buffer: polars_buf,
                offset: buf_offset,
                ..
            } if is_contiguous => {
                // True zero-copy: return the original buffer with adjusted slice
                let combined_offset = buf_offset + offset;
                let sliced = polars_buf.sliced(combined_offset..combined_offset + required_bytes);
                (sliced, shape, dtype)
            }

            // Handle Rust storage with policy-based zero-copy
            BufferStorage::Rust(arc) if is_contiguous => {
                let full_len = arc.len();
                let is_sole_owner = Arc::strong_count(&arc) == 1;

                // Determine if we should use zero-copy based on policy
                let should_zero_copy = is_sole_owner
                    && match policy {
                        SlicePolicy::AlwaysZeroCopy => true,
                        SlicePolicy::AlwaysCopy => offset == 0 && required_bytes == full_len,
                        SlicePolicy::Heuristic { threshold } => {
                            if offset == 0 && required_bytes == full_len {
                                true
                            } else {
                                let ratio = required_bytes as f64 / full_len as f64;
                                ratio >= threshold
                            }
                        }
                    };

                if should_zero_copy {
                    // Try to unwrap the Arc - should succeed since we checked refcount
                    match Arc::try_unwrap(arc) {
                        Ok(bytes) => {
                            // from_owner keeps AlignedBytes alive as the
                            // buffer's owner, so the allocation is freed
                            // with its original alignment — handing the
                            // raw Vec to polars would move the dealloc
                            // mismatch there.
                            let full_len = bytes.len();
                            let full_buffer = polars_buffer::Buffer::from_owner(bytes);
                            if offset == 0 && required_bytes == full_len {
                                (full_buffer, shape, dtype)
                            } else {
                                // Zero-copy slice using Buffer::sliced()
                                (
                                    full_buffer.sliced(offset..offset + required_bytes),
                                    shape,
                                    dtype,
                                )
                            }
                        }
                        Err(arc) => {
                            // Arc unwrap failed (shouldn't happen) - copy the slice
                            let slice = &arc[offset..offset + required_bytes];
                            let buffer = polars_buffer::Buffer::from(slice.to_vec());
                            (buffer, shape, dtype)
                        }
                    }
                } else {
                    // Policy says copy - extract just the slice
                    let slice = &arc[offset..offset + required_bytes];
                    let buffer = polars_buffer::Buffer::from(slice.to_vec());
                    (buffer, shape, dtype)
                }
            }

            // Non-contiguous or non-zero-copy storage - materialize via to_contiguous
            _ => {
                let contig = ViewBuffer {
                    data: self.data,
                    layout: self.layout,
                }
                .to_contiguous();
                let data_len = contig.layout.num_elements() * contig.layout.dtype.size_of();
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                let buffer = polars_buffer::Buffer::from(slice.to_vec());
                (buffer, shape, dtype)
            }
        }
    }

    /// Convert to polars buffer with strided layout preservation.
    ///
    /// Unlike [`into_polars_buffer`], this method supports non-contiguous strided buffers
    /// by returning the FULL underlying buffer along with strides and offset. This enables
    /// zero-copy strided views in Python/NumPy.
    ///
    /// # Returns
    /// A tuple of `(buffer, shape, strides, offset, dtype)` where:
    /// - `buffer`: The full underlying buffer (may be larger than the view)
    /// - `shape`: Array dimensions
    /// - `strides`: Byte strides per dimension (can be non-contiguous)
    /// - `offset`: Byte offset into buffer where the view starts
    /// - `dtype`: Data type
    ///
    /// # Zero-Copy Conditions
    ///
    /// Zero-copy occurs when:
    /// - For `PolarsArrow` storage: always (returns original buffer)
    /// - For `Rust` storage: sole owner (Arc refcount == 1)
    /// - For non-contiguous buffers: if policy allows keeping full buffer
    ///
    /// When zero-copy is not possible, the data is materialized to a contiguous buffer
    /// with standard strides.
    #[cfg(feature = "polars_interop")]
    pub fn into_polars_buffer_strided(
        self,
    ) -> (
        polars_buffer::Buffer<u8>,
        Vec<usize>,
        Vec<isize>,
        usize,
        DType,
    ) {
        self.into_polars_buffer_strided_with_policy(SlicePolicy::default())
    }

    /// Convert to polars buffer with strided layout preservation and configurable policy.
    ///
    /// # Arguments
    /// * `policy` - Controls whether to keep full buffer or copy for small views
    ///
    /// # Returns
    /// A tuple of `(buffer, shape, strides, offset, dtype)`.
    #[cfg(feature = "polars_interop")]
    pub fn into_polars_buffer_strided_with_policy(
        self,
        policy: SlicePolicy,
    ) -> (
        polars_buffer::Buffer<u8>,
        Vec<usize>,
        Vec<isize>,
        usize,
        DType,
    ) {
        let shape = self.layout.shape.to_vec();
        let strides = self.layout.strides.to_vec();
        let dtype = self.layout.dtype;
        let offset = self.layout.offset;
        let required_bytes = self.layout.num_elements() * dtype.size_of();

        match self.data {
            // Handle PolarsArrow storage - always zero-copy, preserve strides
            BufferStorage::PolarsArrow {
                buffer: polars_buf,
                offset: buf_offset,
                len: buf_len,
            } => {
                // Determine if we should keep full buffer based on policy
                let combined_offset = buf_offset + offset;
                let should_zero_copy = match policy {
                    SlicePolicy::AlwaysZeroCopy => true,
                    SlicePolicy::AlwaysCopy => false,
                    SlicePolicy::Heuristic { threshold } => {
                        let ratio = required_bytes as f64 / buf_len as f64;
                        ratio >= threshold
                    }
                };

                if should_zero_copy {
                    // Return the original buffer with stride info
                    (polars_buf, shape, strides, combined_offset, dtype)
                } else {
                    // Materialize to contiguous
                    let contig = ViewBuffer {
                        data: BufferStorage::PolarsArrow {
                            buffer: polars_buf,
                            offset: buf_offset,
                            len: buf_len,
                        },
                        layout: self.layout,
                    }
                    .to_contiguous();
                    let contig_shape = contig.layout.shape.to_vec();
                    let contig_strides = contig.layout.strides.to_vec();
                    let data_len = contig.layout.num_elements() * contig.layout.dtype.size_of();
                    let slice =
                        unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                    let buffer = polars_buffer::Buffer::from(slice.to_vec());
                    (buffer, contig_shape, contig_strides, 0, dtype)
                }
            }

            // Handle Rust storage
            BufferStorage::Rust(arc) => {
                let full_len = arc.len();
                let is_sole_owner = Arc::strong_count(&arc) == 1;

                // Determine if we should keep full buffer based on policy
                let should_zero_copy = is_sole_owner
                    && match policy {
                        SlicePolicy::AlwaysZeroCopy => true,
                        SlicePolicy::AlwaysCopy => false,
                        SlicePolicy::Heuristic { threshold } => {
                            let ratio = required_bytes as f64 / full_len as f64;
                            ratio >= threshold
                        }
                    };

                if should_zero_copy {
                    // Try to unwrap the Arc and return full buffer with stride info
                    match Arc::try_unwrap(arc) {
                        Ok(bytes) => {
                            // Keep AlignedBytes as the owner so the
                            // allocation is freed with its original
                            // alignment.
                            let full_buffer = polars_buffer::Buffer::from_owner(bytes);
                            (full_buffer, shape, strides, offset, dtype)
                        }
                        Err(arc) => {
                            // Arc unwrap failed - copy to contiguous
                            let contig = ViewBuffer {
                                data: BufferStorage::Rust(arc),
                                layout: self.layout,
                            }
                            .to_contiguous();
                            let contig_shape = contig.layout.shape.to_vec();
                            let contig_strides = contig.layout.strides.to_vec();
                            let data_len =
                                contig.layout.num_elements() * contig.layout.dtype.size_of();
                            let slice = unsafe {
                                std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len)
                            };
                            let buffer = polars_buffer::Buffer::from(slice.to_vec());
                            (buffer, contig_shape, contig_strides, 0, dtype)
                        }
                    }
                } else {
                    // Policy says copy - materialize to contiguous
                    let contig = ViewBuffer {
                        data: BufferStorage::Rust(arc),
                        layout: self.layout,
                    }
                    .to_contiguous();
                    let contig_shape = contig.layout.shape.to_vec();
                    let contig_strides = contig.layout.strides.to_vec();
                    let data_len = contig.layout.num_elements() * contig.layout.dtype.size_of();
                    let slice =
                        unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                    let buffer = polars_buffer::Buffer::from(slice.to_vec());
                    (buffer, contig_shape, contig_strides, 0, dtype)
                }
            }

            // Arrow storage - not supported for zero-copy, materialize to contiguous
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => {
                let contig = ViewBuffer {
                    data: self.data,
                    layout: self.layout,
                }
                .to_contiguous();
                let contig_shape = contig.layout.shape.to_vec();
                let contig_strides = contig.layout.strides.to_vec();
                let data_len = contig.layout.num_elements() * contig.layout.dtype.size_of();
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                let buffer = polars_buffer::Buffer::from(slice.to_vec());
                (buffer, contig_shape, contig_strides, 0, dtype)
            }
        }
    }

    /// Check if this buffer can be zero-copy transferred with strided output.
    ///
    /// Unlike [`can_zero_copy_transfer`], this also returns true for non-contiguous
    /// buffers that can preserve their strides for zero-copy strided output.
    ///
    /// Uses the default [`SlicePolicy`] (Heuristic with 50% threshold).
    #[cfg(feature = "polars_interop")]
    pub fn can_zero_copy_strided(&self) -> bool {
        self.can_zero_copy_strided_with_policy(SlicePolicy::default())
    }

    /// Check if this buffer can be zero-copy transferred with strided output and policy.
    #[cfg(feature = "polars_interop")]
    pub fn can_zero_copy_strided_with_policy(&self, policy: SlicePolicy) -> bool {
        let required_bytes = self.layout.num_elements() * self.layout.dtype.size_of();

        match &self.data {
            BufferStorage::Rust(arc) => {
                let is_sole_owner = Arc::strong_count(arc) == 1;
                if !is_sole_owner {
                    return false;
                }

                let full_len = arc.len();
                match policy {
                    SlicePolicy::AlwaysZeroCopy => true,
                    SlicePolicy::AlwaysCopy => false,
                    SlicePolicy::Heuristic { threshold } => {
                        let ratio = required_bytes as f64 / full_len as f64;
                        ratio >= threshold
                    }
                }
            }

            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { len, .. } => match policy {
                SlicePolicy::AlwaysZeroCopy => true,
                SlicePolicy::AlwaysCopy => false,
                SlicePolicy::Heuristic { threshold } => {
                    let ratio = required_bytes as f64 / *len as f64;
                    ratio >= threshold
                }
            },

            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => false,
        }
    }

    /// Check if this buffer can be zero-copy transferred.
    ///
    /// Uses the default [`SlicePolicy`] (Heuristic with 50% threshold).
    /// For custom policy, use [`can_zero_copy_transfer_with_policy`].
    ///
    /// Useful for testing and debugging zero-copy behavior.
    #[cfg(feature = "polars_interop")]
    pub fn can_zero_copy_transfer(&self) -> bool {
        self.can_zero_copy_transfer_with_policy(SlicePolicy::default())
    }

    /// Check if this buffer can be zero-copy transferred with a specific policy.
    ///
    /// Returns `true` if `into_polars_buffer_with_policy()` would achieve zero-copy.
    ///
    /// # Zero-Copy Conditions
    ///
    /// - For `Rust` storage: sole owner, contiguous, and policy allows the slice ratio
    /// - For `PolarsArrow` storage: contiguous (always zero-copy via slicing)
    /// - For `Arrow` storage: always false (not supported)
    ///
    /// # Arguments
    /// * `policy` - The slice policy to evaluate against
    #[cfg(feature = "polars_interop")]
    pub fn can_zero_copy_transfer_with_policy(&self, policy: SlicePolicy) -> bool {
        match &self.data {
            BufferStorage::Rust(arc) => {
                // Check basic requirements: sole owner and contiguous
                let is_valid = Arc::strong_count(arc) == 1 && self.layout.is_contiguous();
                if !is_valid {
                    return false;
                }

                let offset = self.layout.offset;
                let required_bytes = self.layout.num_elements() * self.layout.dtype.size_of();
                let full_len = arc.len();

                match policy {
                    SlicePolicy::AlwaysZeroCopy => true,
                    SlicePolicy::AlwaysCopy => offset == 0 && required_bytes == full_len,
                    SlicePolicy::Heuristic { threshold } => {
                        if offset == 0 && required_bytes == full_len {
                            true
                        } else {
                            let ratio = required_bytes as f64 / full_len as f64;
                            ratio >= threshold
                        }
                    }
                }
            }
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => false,
            BufferStorage::PolarsArrow { .. } => {
                // PolarsArrow is always zero-copy if contiguous (via Buffer::sliced)
                self.layout.is_contiguous()
            }
        }
    }

    /// Check if this buffer can be zero-copy transferred (non-polars version).
    ///
    /// Returns `true` if `try_into_owned_bytes()` would succeed.
    /// This is the legacy check that requires offset == 0.
    #[cfg(not(feature = "polars_interop"))]
    pub fn can_zero_copy_transfer(&self) -> bool {
        match &self.data {
            BufferStorage::Rust(arc) => {
                Arc::strong_count(arc) == 1
                    && self.layout.is_contiguous()
                    && self.layout.offset == 0
            }
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => false,
        }
    }

    /// Returns layout facts for this buffer.
    pub fn layout_facts(&self) -> LayoutFacts {
        LayoutFacts::from(&self.layout)
    }

    /// The view's rows as slices, when its elements are packed within each
    /// row ([`LayoutFacts::is_dense_rows`]): rank 2 with a unit column stride,
    /// or rank 3 channels-last with packed pixels. Each slice is one row's
    /// `W` (or `W * C`) elements.
    ///
    /// The row stride may be anything, negative included, so a crop or a
    /// vertical flip is read where it lies instead of being copied to a
    /// contiguous buffer first. `None` for any other layout (a transpose, a
    /// horizontal flip), which the caller must materialise.
    ///
    /// # Panics
    /// Panics if `T` is not this buffer's dtype.
    #[cfg(feature = "image_interop")]
    pub(crate) fn dense_rows<T: ViewType>(&self) -> Option<Vec<&[T]>> {
        assert_eq!(
            T::DTYPE,
            self.dtype(),
            "dense_rows: asked for {:?} rows of a {:?} buffer",
            T::DTYPE,
            self.dtype()
        );
        if !self.layout_facts().is_dense_rows() {
            return None;
        }
        let shape = &self.layout.shape;
        let row_len = shape[1] * shape.get(2).copied().unwrap_or(1);
        let row_stride = self.layout.strides[0];
        // SAFETY: the offset is inside the data (layouts are validated when
        // built), and every row `y < H` starts `y * row_stride` bytes from it
        // and spans `row_len` packed elements, which the layout keeps inside
        // the data. Offsets and strides are whole elements, so each row is
        // aligned for `T` (CR-41).
        let base = unsafe { self.data.as_ptr().add(self.layout.offset) };
        Some(
            (0..shape[0])
                .map(|y| unsafe {
                    std::slice::from_raw_parts(
                        base.offset(y as isize * row_stride).cast::<T>(),
                        row_len,
                    )
                })
                .collect(),
        )
    }

    /// Returns true if the buffer is compatible with the target external layout.
    pub fn is_compatible_with(&self, target: ExternalLayout) -> bool {
        self.layout_facts().compatible_with(target)
    }

    // --- Serialization (Protocol) ---

    /// Serializes the view to a binary blob (ViewBlob format).
    /// Always forces materialization to contiguous layout for transport efficiency.
    pub fn to_blob(&self) -> Vec<u8> {
        let mut blob = Vec::with_capacity(self.blob_len());
        self.write_blob_into(&mut blob);
        blob
    }

    /// The nominal byte length `write_blob_into` will append for this view
    /// (header + shape + strides + logical payload). Intended for buffer
    /// pre-allocation; `write_blob_into` reserves what it actually needs.
    pub fn blob_len(&self) -> usize {
        let rank = self.shape().len();
        let data_len: usize = self.layout.num_elements() * self.dtype().size_of();
        HEADER_SIZE + rank * 8 + rank * 8 + data_len
    }

    /// Appends the ViewBlob serialization of this view to `out`.
    ///
    /// Byte-identical to `to_blob` (which is implemented on top of this);
    /// callers that reuse a scratch buffer avoid the per-call allocation.
    pub fn write_blob_into(&self, out: &mut Vec<u8>) {
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

        // The view's elements: a contiguous view may cover only part of its
        // storage (leading or middle rows), so the storage length is not it.
        let data_len = buffer.logical_len_bytes();
        out.reserve((data_offset as usize) + data_len);

        // 3. Write Parts
        // Header
        // Use unsafe copy to bytes for the #[repr(C)] struct
        let header_slice = unsafe {
            std::slice::from_raw_parts(&header as *const ViewHeader as *const u8, HEADER_SIZE)
        };
        out.extend_from_slice(header_slice);

        // Shape (u64)
        for &dim in shape {
            out.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        // Strides (i64)
        for &stride in strides {
            out.extend_from_slice(&(stride as i64).to_le_bytes());
        }

        // Data
        // Since we called to_contiguous, the data is just the raw buffer content.
        // We use the pointer to copy the bytes.
        let raw_ptr = unsafe { buffer.as_ptr::<u8>() };
        let raw_slice = unsafe { std::slice::from_raw_parts(raw_ptr, data_len) };
        out.extend_from_slice(raw_slice);
    }

    /// Deserializes a ViewBuffer from a binary blob.
    ///
    /// The header is validated by [`crate::protocol::parse_blob`], the same
    /// parser the plugin's zero-copy decode uses. The payload window is copied
    /// into storage aligned to the element size, so the result owns its data
    /// whatever the alignment of `data` itself. A strided layout keeps its
    /// stored strides; its whole window is copied, since every element they
    /// address has been checked to lie inside it.
    pub fn from_blob(data: &[u8]) -> Result<ViewBuffer, BufferError> {
        let blob = crate::protocol::parse_blob(data).map_err(BufferError::InvalidProtocol)?;
        let window = &data[blob.data_offset..blob.data_offset + blob.data_len];
        let bytes = AlignedBytes::copy_from_slice_aligned(window, blob.dtype.size_of());
        let layout = match blob.strides {
            None => Layout::new_contiguous(blob.shape, blob.dtype),
            Some(strides) => Layout {
                shape: blob.shape,
                strides,
                offset: 0,
                dtype: blob.dtype,
            },
        };
        Ok(ViewBuffer {
            data: BufferStorage::Rust(Arc::new(bytes)),
            layout,
        })
    }

    // --- Views ---

    /// Permutes the dimensions of the buffer.
    pub fn permute(&self, dims: &[usize]) -> Self {
        let mut new_shape = Dims::from_elem(0, self.layout.shape.len());
        let mut new_strides = Strides::from_elem(0, self.layout.strides.len());

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
        let mut new_shape = Dims::new();

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
    /// The copy coalesces the layout into the longest packed runs it has (one
    /// `memcpy` for a contiguous view, one per row for a crop or a vertical
    /// flip, constant-size pixel copies for a horizontal flip or transpose).
    ///
    /// # Panics
    /// Panics if the total allocation size would overflow `usize`.
    pub fn to_contiguous(&self) -> Self {
        if self.layout.is_contiguous() {
            return self.clone();
        }
        let total_bytes = self.logical_len_bytes();
        let mut new_data: Vec<u8> = Vec::with_capacity(total_bytes);
        // SAFETY: `new_data` has room for `total_bytes`, which
        // `copy_elements_into` writes in full before the length is set.
        unsafe {
            self.copy_elements_into(new_data.as_mut_ptr());
            new_data.set_len(total_bytes);
        }
        let new_layout = Layout::new_contiguous(self.layout.shape.clone(), self.dtype());
        Self {
            data: BufferStorage::Rust(Arc::new(AlignedBytes::from(new_data))),
            layout: new_layout,
        }
    }

    /// Append this buffer's elements, in row-major order, to `out`.
    ///
    /// Each element is copied once, straight from wherever the view's strides
    /// put it, so a strided view is not first materialised by
    /// [`to_contiguous`](Self::to_contiguous). This is how a buffer joins a
    /// flat values array (a `list`/`array` column's) without an intermediate.
    ///
    /// # Panics
    /// Panics if `T` is not this buffer's dtype.
    pub fn append_to<T: ViewType>(&self, out: &mut Vec<T>) {
        assert_eq!(
            T::DTYPE,
            self.dtype(),
            "append_to: the vector holds {:?} but the buffer's dtype is {:?}",
            T::DTYPE,
            self.dtype()
        );
        let count: usize = self.layout.shape.iter().product();
        out.reserve(count);
        let len = out.len();
        // SAFETY: `reserve` made room for `count` more `T`s past `len`, and
        // `copy_elements_into` writes exactly `count * size_of::<T>()` bytes
        // of `T`s (the dtype check above) there before the length grows.
        unsafe {
            self.copy_elements_into(out.as_mut_ptr().add(len).cast::<u8>());
            out.set_len(len + count);
        }
    }

    /// Write this buffer's elements, in row-major order, into `out`, which
    /// must hold exactly as many elements — one copy through the view's
    /// strides, as [`append_to`](Self::append_to) makes, into a slice the
    /// caller owns (a row's place in a column's values).
    ///
    /// # Panics
    /// Panics if `T` is not this buffer's dtype or `out` has another length.
    pub fn write_to<T: ViewType>(&self, out: &mut [std::mem::MaybeUninit<T>]) {
        assert_eq!(
            T::DTYPE,
            self.dtype(),
            "write_to: the slice holds {:?} but the buffer's dtype is {:?}",
            T::DTYPE,
            self.dtype()
        );
        let count: usize = self.layout.shape.iter().product();
        assert_eq!(out.len(), count, "write_to: the slice has the wrong length");
        // SAFETY: `out` holds exactly `count` `T`s (checked), which is
        // `logical_len_bytes()` bytes of this dtype (checked), and cannot
        // overlap this buffer's data, which is only borrowed immutably.
        unsafe { self.copy_elements_into(out.as_mut_ptr().cast::<u8>()) };
    }

    /// The bytes this view's elements occupy when packed row-major.
    fn logical_len_bytes(&self) -> usize {
        // Checked, so an overflowing shape fails with a clear message.
        self.layout
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .and_then(|elems| elems.checked_mul(self.dtype().size_of()))
            .expect("allocation size overflow: buffer is too large to materialize")
    }

    /// Copy this view's elements, row-major, to `dst`, through the one walk
    /// over a view's memory ([`Walk`](crate::core::strided::Walk)).
    ///
    /// # Safety
    /// `dst` must be valid for writes of [`logical_len_bytes`](Self::logical_len_bytes)
    /// bytes and must not overlap this buffer's data.
    unsafe fn copy_elements_into(&self, dst: *mut u8) {
        // SAFETY: the caller's contract is the walk's.
        unsafe { crate::core::strided::Walk::of(self).copy_to(dst) };
    }

    /// The elements as a mutable slice, when this buffer may be written in
    /// place: contiguous, of dtype `T`, in its own Rust allocation, and the
    /// allocation's sole owner (no other `ViewBuffer`, and no Arrow or Polars
    /// column, can see the write).
    ///
    /// **The one sole-owner check**: every kernel that writes its input in
    /// place asks here, rather than re-deriving ownership.
    ///
    /// # Panics
    /// Panics if `T` is not this buffer's dtype.
    pub(crate) fn unique_contiguous_mut<T: ViewType>(&mut self) -> Option<&mut [T]> {
        assert_eq!(
            T::DTYPE,
            self.dtype(),
            "unique_contiguous_mut: asked for {:?} elements of a {:?} buffer",
            T::DTYPE,
            self.dtype()
        );
        if !self.layout.is_contiguous() {
            return None;
        }
        // Only a Rust allocation can be written; Arrow and polars memory
        // is shared.
        let bytes = match self.data {
            BufferStorage::Rust(ref mut arc) => Arc::get_mut(arc)?,
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(_) => return None,
            #[cfg(feature = "polars_interop")]
            BufferStorage::PolarsArrow { .. } => return None,
        };
        let count: usize = self.layout.shape.iter().product();
        // SAFETY: the view is contiguous, so its `count` elements are packed
        // from `offset` inside the allocation this buffer solely owns;
        // offsets are whole, aligned elements (CR-41).
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                bytes.as_mut_ptr().add(self.layout.offset).cast::<T>(),
                count,
            )
        })
    }

    /// Applies a fused kernel of scalar operations element-wise.
    ///
    /// Accepts any numeric input dtype: each element is read as `f32`, the
    /// ops run as `f32` passes, and the result is converted to
    /// `kernel.out_dtype` (the rule `cast` uses). Runs through the
    /// element-wise engine (`ops::elementwise`); `&self` is never written.
    pub fn apply_fused_kernel(&self, kernel: &FusedKernel) -> ViewBuffer {
        crate::ops::elementwise::run_kernel(self.clone(), kernel)
    }

    /// Casts the buffer to a different data type.
    ///
    /// Supports all dtype pairs (u8, i8, u16, i16, u32, i32, u64, i64, f32, f64).
    /// This is an alias for [`cast_to`] for API compatibility.
    pub fn cast(&self, target: DType) -> Self {
        self.cast_to(target)
    }

    /// Reshapes the buffer to a new shape.
    pub fn reshape(mut self, shape: impl Into<Dims>) -> Self {
        let shape: Dims = shape.into();
        // A different element count would describe memory this buffer does
        // not own. `ViewOp::Reshape::validate` rejects it before execution;
        // this is the backstop for direct callers.
        assert_eq!(
            self.layout.shape.iter().product::<usize>(),
            shape.iter().product::<usize>(),
            "reshape from {:?} to {:?} changes the element count",
            self.layout.shape,
            shape
        );
        // A contiguous view may start past its buffer's first element (a crop
        // below the first row): the new layout keeps where it starts.
        let offset = self.layout.offset;
        self.layout = Layout::new_contiguous(shape, self.layout.dtype);
        self.layout.offset = offset;
        self
    }
}

#[cfg(test)]
mod reshape_offset_tests {
    use crate::{ViewBuffer, ViewDto, ViewExpr, ViewOp};

    /// A view that starts past its buffer's first element (a crop below the
    /// first row) and is still contiguous reshapes to *its* elements. The
    /// reshape rebuilt the layout at offset 0 and read from the uncropped
    /// start.
    #[test]
    fn reshape_keeps_a_views_offset() {
        let buf = ViewBuffer::from_vec_with_shape((0u8..6).collect::<Vec<u8>>(), vec![3, 2, 1]);
        let cropped = buf.slice(&[1, 0, 0], &[3, 2, 1]);
        assert!(cropped.layout.is_contiguous());
        let flat = cropped.reshape(vec![4]);
        assert_eq!(flat.to_contiguous().as_slice::<u8>(), &[2, 3, 4, 5]);
    }

    /// A reshape of a view whose elements are not in row-major order (after a
    /// flip or a transpose) is planned with a copy first, as NumPy's reshape
    /// copies: it used to be refused at execution.
    #[test]
    fn reshape_of_a_strided_view_copies_first() {
        let image: Vec<u8> = (0..12).collect();
        let source = || ViewBuffer::from_vec_with_shape(image.clone(), vec![4, 3, 1]);
        let reshape = ViewDto::View(ViewOp::Reshape { shape: vec![12, 1] });
        let flip = ViewDto::View(ViewOp::Flip { axes: vec![0] });
        let out = ViewExpr::new_source(source())
            .try_apply_op(flip)
            .unwrap()
            .try_apply_op(reshape.clone())
            .unwrap()
            .plan()
            .execute();
        let flipped: Vec<u8> = image.chunks(3).rev().flatten().copied().collect();
        assert_eq!(out.to_contiguous().as_slice::<u8>(), &flipped[..]);
        let transpose = ViewDto::View(ViewOp::Transpose {
            axes: vec![1, 0, 2],
        });
        let out = ViewExpr::new_source(source())
            .try_apply_op(transpose)
            .unwrap()
            .try_apply_op(reshape)
            .unwrap()
            .plan()
            .execute();
        let transposed: Vec<u8> = (0..3)
            .flat_map(|c| (0..4).map(move |r| (r * 3 + c) as u8))
            .collect();
        assert_eq!(out.to_contiguous().as_slice::<u8>(), &transposed[..]);
    }

    /// `channel_select` on a single-channel image drops the channel axis
    /// through the same reshape.
    #[test]
    fn channel_select_after_a_crop_reads_the_crop() {
        let buf = ViewBuffer::from_vec_with_shape((0u8..6).collect::<Vec<u8>>(), vec![3, 2, 1]);
        let crop = ViewDto::View(ViewOp::Crop {
            top: 1,
            left: 0,
            height: Some(2),
            width: Some(2),
        });
        let select = ViewDto::View(ViewOp::ChannelSelect { index: 0 });
        let out = ViewExpr::new_source(buf)
            .apply_op(crop)
            .apply_op(select)
            .plan()
            .execute();
        assert_eq!(out.to_contiguous().as_slice::<u8>(), &[2, 3, 4, 5]);
    }
}

#[cfg(test)]
mod cast_round_tests {
    use crate::{DType, ViewBuffer};

    #[test]
    fn float_to_int_rounds_to_nearest() {
        // f32 -> u8 must round to nearest, not truncate (regression guard for the
        // RGB<->YCbCr roundtrip which drifted under truncation).
        let buf = ViewBuffer::from_vec(vec![140.75f32, 0.6, 1.4, 254.6, 255.9]);
        let out = buf.cast(DType::U8);
        assert_eq!(out.as_slice::<u8>(), &[141, 1, 1, 255, 255]);
    }

    #[test]
    fn float_to_int_handles_negative_and_nan() {
        let buf = ViewBuffer::from_vec(vec![-0.6f32, -0.4, f32::NAN]);
        let out = buf.cast(DType::I8);
        // -0.6 -> -1, -0.4 -> 0, NaN -> 0 (saturating float->int cast)
        assert_eq!(out.as_slice::<i8>(), &[-1, 0, 0]);
    }

    #[test]
    fn float_to_float_does_not_round() {
        let buf = ViewBuffer::from_vec(vec![1.25f32, 2.75]);
        let out = buf.cast(DType::F64);
        assert_eq!(out.as_slice::<f64>(), &[1.25f64, 2.75]);
    }

    #[test]
    fn int_to_int_uses_plain_cast() {
        // Integer narrowing keeps plain `as` (wrapping) semantics — only
        // float sources gained rounding.
        let buf = ViewBuffer::from_vec(vec![10u16, 250, 300]);
        let out = buf.cast(DType::U8);
        assert_eq!(out.as_slice::<u8>(), &[10, 250, 44]);
    }
}

#[cfg(test)]
mod scalar_dual_path_tests {
    //! The scalar arithmetic is written twice — the f32 bulk kernel
    //! (`apply_fused_op_passes`) and `ScalarOp::apply_f64` (the f64 cold path).
    //! These tests pin the two paths to each other so an edit to one that is
    //! not mirrored in the other fails loudly.
    use crate::ops::scalar::{FusedKernel, ScalarOp};
    use crate::{DType, ViewBuffer};

    /// Every `ScalarOp` computes the same function through the f32 bulk kernel
    /// and through `apply_f64`. Referenced by name in `ScalarOp::apply_f64`'s
    /// docstring — keep the name in sync if this moves.
    #[test]
    fn scalar_f64_matches_f32_kernel() {
        let ops = [
            ScalarOp::Add(1.5),
            ScalarOp::Sub(1.5),
            ScalarOp::Mul(2.0),
            ScalarOp::Div(4.0),
            ScalarOp::Pow(2.0),
            ScalarOp::Neg,
            ScalarOp::Abs,
            ScalarOp::Sqrt,
            ScalarOp::Square,
            ScalarOp::Recip,
            ScalarOp::Min(0.5),
            ScalarOp::Max(0.5),
            ScalarOp::Sign,
            ScalarOp::Floor,
            ScalarOp::Ceil,
            ScalarOp::Round,
            ScalarOp::Trunc,
            ScalarOp::Relu,
            ScalarOp::Clamp(0.0, 1.0),
        ];
        let xs: [f32; 12] = [
            -2.5,
            -1.0,
            -0.4,
            0.0,
            0.4,
            1.0,
            2.5,
            4.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
        ];
        for op in ops {
            let buf = ViewBuffer::from_vec_with_shape(xs.to_vec(), vec![xs.len()]);
            let kernel = FusedKernel {
                ops: vec![op.clone()],
                out_dtype: DType::F32,
            };
            let out = buf.apply_fused_kernel(&kernel);
            let got = out.as_slice::<f32>();
            for (i, &x) in xs.iter().enumerate() {
                let via_f64 = op.apply_f64(x as f64) as f32;
                let via_f32 = got[i];
                let ok = if via_f64.is_nan() {
                    via_f32.is_nan()
                } else if via_f64.is_infinite() {
                    via_f32.is_infinite() && via_f32.signum() == via_f64.signum()
                } else {
                    (via_f32 - via_f64).abs() <= 1e-5 * (1.0 + via_f64.abs())
                };
                assert!(
                    ok,
                    "{op:?} on {x}: f32 kernel gave {via_f32}, apply_f64 gave {via_f64}"
                );
            }
        }
    }

    /// NaN in, NaN out, for every scalar op — the one-sided bounds included,
    /// as NumPy and PyTorch propagate it (`relu(NaN)` is NaN, not 0) and as
    /// `clamp` already did. Fused (f32 kernel) and unfused (`apply_f64`).
    #[test]
    fn every_scalar_op_propagates_nan() {
        let ops = [
            ScalarOp::Min(0.5),
            ScalarOp::Max(0.5),
            ScalarOp::Relu,
            ScalarOp::Clamp(0.0, 1.0),
            ScalarOp::Abs,
            ScalarOp::Sign,
            ScalarOp::Round,
        ];
        for op in ops {
            assert!(op.apply_f64(f64::NAN).is_nan(), "{op:?}: apply_f64(NaN)");
            let buf = ViewBuffer::from_vec_with_shape(vec![f32::NAN; 3], vec![3]);
            let kernel = FusedKernel {
                ops: vec![op.clone()],
                out_dtype: DType::F32,
            };
            let out = buf.apply_fused_kernel(&kernel);
            assert!(
                out.as_slice::<f32>().iter().all(|v| v.is_nan()),
                "{op:?}: f32 kernel turned NaN into {:?}",
                out.as_slice::<f32>()
            );
        }
    }

    /// `round` breaks ties to even (`round_ties_even`), matching Polars' and
    /// numpy's `round` — verified against `pl.Series.round` at 1.42. The test
    /// makes the tie-breaking load-bearing so a silent switch to away-from-zero
    /// (`f32::round`) is caught. Both arithmetic paths must agree.
    #[test]
    fn round_breaks_ties_to_even() {
        let xs: [f32; 7] = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5];
        let buf = ViewBuffer::from_vec_with_shape(xs.to_vec(), vec![xs.len()]);
        let kernel = FusedKernel {
            ops: vec![ScalarOp::Round],
            out_dtype: DType::F32,
        };
        let got = buf.apply_fused_kernel(&kernel);
        // Ties to even: 2.5 -> 2 (away-from-zero would give 3), 3.5 -> 4.
        // -0.0 compares equal to 0.0 under IEEE, so the first element holds.
        assert_eq!(
            got.as_slice::<f32>(),
            &[0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0]
        );
        for &x in &xs {
            assert_eq!(
                ScalarOp::Round.apply_f64(x as f64),
                (x as f64).round_ties_even()
            );
        }
    }
}
