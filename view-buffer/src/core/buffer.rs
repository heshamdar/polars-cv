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
//! - **Rust storage** (`Arc<Vec<u8>>`): Zero-copy requires sole ownership (refcount == 1)
//! - **PolarsArrow storage**: Always zero-copy via `Buffer::sliced()`
//! - **Arrow storage**: Not currently supported for zero-copy transfer

use std::sync::Arc;

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
    /// let buf = ViewBuffer::from_slice_aligned(&data, 64);
    /// assert_eq!(buf.shape(), &[4]);
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

        // Copy to a standard Vec for safe deallocation. Vec<u8> uses align=1 for
        // dealloc, which differs from our custom alignment. Using from_raw_parts
        // directly would be UB. The copy is the cost of correctness; for truly
        // zero-copy aligned storage, a custom allocator would be needed.
        let aligned_vec = unsafe {
            let slice = std::slice::from_raw_parts(aligned_ptr, len_bytes);
            let v = slice.to_vec();
            std::alloc::dealloc(aligned_ptr, alloc_layout);
            v
        };

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

        // Macro to handle all dtype combinations.
        //
        // Two variants by source kind:
        // - `int`:   integer source -> plain `as` (no fractional part to round).
        // - `float`: float source -> **round to nearest** before narrowing to an
        //            integer target, so e.g. 140.75_f32 -> 141_u8 rather than 140.
        //            Rust's float->int `as` already saturates out-of-range values
        //            and maps NaN -> 0, so clamping behaviour is preserved.
        //            Float->float targets keep plain `as` (no rounding).
        macro_rules! cast_impl {
            (int $src:ty, $dst_dtype:expr) => {{
                let src_data = contig.as_slice::<$src>();
                match $dst_dtype {
                    DType::U8 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                        shape,
                    ),
                    DType::I8 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as i8).collect::<Vec<i8>>(),
                        shape,
                    ),
                    DType::U16 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as u16).collect::<Vec<u16>>(),
                        shape,
                    ),
                    DType::I16 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as i16).collect::<Vec<i16>>(),
                        shape,
                    ),
                    DType::U32 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
                        shape,
                    ),
                    DType::I32 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
                        shape,
                    ),
                    DType::U64 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as u64).collect::<Vec<u64>>(),
                        shape,
                    ),
                    DType::I64 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as i64).collect::<Vec<i64>>(),
                        shape,
                    ),
                    DType::F32 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                        shape,
                    ),
                    DType::F64 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
                        shape,
                    ),
                }
            }};
            (float $src:ty, $dst_dtype:expr) => {{
                let src_data = contig.as_slice::<$src>();
                match $dst_dtype {
                    DType::U8 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as u8)
                            .collect::<Vec<u8>>(),
                        shape,
                    ),
                    DType::I8 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as i8)
                            .collect::<Vec<i8>>(),
                        shape,
                    ),
                    DType::U16 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as u16)
                            .collect::<Vec<u16>>(),
                        shape,
                    ),
                    DType::I16 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as i16)
                            .collect::<Vec<i16>>(),
                        shape,
                    ),
                    DType::U32 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as u32)
                            .collect::<Vec<u32>>(),
                        shape,
                    ),
                    DType::I32 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as i32)
                            .collect::<Vec<i32>>(),
                        shape,
                    ),
                    DType::U64 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as u64)
                            .collect::<Vec<u64>>(),
                        shape,
                    ),
                    DType::I64 => Self::from_vec_with_shape(
                        src_data
                            .iter()
                            .map(|&x| x.round() as i64)
                            .collect::<Vec<i64>>(),
                        shape,
                    ),
                    // Float -> float: no rounding.
                    DType::F32 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                        shape,
                    ),
                    DType::F64 => Self::from_vec_with_shape(
                        src_data.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
                        shape,
                    ),
                }
            }};
        }

        match self.layout.dtype {
            DType::U8 => cast_impl!(int u8, target_dtype),
            DType::I8 => cast_impl!(int i8, target_dtype),
            DType::U16 => cast_impl!(int u16, target_dtype),
            DType::I16 => cast_impl!(int i16, target_dtype),
            DType::U32 => cast_impl!(int u32, target_dtype),
            DType::I32 => cast_impl!(int i32, target_dtype),
            DType::U64 => cast_impl!(int u64, target_dtype),
            DType::I64 => cast_impl!(int i64, target_dtype),
            DType::F32 => cast_impl!(float f32, target_dtype),
            DType::F64 => cast_impl!(float f64, target_dtype),
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
        buffer: polars_arrow::buffer::Buffer<u8>,
        offset: usize,
        len: usize,
        shape: Vec<usize>,
        strides: Vec<isize>,
        dtype: DType,
    ) -> Self {
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

    /// Returns the number of bytes occupied by the logical elements of this view.
    ///
    /// For non-contiguous views this is the number of *elements* × element size,
    /// not the underlying storage size.  Used by the execution strategy to
    /// decide whether tiling is worthwhile.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.layout.num_elements() * self.layout.dtype.size_of()
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
                // Try to unwrap the Arc - only succeeds if refcount == 1
                Arc::try_unwrap(arc).ok()
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
    pub fn into_polars_buffer(self) -> (polars_arrow::buffer::Buffer<u8>, Vec<usize>, DType) {
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
    ) -> (polars_arrow::buffer::Buffer<u8>, Vec<usize>, DType) {
        let shape = self.layout.shape.clone();
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
                let sliced = polars_buf.sliced(combined_offset, required_bytes);
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
                        Ok(full_vec) => {
                            let full_buffer = polars_arrow::buffer::Buffer::from(full_vec);
                            if offset == 0 && required_bytes == full_buffer.len() {
                                (full_buffer, shape, dtype)
                            } else {
                                // Zero-copy slice using Buffer::sliced()
                                (full_buffer.sliced(offset, required_bytes), shape, dtype)
                            }
                        }
                        Err(arc) => {
                            // Arc unwrap failed (shouldn't happen) - copy the slice
                            let slice = &arc[offset..offset + required_bytes];
                            let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());
                            (buffer, shape, dtype)
                        }
                    }
                } else {
                    // Policy says copy - extract just the slice
                    let slice = &arc[offset..offset + required_bytes];
                    let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());
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
                let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());
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
        polars_arrow::buffer::Buffer<u8>,
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
        polars_arrow::buffer::Buffer<u8>,
        Vec<usize>,
        Vec<isize>,
        usize,
        DType,
    ) {
        let shape = self.layout.shape.clone();
        let strides = self.layout.strides.clone();
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
                    let contig_shape = contig.layout.shape.clone();
                    let contig_strides = contig.layout.strides.clone();
                    let data_len = contig.layout.num_elements() * contig.layout.dtype.size_of();
                    let slice =
                        unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                    let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());
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
                        Ok(full_vec) => {
                            let full_buffer = polars_arrow::buffer::Buffer::from(full_vec);
                            (full_buffer, shape, strides, offset, dtype)
                        }
                        Err(arc) => {
                            // Arc unwrap failed - copy to contiguous
                            let contig = ViewBuffer {
                                data: BufferStorage::Rust(arc),
                                layout: self.layout,
                            }
                            .to_contiguous();
                            let contig_shape = contig.layout.shape.clone();
                            let contig_strides = contig.layout.strides.clone();
                            let data_len =
                                contig.layout.num_elements() * contig.layout.dtype.size_of();
                            let slice = unsafe {
                                std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len)
                            };
                            let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());
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
                    let contig_shape = contig.layout.shape.clone();
                    let contig_strides = contig.layout.strides.clone();
                    let data_len = contig.layout.num_elements() * contig.layout.dtype.size_of();
                    let slice =
                        unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                    let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());
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
                let contig_shape = contig.layout.shape.clone();
                let contig_strides = contig.layout.strides.clone();
                let data_len = contig.layout.num_elements() * contig.layout.dtype.size_of();
                let slice = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), data_len) };
                let buffer = polars_arrow::buffer::Buffer::from(slice.to_vec());
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

        let data_len = buffer.data.len();
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
    /// Reads the full VIEW protocol header including shape and strides.
    /// If the blob was serialized with a non-contiguous layout, the stored
    /// strides are preserved. Performs a copy of the data payload into a
    /// new `Vec<u8>`.
    pub fn from_blob(data: &[u8]) -> Result<ViewBuffer, BufferError> {
        if data.len() < HEADER_SIZE {
            return Err(BufferError::InvalidProtocol(
                "Data too short for header".into(),
            ));
        }

        // 1. Read Header
        // Read header using ptr::read_unaligned to avoid UB from unaligned access.
        // The input &[u8] may not be aligned to ViewHeader's required alignment.
        let header = unsafe { std::ptr::read_unaligned(data.as_ptr() as *const ViewHeader) };

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
        let flags = header.flags;
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

        // 4. Construct ViewBuffer — use stored strides if non-contiguous
        let layout = if flags == 1 || strides.is_empty() {
            Layout::new_contiguous(shape, dtype)
        } else {
            Layout {
                shape,
                strides,
                offset: 0,
                dtype,
            }
        };

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
    /// When the innermost dimension has contiguous stride, copies entire rows
    /// at once (memcpy) instead of element-by-element for much better performance.
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

        let shape = &self.layout.shape;
        let strides = &self.layout.strides;
        let ndim = shape.len();

        // Optimization: check if innermost dimension is contiguous (stride == dtype_size).
        // If so, we can copy entire rows at once using memcpy instead of element-by-element.
        let inner_contiguous = ndim > 0 && strides[ndim - 1] == dtype_size as isize;

        if inner_contiguous && ndim >= 2 {
            // Fast path: copy row-by-row
            let row_len = shape[ndim - 1];
            let row_bytes = row_len * dtype_size;
            // Number of rows = product of all outer dimensions
            let num_rows: usize = shape[..ndim - 1].iter().product();

            let mut new_data = Vec::with_capacity(total_bytes);
            let ptr = self.data.as_ptr();
            let base_offset = self.layout.offset;
            let data_len = self.data.len();

            let mut indices = vec![0usize; ndim - 1]; // Only outer dims

            for _ in 0..num_rows {
                // Compute offset for this row's first element
                let mut offset = base_offset as isize;
                for (dim, &idx) in indices.iter().enumerate() {
                    offset += (idx as isize) * strides[dim];
                }

                debug_assert!(offset >= 0, "Negative offset in row-copy path");
                debug_assert!(
                    (offset as usize) + row_bytes <= data_len,
                    "Row read overrun: offset={offset}, row_bytes={row_bytes}, data_len={data_len}"
                );

                unsafe {
                    let src = ptr.offset(offset);
                    new_data.extend_from_slice(std::slice::from_raw_parts(src, row_bytes));
                }

                // Increment outer indices
                for dim in (0..indices.len()).rev() {
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
        } else {
            // Slow path: element-by-element copy for non-contiguous innermost dim
            let mut new_data = Vec::with_capacity(total_bytes);
            let mut indices = vec![0; ndim];
            let ptr = self.data.as_ptr();
            let base_offset = self.layout.offset;
            let data_len = self.data.len();

            for _ in 0..total_elems {
                let mut offset = base_offset as isize;
                for (dim, &idx) in indices.iter().enumerate() {
                    offset += (idx as isize) * strides[dim];
                }

                debug_assert!(offset >= 0, "Negative offset calculation");
                debug_assert!(
                    (offset as usize) + dtype_size <= data_len,
                    "Read overrun during compaction: offset={offset}, dtype_size={dtype_size}, data_len={data_len}"
                );

                unsafe {
                    let src = ptr.offset(offset);
                    new_data.extend_from_slice(std::slice::from_raw_parts(src, dtype_size));
                }

                for dim in (0..ndim).rev() {
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
    }

    /// Applies a fused kernel of scalar operations in-place, without allocation.
    ///
    /// Succeeds only when no dtype conversion is involved on either end
    /// (F32 buffer, F32 kernel output), the buffer is contiguous, and this
    /// `Arc` has exactly one strong reference (i.e. the caller holds exclusive
    /// ownership). In that case the inner `Vec<u8>` is mutated directly —
    /// zero heap allocation.
    ///
    /// Returns `true` if the in-place path was taken, `false` if the caller should
    /// fall back to the allocating [`apply_fused_kernel`] path.
    pub fn try_apply_fused_kernel_inplace(&mut self, kernel: &FusedKernel) -> bool {
        if self.dtype() != DType::F32
            || kernel.out_dtype != DType::F32
            || !self.layout.is_contiguous()
        {
            return false;
        }
        let BufferStorage::Rust(ref mut arc) = self.data else {
            return false;
        };
        let Some(vec) = Arc::get_mut(arc) else {
            return false;
        };

        let total_elems: usize = self.layout.shape.iter().product();
        let data = unsafe {
            std::slice::from_raw_parts_mut(
                vec.as_mut_ptr().add(self.layout.offset) as *mut f32,
                total_elems,
            )
        };

        // One full-array pass per op — the inner loop is a simple scalar
        // operation that LLVM can auto-vectorize with SIMD (the closure is
        // known at compile time within each match arm, unlike the old
        // chunk-then-ops order which blocked auto-vectorization).
        apply_fused_op_passes(data, &kernel.ops);
        true
    }

    /// Applies a fused kernel of scalar operations element-wise.
    ///
    /// Accepts any numeric input dtype: the input is converted to `f32`
    /// during the gather (equivalent to a fused leading `Cast`), the ops run
    /// as SIMD-friendly full-array `f32` passes, and the result is converted
    /// to `kernel.out_dtype` while writing the output buffer (equivalent to a
    /// fused trailing `Cast`, matching [`ViewBuffer::cast_to`] semantics).
    /// Compared to bracketing the kernel with separate casts, this removes
    /// the intermediate materializations.
    pub fn apply_fused_kernel(&self, kernel: &FusedKernel) -> ViewBuffer {
        let total_elems: usize = self.layout.shape.iter().product();

        // Gather to f32 (handles dtype conversion and striding in one pass).
        let mut acc: Vec<f32> = self.gather_to_f32(total_elems);

        // One full-array pass per op (auto-vectorized; see inplace docs).
        apply_fused_op_passes(&mut acc, &kernel.ops);

        // Convert to the kernel's output dtype while writing the result.
        finish_fused_output(acc, self.layout.shape.clone(), kernel.out_dtype)
    }

    /// Read every element as `f32`, in logical (row-major) order.
    ///
    /// Contiguous buffers convert with a monomorphic per-dtype loop; strided
    /// buffers gather through the stride walk (also monomorphic per dtype).
    fn gather_to_f32(&self, total_elems: usize) -> Vec<f32> {
        if self.layout.is_contiguous() {
            macro_rules! convert_contig {
                ($t:ty) => {{
                    let src_ptr =
                        unsafe { self.data.as_ptr().add(self.layout.offset) as *const $t };
                    let src = unsafe { std::slice::from_raw_parts(src_ptr, total_elems) };
                    src.iter().map(|&x| x as f32).collect()
                }};
            }
            return match self.dtype() {
                DType::F32 => {
                    let src_ptr =
                        unsafe { self.data.as_ptr().add(self.layout.offset) as *const f32 };
                    let src = unsafe { std::slice::from_raw_parts(src_ptr, total_elems) };
                    src.to_vec()
                }
                DType::U8 => convert_contig!(u8),
                DType::I8 => convert_contig!(i8),
                DType::U16 => convert_contig!(u16),
                DType::I16 => convert_contig!(i16),
                DType::U32 => convert_contig!(u32),
                DType::I32 => convert_contig!(i32),
                DType::U64 => convert_contig!(u64),
                DType::I64 => convert_contig!(i64),
                DType::F64 => convert_contig!(f64),
            };
        }

        // Strided gather: walk logical indices, reading each element at its
        // byte offset. The walk is monomorphized per dtype so the inner read
        // has no per-element dispatch.
        macro_rules! gather_strided {
            ($t:ty) => {{
                let mut out: Vec<f32> = Vec::with_capacity(total_elems);
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
                        offset >= 0 && (offset as usize) + std::mem::size_of::<$t>() <= data_len,
                        "Fused kernel read OOB"
                    );
                    let value = unsafe { *(ptr.offset(offset) as *const $t) };
                    out.push(value as f32);
                    for dim in (0..shape.len()).rev() {
                        indices[dim] += 1;
                        if indices[dim] < shape[dim] {
                            break;
                        }
                        indices[dim] = 0;
                    }
                }
                out
            }};
        }
        match self.dtype() {
            DType::U8 => gather_strided!(u8),
            DType::I8 => gather_strided!(i8),
            DType::U16 => gather_strided!(u16),
            DType::I16 => gather_strided!(i16),
            DType::U32 => gather_strided!(u32),
            DType::I32 => gather_strided!(i32),
            DType::U64 => gather_strided!(u64),
            DType::I64 => gather_strided!(i64),
            DType::F32 => gather_strided!(f32),
            DType::F64 => gather_strided!(f64),
        }
    }

    /// Casts the buffer to a different data type.
    ///
    /// Supports all dtype pairs (u8, i8, u16, i16, u32, i32, u64, i64, f32, f64).
    /// This is an alias for [`cast_to`] for API compatibility.
    pub fn cast(&self, target: DType) -> Self {
        self.cast_to(target)
    }

    /// Reshapes the buffer to a new shape.
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        self.layout.shape = shape;
        self.layout = Layout::new_contiguous(self.layout.shape, self.layout.dtype);
        self
    }
}

/// Apply each fused scalar op as a full-array pass over `f32` data.
///
/// One pass per op (vs one pass total) costs slightly more bandwidth but lets
/// LLVM auto-vectorize each inner loop with AVX/AVX2/NEON — the inner loop is
/// a simple scalar operation with no enum dispatch. The bandwidth tradeoff
/// breaks even at ~2 ops for typical L2-resident sizes.
fn apply_fused_op_passes(data: &mut [f32], ops: &[ScalarOp]) {
    for op in ops {
        match op {
            ScalarOp::Add(c) => {
                for x in data.iter_mut() {
                    *x += *c;
                }
            }
            ScalarOp::Mul(c) => {
                for x in data.iter_mut() {
                    *x *= *c;
                }
            }
            ScalarOp::Div(c) => {
                for x in data.iter_mut() {
                    *x /= *c;
                }
            }
            ScalarOp::Pow(c) => {
                for x in data.iter_mut() {
                    *x = x.powf(*c);
                }
            }
            ScalarOp::Relu => {
                for x in data.iter_mut() {
                    *x = x.max(0.0);
                }
            }
            ScalarOp::Clamp(lo, hi) => {
                for x in data.iter_mut() {
                    *x = x.clamp(*lo, *hi);
                }
            }
        }
    }
}

/// Materialize the kernel's f32 result as a contiguous buffer of `out_dtype`.
///
/// Integer targets round to nearest then saturate (`x.round() as T`), exactly
/// mirroring [`ViewBuffer::cast_to`]'s float→int semantics; `F32` reuses the
/// accumulator allocation without copying.
fn finish_fused_output(acc: Vec<f32>, shape: Vec<usize>, out_dtype: DType) -> ViewBuffer {
    macro_rules! convert_out {
        (int $t:ty) => {
            ViewBuffer::from_vec_with_shape(
                acc.iter().map(|&x| x.round() as $t).collect::<Vec<$t>>(),
                shape,
            )
        };
    }
    match out_dtype {
        DType::F32 => {
            // Reuse the accumulator allocation: transmute Vec<f32> to bytes.
            let byte_data = unsafe {
                let mut acc = std::mem::ManuallyDrop::new(acc);
                let ptr = acc.as_mut_ptr() as *mut u8;
                let len = acc.len() * 4;
                let cap = acc.capacity() * 4;
                Vec::from_raw_parts(ptr, len, cap)
            };
            ViewBuffer {
                data: BufferStorage::Rust(Arc::new(byte_data)),
                layout: Layout::new_contiguous(shape, DType::F32),
            }
        }
        DType::F64 => ViewBuffer::from_vec_with_shape(
            acc.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
            shape,
        ),
        DType::U8 => convert_out!(int u8),
        DType::I8 => convert_out!(int i8),
        DType::U16 => convert_out!(int u16),
        DType::I16 => convert_out!(int i16),
        DType::U32 => convert_out!(int u32),
        DType::I32 => convert_out!(int i32),
        DType::U64 => convert_out!(int u64),
        DType::I64 => convert_out!(int i64),
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
