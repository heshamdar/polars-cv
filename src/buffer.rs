use std::sync::Arc;

use num_traits::AsPrimitive;
use thiserror::Error;

use crate::dtype::{DType, ViewType};
use crate::layout::{ExternalLayout, Layout, LayoutFacts, LayoutReport};
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::protocol::{dtype_to_u8, u8_to_dtype, ViewHeader, HEADER_SIZE, MAGIC_BYTES, VERSION};

#[derive(Error, Debug)]
pub enum BufferError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
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
}

impl BufferStorage {
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            BufferStorage::Rust(v) => v.as_ptr(),
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(b) => b.as_ptr(),
        }
    }

    /// Returns the length of the underlying byte buffer.
    pub fn len(&self) -> usize {
        match self {
            BufferStorage::Rust(v) => v.len(),
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(b) => b.len(),
        }
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct ViewBuffer {
    pub(crate) data: BufferStorage,
    pub(crate) layout: Layout,
}

impl ViewBuffer {
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

    pub fn dtype(&self) -> DType {
        self.layout.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

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
            (ptr as usize) % std::mem::align_of::<T>() == 0,
            "ViewBuffer pointer is not aligned for type {}; address={:p}, align={}", 
            std::any::type_name::<T>(), 
            ptr, 
            std::mem::align_of::<T>()
        );
        
        ptr as *const T
    }

    pub fn as_raw_parts(&self) -> (*const u8, &[usize], &[isize], DType) {
        (
            unsafe { self.data.as_ptr().add(self.layout.offset) },
            &self.layout.shape,
            &self.layout.strides,
            self.layout.dtype
        )
    }

    /// Returns a unique identifier for the underlying storage.
    /// Used for zero-copy verification in tests.
    pub fn storage_id(&self) -> usize {
        match &self.data {
            BufferStorage::Rust(arc) => Arc::as_ptr(arc) as usize,
            #[cfg(feature = "arrow_interop")]
            BufferStorage::Arrow(buf) => buf.as_ptr() as usize,
        }
    }

    pub fn layout_facts(&self) -> LayoutFacts {
        LayoutFacts::from(&self.layout)
    }

    pub fn is_compatible_with(&self, target: ExternalLayout) -> bool {
        self.layout_facts().compatible_with(target)
    }

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
            std::slice::from_raw_parts(
                &header as *const ViewHeader as *const u8,
                HEADER_SIZE
            )
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
            return Err(BufferError::InvalidProtocol("Data too short for header".into()));
        }

        // 1. Read Header
        // Unsafe cast from bytes to struct (valid due to #[repr(C)] and POD nature)
        let header = unsafe { 
            &*(data.as_ptr() as *const ViewHeader) 
        };

        // Validate Magic
        if header.magic != MAGIC_BYTES {
            return Err(BufferError::InvalidProtocol("Invalid magic bytes".into()));
        }
        if header.version != VERSION {
            return Err(BufferError::InvalidProtocol(format!("Unsupported version: {}", header.version)));
        }

        let rank = header.rank as usize;
        let dtype = u8_to_dtype(header.dtype)
            .ok_or_else(|| BufferError::InvalidProtocol(format!("Unknown dtype code: {}", header.dtype)))?;
        let data_offset = header.data_offset as usize;

        // 2. Read Shape & Strides
        let shape_start = HEADER_SIZE;
        let stride_start = shape_start + (rank * 8);
        
        if data_offset > data.len() {
             return Err(BufferError::InvalidProtocol("Data offset out of bounds".into()));
        }

        let mut shape = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);

        let mut pos = shape_start;
        for _ in 0..rank {
            if pos + 8 > data.len() { return Err(BufferError::InvalidProtocol("Truncated shape data".into())); }
            let bytes: [u8; 8] = data[pos..pos+8].try_into().unwrap();
            shape.push(u64::from_le_bytes(bytes) as usize);
            pos += 8;
        }

        pos = stride_start;
        for _ in 0..rank {
            if pos + 8 > data.len() { return Err(BufferError::InvalidProtocol("Truncated stride data".into())); }
            let bytes: [u8; 8] = data[pos..pos+8].try_into().unwrap();
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
                 expected_bytes, raw_data.len()
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

    pub fn permute(&self, dims: &[usize]) -> Self {
        let mut new_shape = vec![0; self.layout.shape.len()];
        let mut new_strides = vec![0; self.layout.strides.len()];

        for (i, &p) in dims.iter().enumerate() {
            new_shape[i] = self.layout.shape[p];
            new_strides[i] = self.layout.strides[p];
        }

        Self {
            data: self.data.clone(),
            layout: crate::layout::Layout {
                shape: new_shape,
                strides: new_strides,
                offset: self.layout.offset,
                dtype: self.layout.dtype,
            },
        }
    }

    pub fn slice(&self, start: &[usize], end: &[usize]) -> Self {
        let mut new_offset = self.layout.offset as isize;
        let mut new_shape = Vec::new();
        
        for i in 0..self.layout.shape.len() {
            let s = start[i];
            let e = end[i];
            new_offset += (s as isize) * self.layout.strides[i];
            new_shape.push(e - s);
        }

        Self {
            data: self.data.clone(),
            layout: crate::layout::Layout {
                shape: new_shape,
                strides: self.layout.strides.clone(),
                offset: new_offset as usize,
                dtype: self.layout.dtype,
            },
        }
    }

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
            layout: crate::layout::Layout {
                shape: self.layout.shape.clone(),
                strides: new_strides,
                offset: new_offset as usize,
                dtype: self.layout.dtype,
            },
        }
    }

    // --- Compute / Materialization ---

    pub fn to_contiguous(&self) -> Self {
        if self.layout.is_contiguous() {
            return self.clone();
        }

        let total_elems = self.layout.shape.iter().product();
        let dtype_size = self.dtype().size_of();
        let mut new_data = Vec::with_capacity(total_elems * dtype_size);

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
            debug_assert!((offset as usize) < data_len, "Offset out of bounds: {} vs len {}", offset, data_len);

            unsafe {
                let src = ptr.offset(offset);
                // Ensure we don't read past end when reading the scalar value
                debug_assert!((offset as usize) + dtype_size <= data_len, "Read overrun during compaction");
                
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
            layout: new_layout
        }
    }

    // RESTORED: Fused Kernel Execution
    pub fn apply_fused_kernel(&self, kernel: &FusedKernel) -> ViewBuffer {
        if self.dtype() != DType::F32 {
            panic!("FusedKernel currently only supports F32 views");
        }

        let total_elems = self.layout.shape.iter().product();
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

            debug_assert!(offset >= 0 && (offset as usize) + 4 <= data_len, "Fused kernel read OOB");

            unsafe {
                let src_ptr = ptr.offset(offset) as *const f32;
                let mut acc = *src_ptr;

                for op in &kernel.ops {
                    match op {
                        ScalarOp::Add(c) => acc += c,
                        ScalarOp::Mul(c) => acc *= c,
                        ScalarOp::Relu => if acc < 0.0 { acc = 0.0 },
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
            _ => unimplemented!("Cast pair {:?} -> {:?} not implemented", self.dtype(), target),
        }
    }

    fn cast_impl<S, D>(&self) -> Self 
    where 
        S: ViewType + AsPrimitive<D>,
        D: ViewType + Copy + 'static
    {
        let elem_count = self.layout.shape.iter().product();
        let src_slice = unsafe {
            std::slice::from_raw_parts(self.as_ptr::<S>(), elem_count)
        };

        let new_data: Vec<D> = src_slice.iter().map(|&x| x.as_()).collect();
        Self::from_vec(new_data).reshape(self.layout.shape.clone())
    }

    pub(crate) fn reshape(mut self, shape: Vec<usize>) -> Self {
        self.layout.shape = shape;
        self.layout = crate::layout::Layout::new_contiguous(self.layout.shape, self.layout.dtype);
        self
    }
}