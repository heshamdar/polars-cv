use crate::dtype::{DType, ViewType};
use crate::layout::{ExternalLayout, Layout, LayoutFacts, LayoutReport};
use num_traits::AsPrimitive;
use std::sync::Arc;
use thiserror::Error;

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
}

#[derive(Debug, Clone)]
pub enum BufferStorage {
    Rust(Arc<Vec<u8>>),
    Arrow(arrow::buffer::Buffer),
}

impl BufferStorage {
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            BufferStorage::Rust(v) => v.as_ptr(),
            BufferStorage::Arrow(b) => b.as_ptr(),
        }
    }

    /// Returns the length of the underlying byte buffer.
    pub fn len(&self) -> usize {
        match self {
            BufferStorage::Rust(v) => v.len(),
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
            self.layout.dtype,
        )
    }

    pub fn storage_id(&self) -> usize {
        match &self.data {
            BufferStorage::Rust(arc) => Arc::as_ptr(arc) as usize,
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

        // Safety: as_ptr checks alignment.
        // We also need to ensure we don't read past the end, which is guaranteed
        // if self is contiguous (which it is, called from cast()) and elem_count matches.
        let src_slice = unsafe { std::slice::from_raw_parts(self.as_ptr::<S>(), elem_count) };

        let new_data: Vec<D> = src_slice.iter().map(|&x| x.as_()).collect();
        Self::from_vec(new_data).reshape(self.layout.shape.clone())
    }

    pub(crate) fn reshape(mut self, shape: Vec<usize>) -> Self {
        self.layout.shape = shape;
        self.layout = crate::layout::Layout::new_contiguous(self.layout.shape, self.layout.dtype);
        self
    }
}
