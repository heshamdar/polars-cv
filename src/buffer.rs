use std::sync::Arc;
use crate::dtype::{DType, TensorType};
use crate::layout::{Layout, ExternalLayout, LayoutReport, LayoutFacts};
use thiserror::Error;
use num_traits::AsPrimitive;

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
}

#[derive(Debug, Clone)]
pub struct TensorBuffer {
    pub(crate) data: BufferStorage,
    pub(crate) layout: Layout,
}

impl TensorBuffer {
    pub fn from_vec<T: TensorType>(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
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
    
    pub fn from_arrow_buffer(buffer: arrow::buffer::Buffer, shape: Vec<usize>, dtype: DType) -> Self {
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

    pub unsafe fn as_ptr<T>(&self) -> *const T {
        self.data.as_ptr().add(self.layout.offset) as *const T
    }

    pub fn as_raw_parts(&self) -> (*const u8, &[usize], &[isize], DType) {
        (
            unsafe { self.data.as_ptr().add(self.layout.offset) },
            &self.layout.shape,
            &self.layout.strides,
            self.layout.dtype
        )
    }

    /// Returns a unique identifier for the underlying storage allocation.
    /// Used to verify zero-copy behavior (views should share the same storage ID).
    pub fn storage_id(&self) -> usize {
        match &self.data {
            BufferStorage::Rust(arc) => Arc::as_ptr(arc) as usize,
            // For Arrow, as_ptr returns the pointer to the data.
            // Since we clone the buffer container on slice (but not data), this pointer remains constant.
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

        for _ in 0..total_elems {
            let mut offset = base_offset as isize;
            for (dim, &idx) in indices.iter().enumerate() {
                offset += (idx as isize) * strides[dim];
            }
            
            unsafe {
                let src = ptr.offset(offset);
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
        S: TensorType + AsPrimitive<D>,
        D: TensorType + Copy + 'static
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