use crate::dtype::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalLayout {
    NdArray,
    ImageCrate,
    FastImageResize,
}

/// Canonical layout facts used for validation.
/// This acts as the "Single Source of Truth" for all layout predicate logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutFacts {
    pub rank: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>, // Strides in BYTES
    pub dtype: DType,
    pub offset: usize,       // Offset in BYTES
}

impl LayoutFacts {
    pub fn new(shape: &[usize], strides: &[isize], dtype: DType, offset: usize) -> Self {
        Self {
            rank: shape.len(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            dtype,
            offset,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_strides = vec![0; self.rank];
        let mut current = self.dtype.size_of() as isize;
        
        // Compute standard C-order (row-major) strides
        for i in (0..self.rank).rev() {
            expected_strides[i] = current;
            current *= self.shape[i] as isize;
        }

        self.strides == expected_strides
    }

    pub fn is_channels_last(&self) -> bool {
        // Rank 3 and stride of last dim (C) is exactly the element size (1 element)
        self.rank == 3 && self.strides[2] == self.dtype.size_of() as isize
    }

    pub fn is_dense_rows(&self) -> bool {
        // "rows contiguous but may have padding"
        // This checks if pixels are packed tightly within a row.
        let elem_size = self.dtype.size_of() as isize;
        
        if self.rank == 2 {
            // [H, W]: Stride between pixels (W) must be element size
            self.strides[1] == elem_size
        } else if self.rank == 3 {
            // [H, W, C]: Stride between channels (C) must be element size
            //            Stride between pixels (W) must be C * element size
            let c = self.shape[2] as isize;
            self.strides[2] == elem_size && self.strides[1] == c * elem_size
        } else {
            false
        }
    }

    /// Primary Predicate: Checks if this layout meets the requirements of a target crate.
    pub fn compatible_with(&self, target: ExternalLayout) -> bool {
        match target {
            ExternalLayout::NdArray => {
                // ndarray supports arbitrary strides (assuming element alignment)
                true
            },
            ExternalLayout::ImageCrate => {
                // image crate requires:
                // 1. Rank 2 (Grey) or 3 (RGB/A)
                // 2. Channels last (for Rank 3)
                // 3. Dense rows (no gaps between pixels)
                (self.rank == 2 || self.rank == 3)
                    && (self.rank != 3 || self.is_channels_last())
                    && self.is_dense_rows()
            },
            ExternalLayout::FastImageResize => {
                // fast_image_resize usually requires strictly contiguous buffers
                self.is_contiguous()
            }
        }
    }
}

// Keep Layout struct as the persistent storage
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub dtype: DType,
}

impl Layout {
    pub fn new_contiguous(shape: Vec<usize>, dtype: DType) -> Self {
        let mut strides = vec![0; shape.len()];
        let mut current_stride = dtype.size_of() as isize;
        
        for i in (0..shape.len()).rev() {
            strides[i] = current_stride;
            current_stride *= shape[i] as isize;
        }

        Self {
            shape,
            strides,
            offset: 0,
            dtype,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        LayoutFacts::from(self).is_contiguous()
    }

    pub fn is_compatible_with(&self, target: ExternalLayout) -> bool {
        LayoutFacts::from(self).compatible_with(target)
    }
}

// Convert Layout storage to LayoutFacts view
impl From<&Layout> for LayoutFacts {
    fn from(l: &Layout) -> Self {
        Self::new(&l.shape, &l.strides, l.dtype, l.offset)
    }
}

// Reporting struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutReport {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: DType,
    pub contiguous: bool,
    pub image_compatible: bool,
    pub ndarray_compatible: bool,
}