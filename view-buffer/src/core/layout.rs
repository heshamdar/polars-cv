//! Memory layout types for view-buffer.

use crate::core::dtype::DType;

/// External layout requirements for different libraries.
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
    pub offset: usize, // Offset in BYTES
}

impl LayoutFacts {
    /// Creates a new LayoutFacts from the given parameters.
    pub fn new(shape: &[usize], strides: &[isize], dtype: DType, offset: usize) -> Self {
        Self {
            rank: shape.len(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            dtype,
            offset,
        }
    }

    /// Returns true if the layout is contiguous (C-order/row-major).
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

    /// Returns true if the layout is channels-last (HWC format).
    pub fn is_channels_last(&self) -> bool {
        // Rank 3 and stride of last dim (C) is exactly the element size (1 element)
        self.rank == 3 && self.strides[2] == self.dtype.size_of() as isize
    }

    /// Returns true if rows are densely packed (may have padding between rows).
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

    /// Returns true if all strides are positive.
    ///
    /// Negative strides occur in flipped buffers and cannot be represented
    /// by external libraries that expect forward iteration (e.g., image crate).
    pub fn has_positive_strides(&self) -> bool {
        self.strides.iter().all(|&s| s >= 0)
    }

    /// Primary Predicate: Checks if this layout meets the requirements of a target crate.
    pub fn compatible_with(&self, target: ExternalLayout) -> bool {
        match target {
            ExternalLayout::NdArray => {
                // ndarray supports arbitrary strides (assuming element alignment)
                true
            }
            ExternalLayout::ImageCrate => {
                // image crate requires:
                // 1. Rank 2 (Grey) or 3 (RGB/A)
                // 2. Channels last (for Rank 3)
                // 3. Dense rows (no gaps between pixels)
                // 4. Positive strides (no flipped buffers - can't iterate backwards)
                (self.rank == 2 || self.rank == 3)
                    && (self.rank != 3 || self.is_channels_last())
                    && self.is_dense_rows()
                    && self.has_positive_strides()
            }
            ExternalLayout::FastImageResize => {
                // fast_image_resize usually requires strictly contiguous buffers
                self.is_contiguous()
            }
        }
    }
}

/// Persistent storage for layout information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub dtype: DType,
}

impl Layout {
    /// Creates a new contiguous layout with the given shape and dtype.
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

    /// Returns the total number of elements in the layout.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns true if the layout is contiguous.
    pub fn is_contiguous(&self) -> bool {
        LayoutFacts::from(self).is_contiguous()
    }

    /// Returns true if the layout is compatible with the target external layout.
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

/// A report of layout properties for inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutReport {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: DType,
    pub contiguous: bool,
    pub image_compatible: bool,
    pub ndarray_compatible: bool,
}
