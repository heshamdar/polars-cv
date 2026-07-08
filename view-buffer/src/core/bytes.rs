//! Alignment-aware owned byte storage.

use std::alloc::Layout as AllocLayout;
use std::ops::Deref;
use std::ptr::NonNull;

use super::dtype::ViewType;

/// Owned byte storage that remembers the alignment of its original
/// allocation.
///
/// Kernel outputs are built as `Vec<T>` and stored as raw bytes. The
/// allocator contract requires deallocating with the exact `Layout`
/// (size **and** alignment) the allocation was created with, so
/// rebuilding a `Vec<u8>` (align 1) over a `Vec<f32>` allocation and
/// letting it drop is undefined behavior (`Vec::from_raw_parts` spells
/// this out). `AlignedBytes` keeps the pointer, byte length/capacity,
/// and the original alignment, and deallocates with the matching layout.
#[derive(Debug)]
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    /// Capacity of the original allocation in bytes (0 = no allocation).
    cap: usize,
    /// Alignment of the original allocation.
    align: usize,
}

// SAFETY: AlignedBytes uniquely owns its allocation and exposes no
// interior mutability, so it can be sent and shared across threads.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// Takes ownership of a typed Vec's allocation, recording its alignment.
    pub fn from_typed_vec<T: ViewType>(data: Vec<T>) -> Self {
        // SAFETY: ManuallyDrop transfers ownership of the allocation to
        // AlignedBytes (freed in Drop with the recorded layout); T is Copy
        // via ViewType, so no element Drop glue is lost.
        let mut data = std::mem::ManuallyDrop::new(data);
        let elem = std::mem::size_of::<T>();
        Self {
            ptr: NonNull::new(data.as_mut_ptr() as *mut u8).expect("Vec pointer is never null"),
            len: data.len() * elem,
            cap: data.capacity() * elem,
            align: std::mem::align_of::<T>(),
        }
    }

    /// Allocates `data.len()` bytes at `alignment` and copies `data` in.
    pub fn copy_from_slice_aligned(data: &[u8], alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        let len = data.len();
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                cap: 0,
                align: alignment,
            };
        }
        let layout =
            AllocLayout::from_size_align(len, alignment).expect("Invalid layout parameters");
        // SAFETY: the layout has non-zero size; the copy stays within the
        // fresh allocation.
        unsafe {
            let raw = std::alloc::alloc(layout);
            let Some(ptr) = NonNull::new(raw) else {
                std::alloc::handle_alloc_error(layout)
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), raw, len);
            Self {
                ptr,
                len,
                cap: len,
                align: alignment,
            }
        }
    }

    /// Pointer to the first byte.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Mutable pointer to the first byte (requires exclusive access).
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the storage holds zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignment of the underlying allocation.
    pub fn alignment(&self) -> usize {
        self.align
    }

    /// Reclaims the allocation as a `Vec<u8>`.
    ///
    /// Zero-copy when the original allocation used alignment 1 (it came
    /// from byte data); otherwise the bytes are copied, because a `Vec<u8>`
    /// may not own an allocation with a different alignment.
    pub fn into_vec(self) -> Vec<u8> {
        if self.align == 1 && self.cap != 0 {
            let this = std::mem::ManuallyDrop::new(self);
            // SAFETY: the allocation was created with align 1 and cap
            // bytes, exactly what Vec<u8> will deallocate with.
            unsafe { Vec::from_raw_parts(this.ptr.as_ptr(), this.len, this.cap) }
        } else {
            self.as_ref().to_vec()
        }
    }
}

impl From<Vec<u8>> for AlignedBytes {
    fn from(v: Vec<u8>) -> Self {
        Self::from_typed_vec(v)
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: ptr/len describe the owned, initialized byte range.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl AsRef<[u8]> for AlignedBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        if self.cap != 0 {
            // SAFETY: (cap, align) is exactly the layout of the original
            // allocation: Vec<T> allocates Layout::array::<T>(capacity),
            // whose size is capacity * size_of::<T>() and whose alignment
            // is align_of::<T>(); copy_from_slice_aligned records its own
            // layout directly.
            unsafe {
                std::alloc::dealloc(
                    self.ptr.as_ptr(),
                    AllocLayout::from_size_align_unchecked(self.cap, self.align),
                )
            }
        }
    }
}
