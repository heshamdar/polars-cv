//! The plugin's global allocator: polars' own, through pyo3-polars (CR-60).

/// Every allocation the plugin makes goes to polars' own allocator (CR-60).
///
/// Row buffers cross into polars and back, and on the system `malloc` glibc
/// decided when freed rows went back to the OS: a call holding many large rows
/// re-faulted all of them on every later call. Not in the lib's unit tests,
/// which count allocations through `test_alloc`'s allocator instead (and have
/// no Python for this one to relay to).
#[cfg(not(test))]
#[global_allocator]
static ALLOC: PluginAllocator = PluginAllocator(pyo3_polars::PolarsAllocator::new());

/// pyo3-polars' `PolarsAllocator`, noting that it is in use, so that
/// [`allocator_name`] reports the allocator the plugin actually has rather
/// than the one it declares: without `#[global_allocator]` it is never called.
#[cfg(not(test))]
struct PluginAllocator(pyo3_polars::PolarsAllocator);

/// Set by the first allocation through [`PluginAllocator`]. Read before it is
/// written, so after that first allocation every thread only reads it: a
/// store per allocation would bounce its cache line between the pool's threads.
static PLUGIN_ALLOCATOR_USED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

// SAFETY: every method forwards to `PolarsAllocator` unchanged.
#[cfg(not(test))]
unsafe impl std::alloc::GlobalAlloc for PluginAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        use std::sync::atomic::Ordering::Relaxed;
        if !PLUGIN_ALLOCATOR_USED.load(Relaxed) {
            PLUGIN_ALLOCATOR_USED.store(true, Relaxed);
        }
        unsafe { self.0.alloc(layout) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { self.0.dealloc(ptr, layout) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        unsafe { self.0.alloc_zeroed(layout) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe { self.0.realloc(ptr, layout, new_size) }
    }
}

/// Which allocator the plugin's allocations go to: `"polars"`, or `"system"`
/// when [`ALLOC`] is not the global allocator or polars' allocator capsule
/// cannot be imported (`PolarsAllocator` then falls back to the system one,
/// silently). `tests/test_allocator.py` makes either a failure.
///
/// The capsule half repeats `PolarsAllocator`'s own lookup (pyo3-polars 0.28):
/// the capsule by its name, with Python running -- which it is while this
/// module loads, after the plugin's first allocations.
pub(crate) fn allocator_name() -> &'static str {
    // Boxed so the allocation is not elided: it is what proves `ALLOC` is the
    // global allocator.
    drop(std::hint::black_box(Box::new(0u64)));
    if !PLUGIN_ALLOCATOR_USED.load(std::sync::atomic::Ordering::Relaxed) {
        return "system";
    }
    // SAFETY: called with the GIL held (module init); a null return sets a
    // Python error, which is cleared.
    let capsule = unsafe { pyo3::ffi::PyCapsule_Import(c"polars.polars._allocator".as_ptr(), 0) };
    if capsule.is_null() {
        unsafe { pyo3::ffi::PyErr_Clear() };
        "system"
    } else {
        "polars"
    }
}
