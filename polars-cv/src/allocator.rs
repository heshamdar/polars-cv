//! The plugin's global allocator: polars' own (CR-60).
//!
//! Every allocation goes to the allocator polars exports as the
//! `polars.polars._allocator` capsule. Row buffers cross into polars and back,
//! and on the system `malloc` glibc decided when freed rows went back to the
//! OS: a call holding many large rows re-faulted all of them on every later
//! call. Not in the lib's unit tests, which count allocations through
//! `test_alloc`'s allocator instead (and have no Python to relay to).
//!
//! This is pyo3-polars' `PolarsAllocator` with its capsule lookup made
//! allocation-free. `PolarsAllocator` looks the capsule up under
//! `Python::attach`, on the first allocation; pyo3 0.29.3's attach locks a
//! `std::sync::Mutex` whose first lock allocates on macOS, so when that is the
//! first allocation the lookup re-enters itself until the stack overflows: the
//! extension segfaulted on import on every macOS runner
//! (pola-rs/polars#29731). The lookup here calls only the C API, which
//! allocates through Python's allocator, never this one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicPtr, Ordering};

/// The capsule's layout: polars' allocator as four C functions (the struct
/// pyo3-polars declares for the same capsule).
/// Read only by the global allocator, which the lib's unit tests do not install.
#[cfg_attr(test, allow(dead_code))]
#[repr(C)]
struct AllocatorCapsule {
    alloc: unsafe extern "C" fn(usize, usize) -> *mut u8,
    dealloc: unsafe extern "C" fn(*mut u8, usize, usize),
    alloc_zeroed: unsafe extern "C" fn(usize, usize) -> *mut u8,
    realloc: unsafe extern "C" fn(*mut u8, usize, usize, usize) -> *mut u8,
}

unsafe extern "C" fn system_alloc(size: usize, align: usize) -> *mut u8 {
    // SAFETY: the size and alignment come from a valid `Layout`.
    unsafe { System.alloc(Layout::from_size_align_unchecked(size, align)) }
}

unsafe extern "C" fn system_dealloc(ptr: *mut u8, size: usize, align: usize) {
    // SAFETY: as `system_alloc`; `ptr` was allocated with this layout.
    unsafe { System.dealloc(ptr, Layout::from_size_align_unchecked(size, align)) }
}

unsafe extern "C" fn system_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    // SAFETY: as `system_alloc`.
    unsafe { System.alloc_zeroed(Layout::from_size_align_unchecked(size, align)) }
}

unsafe extern "C" fn system_realloc(
    ptr: *mut u8,
    size: usize,
    align: usize,
    new: usize,
) -> *mut u8 {
    // SAFETY: as `system_dealloc`.
    unsafe { System.realloc(ptr, Layout::from_size_align_unchecked(size, align), new) }
}

/// What the plugin allocates with when polars' capsule cannot be imported (no
/// interpreter, or a polars that moved it). `tests/test_allocator.py` makes
/// that a failure, through [`allocator_name`].
static SYSTEM: AllocatorCapsule = AllocatorCapsule {
    alloc: system_alloc,
    dealloc: system_dealloc,
    alloc_zeroed: system_alloc_zeroed,
    realloc: system_realloc,
};

/// The allocator every allocation goes to, chosen by the first one: null until
/// then, after that polars' capsule or [`SYSTEM`], and never changed again (a
/// buffer must be freed by the allocator that allocated it).
static CHOSEN: AtomicPtr<AllocatorCapsule> = AtomicPtr::new(std::ptr::null_mut());

#[cfg(not(test))]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator;

/// The global allocator: every call goes to [`CHOSEN`].
#[cfg(not(test))]
struct PolarsAllocator;

#[cfg(not(test))]
std::thread_local! {
    /// Set while this thread is choosing the allocator. Const-initialised and
    /// without a destructor, so reading it never allocates.
    static CHOOSING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(not(test))]
impl PolarsAllocator {
    #[inline]
    fn get(&self) -> &'static AllocatorCapsule {
        let chosen = CHOSEN.load(Ordering::Acquire);
        if chosen.is_null() {
            return Self::choose();
        }
        // SAFETY: `CHOSEN` only ever holds a `&'static AllocatorCapsule`.
        unsafe { &*chosen }
    }

    /// Picks the allocator on the first allocation. Threads racing here may
    /// each look the capsule up; the first to store wins and the rest adopt it.
    #[cold]
    fn choose() -> &'static AllocatorCapsule {
        // An allocation from inside the lookup would re-enter it until the
        // stack overflowed (the macOS crash this module exists to avoid), and
        // falling back to the system allocator for it would free its buffer
        // with the wrong allocator later. Abort with the reason instead.
        if CHOOSING.with(|c| c.replace(true)) {
            let msg = b"polars-cv: the allocator re-entered itself while importing \
                        polars' allocator capsule\n";
            // SAFETY: writes a static buffer to stderr; allocates nothing.
            unsafe { libc::write(2, msg.as_ptr().cast(), msg.len() as _) };
            std::process::abort();
        }
        let found = polars_capsule();
        CHOOSING.with(|c| c.set(false));
        let pick = found.unwrap_or(&SYSTEM) as *const AllocatorCapsule as *mut AllocatorCapsule;
        match CHOSEN.compare_exchange(
            std::ptr::null_mut(),
            pick,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            // SAFETY: both arms hold a `&'static AllocatorCapsule`.
            Ok(_) => unsafe { &*pick },
            Err(winner) => unsafe { &*winner },
        }
    }
}

/// Polars' allocator capsule, or `None` with no interpreter or no capsule.
/// Calls the C API alone: it allocates through Python's allocator, so nothing
/// here can come back to [`PolarsAllocator`].
#[cfg(not(test))]
fn polars_capsule() -> Option<&'static AllocatorCapsule> {
    use pyo3::ffi;
    // SAFETY: always safe to call.
    if unsafe { ffi::Py_IsInitialized() } == 0 {
        return None;
    }
    // SAFETY: the interpreter is initialised; `PyGILState_Ensure` attaches this
    // thread (re-entrantly if it already is) and the matching release restores
    // its previous state. A null import sets a Python error, which is cleared.
    let capsule = unsafe {
        let gil = ffi::PyGILState_Ensure();
        let capsule = ffi::PyCapsule_Import(c"polars.polars._allocator".as_ptr(), 0);
        if capsule.is_null() {
            ffi::PyErr_Clear();
        }
        ffi::PyGILState_Release(gil);
        capsule
    };
    // SAFETY: polars' capsule points at a static `AllocatorCapsule`.
    unsafe { capsule.cast::<AllocatorCapsule>().as_ref() }
}

// SAFETY: every method forwards to one allocator, the same for the life of the
// process, so each buffer is freed by the allocator that allocated it.
#[cfg(not(test))]
unsafe impl GlobalAlloc for PolarsAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards a valid layout.
        unsafe { (self.get().alloc)(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by the same allocator with this layout.
        unsafe { (self.get().dealloc)(ptr, layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards a valid layout.
        unsafe { (self.get().alloc_zeroed)(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as `dealloc`, with the caller's valid new size.
        unsafe { (self.get().realloc)(ptr, layout.size(), layout.align(), new_size) }
    }
}

/// Which allocator the plugin's allocations go to: `"polars"`, or `"system"`
/// when [`PolarsAllocator`] is not the global allocator or chose [`SYSTEM`]
/// because polars' capsule could not be imported. `tests/test_allocator.py`
/// makes either a failure. Reads the choice the allocator made, so it cannot
/// disagree with it.
pub(crate) fn allocator_name() -> &'static str {
    // Boxed so the allocation is not elided: through the global allocator it
    // makes the choice if nothing has yet.
    drop(std::hint::black_box(Box::new(0u64)));
    let chosen = CHOSEN.load(Ordering::Acquire);
    if chosen.is_null() || std::ptr::eq(chosen, &SYSTEM) {
        "system"
    } else {
        "polars"
    }
}
