//! Allocation counting for tests of what the executor copies.
//!
//! A copy of a buffer is one allocation at least its size, so counting such
//! allocations over a call observes copies that no output reveals. The count
//! is per thread, so tests running concurrently do not see each other's
//! allocations, and so only work done on the calling thread is counted: a
//! call under test must run inline (one row runs as one range, on the caller).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    /// Allocations of at least this many bytes are counted; 0 is off.
    static THRESHOLD: Cell<usize> = const { Cell::new(0) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    // `try_with`: the allocator also runs during thread teardown.
    let _ = THRESHOLD.try_with(|t| {
        if t.get() != 0 && size >= t.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// How many allocations of at least `threshold` bytes `f` makes on this
/// thread.
pub(crate) fn large_allocations<R>(threshold: usize, f: impl FnOnce() -> R) -> (R, usize) {
    COUNT.with(|c| c.set(0));
    THRESHOLD.with(|t| t.set(threshold));
    let out = f();
    THRESHOLD.with(|t| t.set(0));
    (out, COUNT.with(Cell::get))
}
