//! Runtime CPU dispatch: **the one way a kernel gets a wider instruction set.**
//!
//! Published wheels are built for the x86-64 baseline (SSE2), where much of
//! the engine's arithmetic cannot vectorise: `f32::round` is a `roundf`
//! libcall per element, and the saturating float→int conversions stay
//! scalar. A kernel written as a [`SimdKernel`] and run through [`dispatch`]
//! is compiled twice — as written, and with AVX2 enabled for its whole body —
//! and the AVX2 build runs when the CPU has it.
//!
//! The whole body, not an inner loop: dispatching per row-sized loop measured
//! *slower* than baseline (CR-35), and a closure is a separate function that
//! does not inherit its caller's target features. So a kernel's
//! [`run`](SimdKernel::run) must be `#[inline(always)]` and must not hand its
//! loops to a non-inlined callee (a `thread_local!` `with` closure, say).
//!
//! **Only AVX2 is enabled, never FMA.** Rust never contracts `a * b + c`
//! into a fused multiply-add on its own, so both builds perform the same IEEE
//! operations and the outputs are bit-identical. That is not left as a
//! promise: in a debug build every dispatched call on an AVX2 machine also
//! runs the portable build (on a helper thread, see `run_portable_aside`)
//! and asserts the two outputs are byte-identical.
//! Every test that reaches a kernel through the engine (Rust or Python, which
//! runs the debug extension) therefore checks it, on whatever input it used.
//! The check is meaningful where the portable build is baseline code — CI and
//! the published-wheel flags (`RUSTFLAGS=""`); the local dev config builds
//! everything for `x86-64-v3`, where the two builds coincide.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::ViewType;

/// A kernel [`dispatch`] can compile for AVX2.
///
/// `Clone + Send` so a debug build can run it a second time on a helper
/// thread; kernels hold slices and scalars, so the clone is a copy of a few
/// words.
pub(crate) trait SimdKernel: Clone + Send {
    /// What the kernel produces; compared byte-for-byte in debug builds.
    type Output: KernelOutput + Send;

    /// The kernel's whole body. Implementations must be `#[inline(always)]`
    /// so the body is compiled into each dispatched build.
    fn run(self) -> Self::Output;
}

/// A kernel output's bytes, for the debug parity check.
pub(crate) trait KernelOutput {
    #[cfg_attr(
        not(any(test, debug_assertions)),
        expect(
            dead_code,
            reason = "only the debug parity check reads a kernel's bytes"
        )
    )]
    fn output_bytes(&self) -> &[u8];
}

impl<T: ViewType> KernelOutput for Vec<T> {
    fn output_bytes(&self) -> &[u8] {
        // SAFETY: every `ViewType` is a plain numeric type with no padding,
        // so its elements are readable as `size_of::<T>()` bytes each.
        unsafe {
            std::slice::from_raw_parts(self.as_ptr().cast::<u8>(), std::mem::size_of_val(&self[..]))
        }
    }
}

impl KernelOutput for ViewBuffer {
    fn output_bytes(&self) -> &[u8] {
        assert!(
            self.layout_facts().is_contiguous(),
            "a kernel's output buffer is freshly allocated and contiguous"
        );
        let (ptr, shape, _, dtype) = self.as_raw_parts();
        let len = shape.iter().product::<usize>() * dtype.size_of();
        // SAFETY: a contiguous view's elements are `len` packed bytes from
        // its data pointer.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Run `kernel`: its AVX2 build when the CPU supports AVX2, else as written.
#[inline]
pub(crate) fn dispatch<K: SimdKernel>(kernel: K) -> K::Output {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            #[cfg(debug_assertions)]
            let check = kernel.clone();
            // SAFETY: the CPU supports AVX2, checked just above.
            let out = unsafe { run_avx2(kernel) };
            #[cfg(debug_assertions)]
            assert_bit_identical(&run_portable_aside(check), &out);
            return out;
        }
    }
    kernel.run()
}

/// The portable build of `kernel`, run on a scoped helper thread for the
/// debug parity check.
///
/// On its own thread so the check's allocations are invisible to the
/// per-thread allocation accounting the copy-count guards read (e.g.
/// `polars-cv`'s `scalar_ops_run_in_place_on_an_owned_buffer`): those guards
/// pin what the production path allocates, and a diagnostic second run is not
/// part of it. A panic in the portable build is re-raised here.
#[cfg(debug_assertions)]
fn run_portable_aside<K: SimdKernel>(kernel: K) -> K::Output {
    std::thread::scope(|scope| scope.spawn(move || kernel.run()).join())
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// [`SimdKernel::run`] compiled with AVX2 enabled.
///
/// # Safety
/// The caller must have checked that the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn run_avx2<K: SimdKernel>(kernel: K) -> K::Output {
    #[cfg(test)]
    tests::AVX2_RUNS.with(|n| n.set(n.get() + 1));
    kernel.run()
}

/// Panic unless the two builds of a kernel produced the same bytes.
#[cfg(any(test, debug_assertions))]
fn assert_bit_identical<O: KernelOutput>(portable: &O, dispatched: &O) {
    let (a, b) = (portable.output_bytes(), dispatched.output_bytes());
    if a != b {
        let first = a.iter().zip(b).position(|(x, y)| x != y);
        panic!(
            "internal error: a dispatched kernel's AVX2 build disagrees with its portable \
             build ({} vs {} bytes, first difference at byte {:?}). The AVX2 build must \
             perform the same operations; was FMA or a reassociation introduced?",
            a.len(),
            b.len(),
            first
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// AVX2 builds run on this thread, so a test can see the dispatch
        /// actually took the wide path.
        pub(crate) static AVX2_RUNS: Cell<usize> = const { Cell::new(0) };
    }

    #[derive(Clone)]
    struct Doubled<'a>(&'a [f32]);

    impl SimdKernel for Doubled<'_> {
        type Output = Vec<f32>;
        #[inline(always)]
        fn run(self) -> Vec<f32> {
            self.0.iter().map(|&x| x * 2.0 + 1.0).collect()
        }
    }

    #[test]
    fn dispatch_takes_the_avx2_build_when_the_cpu_has_it() {
        let input: Vec<f32> = (0..37).map(|i| i as f32 * 0.37).collect();
        let before = AVX2_RUNS.with(Cell::get);
        let out = dispatch(Doubled(&input));
        let expected: Vec<f32> = input.iter().map(|&x| x * 2.0 + 1.0).collect();
        assert_eq!(out, expected);
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            assert_eq!(
                AVX2_RUNS.with(Cell::get),
                before + 1,
                "the AVX2 build did not run"
            );
        }
    }

    #[test]
    #[should_panic(expected = "disagrees with its portable build")]
    fn the_parity_check_rejects_outputs_that_differ() {
        assert_bit_identical(&vec![1.0f32, 2.0, 3.0], &vec![1.0f32, 2.0, 3.0000002]);
    }

    #[test]
    fn the_parity_check_compares_bits_not_values() {
        // -0.0 == 0.0 as values, but a kernel that flips a sign bit changed
        // its output.
        let result =
            std::panic::catch_unwind(|| assert_bit_identical(&vec![0.0f32], &vec![-0.0f32]));
        assert!(result.is_err(), "-0.0 and 0.0 differ in their bits");
    }
}
