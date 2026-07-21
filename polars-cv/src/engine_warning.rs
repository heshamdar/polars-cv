//! One-time "running single-threaded" warning for the graph plugin.
//!
//! The plugin does not parallelize *within* a call — all multi-core execution
//! comes from the Polars **streaming** engine slicing the input into morsels and
//! invoking the plugin concurrently. Under the default in-memory engine a single
//! `collect` therefore runs the whole column on one thread, and nothing signals
//! it. This module emits a single, actionable warning when it sees a large batch
//! go through a single call without the engine ever running two calls at once.
//!
//! Two signals combine (both per-process, evaluated per call — never a
//! cumulative counter across queries, which would false-positive on many small
//! collects):
//! - **Per-call row count.** A single call carrying a very large number of rows
//!   is almost certainly the in-memory engine handing over the whole column;
//!   streaming morsels are far smaller.
//! - **Observed concurrency.** A [`CallGuard`] tracks how many calls run at once.
//!   Once two are ever seen concurrently the engine is parallelizing, and the
//!   warning is suppressed for the rest of the process (the user clearly knows
//!   about streaming).
//!
//! Escape hatches:
//! - `POLARS_CV_SILENCE_ENGINE_WARNING=1` — never warn.
//! - `POLARS_CV_ENGINE_WARN_ROWS=<n>` — override the per-call row threshold.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Plugin calls currently executing (RAII-tracked by [`CallGuard`]).
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Maximum number of plugin calls ever seen executing at the same instant.
static MAX_CONCURRENCY: AtomicUsize = AtomicUsize::new(0);
/// Whether the one-time warning has already fired.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Rows in a *single* call above which we treat the call as an in-memory
/// whole-column handover. Chosen well above a typical streaming morsel.
const DEFAULT_WARN_ROWS: u64 = 50_000;

/// RAII guard: bump the in-flight counter for the duration of one plugin call so
/// concurrent calls are actually observed, and check the warning condition on
/// entry.
pub struct CallGuard;

impl CallGuard {
    /// Enter a plugin call processing `n_rows` rows.
    pub fn enter(n_rows: usize) -> Self {
        let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
        // Record the high-water mark of concurrent calls.
        MAX_CONCURRENCY.fetch_max(now, Ordering::SeqCst);
        maybe_warn(n_rows as u64);
        CallGuard
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The pure warning decision, factored out so it is unit-testable without
/// touching any process-global state or environment.
fn should_warn(
    already_warned: bool,
    silenced: bool,
    max_concurrency: usize,
    parallelism: usize,
    call_rows: u64,
    threshold: u64,
) -> bool {
    !already_warned
        && !silenced
        // The engine has parallelized across morsels at least once — no footgun.
        && max_concurrency < 2
        // Nothing to gain on a single-core machine (or an explicit 1-thread cap).
        && parallelism > 1
        // A single call this large is an in-memory whole-column handover.
        && call_rows >= threshold
}

/// Available parallelism, honoring `POLARS_MAX_THREADS` when set.
fn available_parallelism() -> usize {
    if let Ok(v) = std::env::var("POLARS_MAX_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n;
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn warn_row_threshold() -> u64 {
    std::env::var("POLARS_CV_ENGINE_WARN_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_WARN_ROWS)
}

fn maybe_warn(call_rows: u64) {
    let decided = should_warn(
        WARNED.load(Ordering::Relaxed),
        std::env::var("POLARS_CV_SILENCE_ENGINE_WARNING").is_ok(),
        MAX_CONCURRENCY.load(Ordering::SeqCst),
        available_parallelism(),
        call_rows,
        warn_row_threshold(),
    );
    if !decided {
        return;
    }
    // Win the race to warn exactly once.
    if WARNED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        eprintln!(
            "polars-cv: cv.pipe processed a large batch in a single call, which \
             means it ran single-threaded — the plugin only runs multi-core under \
             the Polars streaming engine, and the default in-memory `collect` runs \
             it on one thread. For multi-core throughput use \
             `.collect(engine=\"streaming\")` (or `scan_*` + streaming). Silence \
             this with POLARS_CV_SILENCE_ENGINE_WARNING=1."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::should_warn;

    const T: u64 = 50_000;

    #[test]
    fn warns_on_large_single_threaded_call() {
        // Large single call, no observed concurrency, multi-core, not silenced.
        assert!(should_warn(false, false, 1, 8, T, T));
        assert!(should_warn(false, false, 1, 8, T + 1, T));
    }

    #[test]
    fn suppressed_when_already_warned_or_silenced() {
        assert!(!should_warn(true, false, 1, 8, T, T));
        assert!(!should_warn(false, true, 1, 8, T, T));
    }

    #[test]
    fn suppressed_once_concurrency_observed() {
        // Streaming ran two calls at once — no footgun even for a huge call.
        assert!(!should_warn(false, false, 2, 8, 10 * T, T));
    }

    #[test]
    fn suppressed_on_single_core_or_small_call() {
        assert!(!should_warn(false, false, 1, 1, 10 * T, T)); // single core
        assert!(!should_warn(false, false, 1, 8, T - 1, T)); // below threshold
    }
}
