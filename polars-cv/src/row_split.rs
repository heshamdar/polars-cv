//! A plugin call's rows, split over the plugin's thread pool.
//!
//! Rows are independent, so a call can be split into contiguous row ranges
//! that run on the plugin's thread pool and are concatenated in order.
//! Without this a call used one core however many rows it held, so the
//! in-memory engine was single-threaded on a single-chunk frame (CR-32). The
//! pool is the plugin's own copy of polars' `THREAD_POOL` (a plugin links its
//! own polars-core, so it cannot join the host's), sized by
//! `POLARS_MAX_THREADS`.
//!
//! The streaming engine is already parallel: it runs a query's morsels as
//! concurrent calls of the same function, one per host thread. Splitting
//! those as well only moved every row's buffers between threads and
//! oversubscribed the cores (up to half a byte-heavy streaming query's
//! throughput), so a call spreads only when it runs alone and the last call
//! did too ([`Call::spreads`]).
//!
//! **The one row splitter of the plugin**: the pipeline executor
//! (`graph::compiled`) and the geometry accessors (`geom_arity`) both run
//! their rows through [`run_split`], each with its own [`CallTracker`].

use std::ops::Range;

/// This call site's own [`CallTracker`], in a `static`: every plugin function
/// that splits its rows keeps one, so its calls' overlap is its own.
#[macro_export]
macro_rules! geom_calls {
    () => {{
        static CALLS: $crate::row_split::CallTracker = $crate::row_split::CallTracker::new();
        &CALLS
    }};
}
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use pyo3_polars::export::polars_core::runtime::THREAD_POOL;

/// Run `run(range_idx, rows)` over `0..len` in contiguous row ranges and
/// return each range's result in row order.
///
/// The ranges run on the plugin's pool when this call spreads
/// ([`Call::spreads`]), and as one range on the caller's thread otherwise.
pub(crate) fn run_split<R: Send>(
    calls: &CallTracker,
    len: usize,
    run: impl Fn(usize, Range<usize>) -> R + Sync,
) -> Vec<R> {
    let call = calls.enter();
    let workers = if call.spreads() {
        THREAD_POOL.current_num_threads()
    } else {
        1
    };
    let ranges = row_ranges(len, workers);
    if let [only] = ranges.as_slice() {
        return vec![run(0, only.clone())];
    }
    let mut outcomes: Vec<Option<R>> = ranges.iter().map(|_| None).collect();
    let run = &run;
    THREAD_POOL.scope(|scope| {
        for ((range_idx, rows), slot) in ranges.into_iter().enumerate().zip(outcomes.iter_mut()) {
            scope.spawn(move |_| *slot = Some(run(range_idx, rows)));
        }
    });
    outcomes
        .into_iter()
        .map(|o| o.expect("every range ran to completion inside the scope"))
        .collect()
}

/// Contiguous row ranges covering `0..len`, for `threads` workers.
///
/// A few ranges per thread so an expensive stretch of rows does not leave
/// the other threads idle; one range when there is nothing to split.
pub(crate) fn row_ranges(len: usize, threads: usize) -> Vec<Range<usize>> {
    const RANGES_PER_THREAD: usize = 4;
    // One worker gains nothing from ranges: they would only hand the rows to
    // another thread while this one waits.
    if threads < 2 {
        return std::iter::once(0..len).collect();
    }
    let count = (threads * RANGES_PER_THREAD).clamp(1, len.max(1));
    let (base, extra) = (len / count, len % count);
    let mut start = 0;
    (0..count)
        .map(|i| {
            let end = start + base + usize::from(i < extra);
            let range = start..end;
            start = end;
            range
        })
        .collect()
}

/// How a graph's calls overlap in time.
///
/// The host engine decides how a plugin is called and does not say which
/// engine it is. Overlap is the observable difference: the streaming engine
/// runs a graph's morsels as concurrent calls, the in-memory engine makes one
/// call per query.
pub(crate) struct CallTracker {
    /// Calls running now.
    pub(crate) running: AtomicUsize,
    /// Calls ever started; a call that sees it move on was overlapped.
    started: AtomicUsize,
    /// Whether the last call to finish overlapped another.
    pub(crate) overlapping: AtomicBool,
}

impl CallTracker {
    /// A tracker that has seen no call; `const`, so a plugin function can
    /// keep one in a `static`.
    pub(crate) const fn new() -> Self {
        CallTracker {
            running: AtomicUsize::new(0),
            started: AtomicUsize::new(0),
            overlapping: AtomicBool::new(false),
        }
    }

    pub(crate) fn enter(&self) -> Call<'_> {
        let ticket = self.started.fetch_add(1, Ordering::SeqCst) + 1;
        let alone = self.running.fetch_add(1, Ordering::SeqCst) == 0;
        Call {
            tracker: self,
            ticket,
            alone,
        }
    }
}

/// One running call of a graph (see [`CallTracker`]).
pub(crate) struct Call<'a> {
    tracker: &'a CallTracker,
    /// This call's number among the graph's calls.
    ticket: usize,
    /// No other call was running when this one started.
    alone: bool,
}

impl Call<'_> {
    /// Whether this call spreads its rows over the thread pool: it started
    /// alone, and the last call to finish ran alone too. A streaming query's
    /// first morsel also starts alone, a moment before the others, so being
    /// alone at the start cannot tell the engines apart by itself.
    pub(crate) fn spreads(&self) -> bool {
        self.alone && !self.tracker.overlapping.load(Ordering::SeqCst)
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        let overlapped = !self.alone || self.tracker.started.load(Ordering::SeqCst) != self.ticket;
        self.tracker.overlapping.store(overlapped, Ordering::SeqCst);
        self.tracker.running.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::row_ranges;

    /// One worker has nothing to split across: the rows run on the caller.
    #[test]
    fn a_single_worker_takes_the_whole_call() {
        assert_eq!(
            row_ranges(300, 1),
            std::iter::once(0..300).collect::<Vec<_>>()
        );
        assert_eq!(row_ranges(0, 1), std::iter::once(0..0).collect::<Vec<_>>());
        assert_eq!(row_ranges(300, 2).len(), 8);
    }
}
