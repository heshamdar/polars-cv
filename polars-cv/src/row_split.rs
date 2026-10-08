//! A plugin call's rows, split over the plugin's thread pool.
//!
//! Rows are independent, so a call can be split into contiguous row ranges
//! that run on several threads and come back in row order. Without this a
//! call used one core however many rows it held, so the in-memory engine was
//! single-threaded on a single-chunk frame (CR-32). The pool is the plugin's
//! own copy of polars' `THREAD_POOL` (a plugin links its own polars-core, so
//! it cannot join the host's), sized by `POLARS_MAX_THREADS`.
//!
//! **How many threads a call gets is decided by what runs now, and nothing
//! else.** Every thread running a plugin call's rows counts against one
//! plugin-wide budget, the pool's size ([`BUSY`]): the thread that made the
//! call, and each pool thread helping it. A call always runs its rows on its
//! own thread, one range at a time, and before each range invites idle pool
//! threads to help while the count is under budget. A helper leaves after
//! any range that finds the count over budget. So:
//!
//! - a call running alone (the in-memory engine's one call per query) uses
//!   the whole pool;
//! - in the streaming engine's steady state, as many concurrent calls (one
//!   per morsel) as threads, each call keeps its rows on its own thread. The
//!   cores are already busy, and helping would only move every row's buffers
//!   between threads (measured: up to half a byte-heavy streaming query's
//!   throughput);
//! - a large call beside small ones (a big row group's morsel next to a
//!   small one's) takes up the threads the small ones free as they finish.
//!
//! No state outlives a call. The overlap *history* this replaced (whether
//! the last call of a graph had overlapped another) let one query decide how
//! the next ran: an eager query after a streaming run of the same pipeline
//! ran on one thread, 3.9x slower.
//!
//! **The one row splitter of the plugin**: the pipeline executor
//! (`graph::compiled`) and the geometry accessors (`geom_params`) both run
//! their rows through [`run_split`], or [`split`] when a call splits more
//! than one phase (running the rows, then filling the output column).

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use pyo3_polars::export::polars_core::runtime::THREAD_POOL;

/// Threads running a plugin call's rows now, callers and helpers alike: the
/// one count every call's split reads. Plugin-wide, because every call
/// shares the one pool.
static BUSY: AtomicUsize = AtomicUsize::new(0);

/// Exclusive use of the plugin's split budget, for a test whose assertion
/// depends on [`BUSY`]: how many threads a call runs on. `BUSY` is
/// plugin-wide, and the test binary runs tests on several threads, so
/// another test's call would take part of the budget. While the guard is
/// held, a call from any other thread waits in [`split`] until it is dropped
/// (threads the test starts itself are let through with
/// [`exclusive::admit_current_thread`]), and the guard is only returned once
/// the calls already running have finished.
#[cfg(test)]
pub(crate) fn exclusive_pool() -> exclusive::Guard {
    exclusive::acquire()
}

/// The test-only gate behind [`exclusive_pool`].
#[cfg(test)]
pub(crate) mod exclusive {
    use std::collections::HashSet;
    use std::sync::{Condvar, Mutex, MutexGuard};
    use std::thread::ThreadId;

    /// Serialises the exclusive tests themselves.
    static OWNER: Mutex<()> = Mutex::new(());
    /// The admitted threads while a test holds the pool.
    static ADMITTED: Mutex<Option<HashSet<ThreadId>>> = Mutex::new(None);
    static RELEASED: Condvar = Condvar::new();

    pub(crate) struct Guard {
        _owner: MutexGuard<'static, ()>,
    }

    fn admitted() -> MutexGuard<'static, Option<HashSet<ThreadId>>> {
        ADMITTED.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(super) fn acquire() -> Guard {
        let owner = OWNER.lock().unwrap_or_else(|p| p.into_inner());
        *admitted() = Some(HashSet::from([std::thread::current().id()]));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while super::BUSY.load(std::sync::atomic::Ordering::SeqCst) != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "other tests' calls never finished"
            );
            std::thread::yield_now();
        }
        Guard { _owner: owner }
    }

    /// Let the current thread's calls through while a test holds the pool.
    pub(crate) fn admit_current_thread() {
        if let Some(set) = admitted().as_mut() {
            set.insert(std::thread::current().id());
        }
    }

    /// Wait while another test holds the pool and has not admitted this
    /// thread.
    pub(super) fn gate() {
        let me = std::thread::current().id();
        drop(
            RELEASED
                .wait_while(admitted(), |a| a.as_ref().is_some_and(|a| !a.contains(&me)))
                .unwrap_or_else(|p| p.into_inner()),
        );
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            *admitted() = None;
            RELEASED.notify_all();
        }
    }
}

/// Run `run(range_idx, rows)` over `0..len` in contiguous row ranges and
/// return each range's result in row order: [`split`] then [`Split::run`].
pub(crate) fn run_split<R: Send>(
    len: usize,
    run: impl Fn(usize, Range<usize>) -> R + Sync,
) -> Vec<R> {
    split().run(len, run)
}

/// Start a call: its thread counts as busy until the returned [`Split`] is
/// dropped. Hold it for every phase of the call that splits its rows.
pub(crate) fn split() -> Split {
    #[cfg(test)]
    exclusive::gate();
    BUSY.fetch_add(1, Ordering::SeqCst);
    Split { _caller: Slot }
}

/// One running call, whose own thread holds a slot of the budget.
pub(crate) struct Split {
    _caller: Slot,
}

/// One thread's place in [`BUSY`], given back on drop (a panic included).
struct Slot;

impl Slot {
    /// A place for a helper, if the pool has an idle thread.
    fn claim() -> Option<Slot> {
        let threads = THREAD_POOL.current_num_threads();
        let mut busy = BUSY.load(Ordering::SeqCst);
        while busy < threads {
            match BUSY.compare_exchange_weak(busy, busy + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return Some(Slot),
                Err(now) => busy = now,
            }
        }
        None
    }

    /// Whether more threads run rows than the pool has, so a helper should
    /// leave. A caller is never refused, so the count can exceed it.
    fn over_budget() -> bool {
        BUSY.load(Ordering::SeqCst) > THREAD_POOL.current_num_threads()
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        BUSY.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Split {
    /// Run `run(range_idx, rows)` over `0..len` in contiguous row ranges and
    /// return each range's result in row order.
    ///
    /// The ranges are taken in order from a shared counter: by this thread,
    /// and by the idle pool threads it invites before each range (see the
    /// module docs). A range that panics propagates the panic here once the
    /// helpers in flight have finished.
    pub(crate) fn run<R: Send>(
        &self,
        len: usize,
        run: impl Fn(usize, Range<usize>) -> R + Sync,
    ) -> Vec<R> {
        MAX_SPLIT_ROWS.fetch_max(len, Ordering::SeqCst);
        let ranges = row_ranges(len, THREAD_POOL.current_num_threads());
        if let [only] = ranges.as_slice() {
            let out = run(0, only.clone());
            LAST_SPLIT_WORKERS.store(1, Ordering::SeqCst);
            return vec![out];
        }
        let next = AtomicUsize::new(0);
        // A `Mutex` per range, so ranges can be filled from any thread with
        // results that are only `Send`; each is locked twice, uncontended.
        let outcomes: Vec<Mutex<Option<R>>> = ranges.iter().map(|_| Mutex::new(None)).collect();
        // Take the next range and run it; false once none is left.
        let take = || -> bool {
            let idx = next.fetch_add(1, Ordering::SeqCst);
            let Some(rows) = ranges.get(idx) else {
                return false;
            };
            let out = run(idx, rows.clone());
            *outcomes[idx].lock().unwrap_or_else(|p| p.into_inner()) = Some(out);
            true
        };
        let helpers = AtomicUsize::new(0);
        // This thread, and each helper that took a range.
        let workers = AtomicUsize::new(1);
        THREAD_POOL.in_place_scope(|scope| loop {
            // Invite idle threads, no more than there are ranges left for
            // them beside this thread's next one.
            let left = ranges.len().saturating_sub(next.load(Ordering::SeqCst));
            while helpers.load(Ordering::SeqCst) + 1 < left {
                let Some(slot) = Slot::claim() else { break };
                helpers.fetch_add(1, Ordering::SeqCst);
                let (take, helpers, workers) = (&take, &helpers, &workers);
                scope.spawn(move |_| {
                    let _slot = slot;
                    if take() {
                        workers.fetch_add(1, Ordering::SeqCst);
                        while !Slot::over_budget() && take() {}
                    }
                    helpers.fetch_sub(1, Ordering::SeqCst);
                });
            }
            if !take() {
                break;
            }
        });
        LAST_SPLIT_WORKERS.store(workers.into_inner(), Ordering::SeqCst);
        outcomes
            .into_iter()
            .map(|o| {
                o.into_inner()
                    .unwrap_or_else(|p| p.into_inner())
                    .expect("every range ran inside the scope")
            })
            .collect()
    }
}

/// Contiguous row ranges covering `0..len`, for `threads` workers.
///
/// A few ranges per thread so an expensive stretch of rows does not leave
/// the other threads idle, and so a thread freed part-way through a call
/// still finds ranges to take; one range when there is nothing to split.
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

/// How many workers ran the ranges of the most recent split: its calling
/// thread plus each pool thread that helped. Read through
/// `_lib._last_split_workers` by a Python test that checks a call's
/// parallelism at the user-facing entry point, where no Rust test hook
/// reaches. "Most recent" is racy under concurrent calls; it is meant for a
/// test that runs one query at a time.
static LAST_SPLIT_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// See [`LAST_SPLIT_WORKERS`].
pub(crate) fn last_split_workers() -> usize {
    LAST_SPLIT_WORKERS.load(Ordering::SeqCst)
}

/// The most rows one call ran since [`take_max_split_rows`] was last read:
/// under the streaming engine, the largest morsel a plugin call received.
/// Read through `_lib._take_max_split_rows` by the test that holds the
/// streaming guide's claims about morsel size to the plugin.
static MAX_SPLIT_ROWS: AtomicUsize = AtomicUsize::new(0);

/// See [`MAX_SPLIT_ROWS`]; reading it starts a new count.
pub(crate) fn take_max_split_rows() -> usize {
    MAX_SPLIT_ROWS.swap(0, Ordering::SeqCst)
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
