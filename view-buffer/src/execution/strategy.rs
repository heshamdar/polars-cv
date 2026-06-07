//! Execution strategy for pipeline runs.
//!
//! The strategy is stored in the `ExecutionPlan` (set from the thread-local
//! default when `ViewExpr::plan()` is called) and resolved against the actual
//! source image size at `ExecutionPlan::execute()` time.
//!
//! # Choosing a strategy
//!
//! * **`FullImage`** – one pass over the full buffer per op.  Best for small
//!   images that already fit in L2.
//! * **`Tiled`** – "outer-loop" segment tiling: a group of consecutive
//!   tileable ops is executed tile-by-tile, keeping each tile resident in
//!   L1/L2 through all ops in the segment before the next tile is fetched.
//!   Only activates when the image exceeds `threshold_bytes`.
//! * **`Adaptive`** (default) – selects `Tiled` automatically when the source
//!   exceeds 512 KB; otherwise falls back to `FullImage`.
//!
//! # Thread-local default
//!
//! The default strategy is `Adaptive`.  Change it for the current thread with
//! [`set_execution_strategy`] or transiently with [`with_execution_strategy`].
//! The environment variable `VIEW_BUFFER_STRATEGY` can pre-set the default
//! (`full`, `tiled`, or `adaptive`).

use std::cell::RefCell;

/// Default tile edge length in pixels (256 × 256 × 3 ≈ 192 KB for RGB u8,
/// fits in typical L2 caches).
pub const DEFAULT_TILE_SIZE: usize = 256;

/// Default threshold above which `Adaptive` activates tiling.
/// 512 KB means we tile when the image itself won't fit in L2.
pub const ADAPTIVE_THRESHOLD_BYTES: usize = 512 * 1024;

/// Controls how an [`ExecutionPlan`](crate::execution::plan::ExecutionPlan)
/// processes its buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    /// Always process the full buffer in a single pass per op.
    ///
    /// Optimal for images that already fit in L2 cache.  This is the
    /// minimal-overhead path and matches the pre-tiling behaviour exactly.
    FullImage,

    /// Process the image in cache-sized tiles, executing all ops within a
    /// tileable segment on each tile before moving to the next tile.
    ///
    /// Tiling only activates when the source image's byte footprint exceeds
    /// `threshold_bytes`; smaller images fall back to `FullImage` automatically.
    Tiled {
        /// Edge length of each square tile in pixels.
        tile_size: usize,
        /// Minimum source image size (bytes) before tiling kicks in.
        threshold_bytes: usize,
    },

    /// Auto-select: `Tiled` with defaults when source image > 512 KB,
    /// `FullImage` otherwise.
    Adaptive,
}

impl Default for ExecutionStrategy {
    fn default() -> Self {
        ExecutionStrategy::Adaptive
    }
}

impl ExecutionStrategy {
    /// Resolve the strategy to a concrete tile size, or `None` for full-image.
    ///
    /// `image_bytes` is the byte footprint of the source buffer.
    #[inline]
    pub(crate) fn resolve(&self, image_bytes: usize) -> Option<usize> {
        match self {
            ExecutionStrategy::FullImage => None,
            ExecutionStrategy::Tiled {
                tile_size,
                threshold_bytes,
            } => {
                if image_bytes >= *threshold_bytes {
                    Some(*tile_size)
                } else {
                    None
                }
            }
            ExecutionStrategy::Adaptive => {
                if image_bytes >= ADAPTIVE_THRESHOLD_BYTES {
                    Some(DEFAULT_TILE_SIZE)
                } else {
                    None
                }
            }
        }
    }
}

// ── Thread-local state ────────────────────────────────────────────────────────

fn init_default_strategy() -> ExecutionStrategy {
    match std::env::var("VIEW_BUFFER_STRATEGY")
        .as_deref()
        .map(str::to_lowercase)
        .as_deref()
    {
        Ok("full") | Ok("fullimage") | Ok("full_image") => ExecutionStrategy::FullImage,
        Ok("tiled") => ExecutionStrategy::Tiled {
            tile_size: DEFAULT_TILE_SIZE,
            threshold_bytes: ADAPTIVE_THRESHOLD_BYTES,
        },
        _ => ExecutionStrategy::Adaptive,
    }
}

thread_local! {
    static STRATEGY: RefCell<ExecutionStrategy> = RefCell::new(init_default_strategy());
}

/// Returns the execution strategy for the current thread.
pub fn get_execution_strategy() -> ExecutionStrategy {
    STRATEGY.with(|s| s.borrow().clone())
}

/// Sets the execution strategy for the current thread.
///
/// Affects all `ExecutionPlan`s built (via [`ViewExpr::plan`]) after this
/// call on the same thread.
pub fn set_execution_strategy(strategy: ExecutionStrategy) {
    STRATEGY.with(|s| *s.borrow_mut() = strategy);
}

/// Runs `f` with the given strategy, restoring the previous value afterward.
pub fn with_execution_strategy<T, F: FnOnce() -> T>(strategy: ExecutionStrategy, f: F) -> T {
    let prev = get_execution_strategy();
    set_execution_strategy(strategy);
    let result = f();
    set_execution_strategy(prev);
    result
}
