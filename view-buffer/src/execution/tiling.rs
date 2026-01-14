//! Tiled execution for cache-efficient processing of large images.
//!
//! This module provides opt-in tiled execution that improves cache efficiency
//! for large images by processing in fixed-size tiles (e.g., 256x256).
//!
//! # Overview
//!
//! Tiled execution sits as a transparent wrapper around existing operations:
//! - Input tiles are extracted as zero-copy views via [`ViewBuffer::slice()`]
//! - Tiles are processed sequentially in row-major order for cache locality
//! - Output is pre-allocated and tiles are copied incrementally
//! - No changes to existing operation implementations are required
//!
//! # Usage
//!
//! Enable tiled execution via environment variable:
//!
//! ```bash
//! VIEW_BUFFER_TILE_SIZE=256 ./my_program
//! ```
//!
//! Or programmatically via [`with_tile_config()`]:
//!
//! ```ignore
//! use view_buffer::{TileConfig, with_tile_config};
//!
//! let result = with_tile_config(Some(TileConfig::default()), || {
//!     // Operations here will use tiled execution for large images
//!     expr.plan().execute()
//! });
//! ```
//!
//! # Cache Efficiency
//!
//! The default tile size of 256×256 produces ~192KB tiles for RGB images,
//! which fits comfortably in L2 cache on most modern CPUs. This reduces
//! memory bandwidth pressure and improves cache hit rates.
//!
//! # Zero-Copy Semantics
//!
//! Input tiles are extracted using [`ViewBuffer::slice()`], which creates
//! views without copying data. This preserves the zero-copy paradigm:
//! - Input tile extraction: zero-copy (shares Arc storage)
//! - Strided input handling: works correctly through stride composition
//! - Output allocation: inherent to compute ops, same as non-tiled

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::core::buffer::{BufferStorage, ViewBuffer};
use crate::core::layout::Layout;

// ============================================================
// Fast-Path Atomic Flag
// ============================================================

/// Global atomic flag for fast "is tiling enabled" check.
///
/// This avoids expensive thread-local storage access on every operation
/// when tiling is disabled. The flag is kept in sync with the thread-local
/// configuration via [`set_tile_config()`].
///
/// Note: This is a "hint" that may lag behind thread-local state in multi-threaded
/// scenarios, but that's acceptable since:
/// - Most usage is single-threaded (Polars plugin)
/// - False positives just mean we do the TLS lookup (small overhead)
/// - False negatives are impossible since we set atomic BEFORE TLS on enable
///
/// Tiling is currently disabled by default due to performance issues.
static TILING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Fast check if tiling is potentially enabled.
///
/// This is a cheap atomic load that avoids thread-local storage access.
/// Returns `true` if tiling might be enabled (requires TLS check to confirm).
/// Returns `false` if tiling is definitely disabled (no TLS check needed).
#[inline(always)]
pub fn is_tiling_enabled() -> bool {
    TILING_ENABLED.load(Ordering::Relaxed)
}

/// Configuration for tiled execution.
///
/// Controls when and how tiling is applied to image operations.
///
/// # Tile Size Selection
///
/// The default tile size of 256×256 is chosen to fit in L2 cache:
/// - 256×256×3 (RGB) = 192KB
/// - 256×256×1 (grayscale) = 64KB
///
/// Both fit comfortably in typical L2 caches (256KB-1MB on Intel/AMD,
/// 4MB on Apple M-series).
#[derive(Debug, Clone, PartialEq)]
pub struct TileConfig {
    /// Tile size in pixels (both width and height).
    ///
    /// Default: 256 (192KB for RGB, fits L2 cache).
    pub tile_size: usize,

    /// Minimum image dimension to enable tiling.
    ///
    /// Images smaller than this in both dimensions skip tiling
    /// since they likely already fit in cache.
    ///
    /// Default: 512.
    pub min_image_size: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            min_image_size: 512,
        }
    }
}

impl TileConfig {
    /// Creates a new tile configuration with the specified tile size.
    ///
    /// Uses default minimum image size of 512.
    pub fn new(tile_size: usize) -> Self {
        Self {
            tile_size,
            ..Default::default()
        }
    }

    /// Creates a new tile configuration with both tile size and minimum image size.
    pub fn with_min_size(tile_size: usize, min_image_size: usize) -> Self {
        Self {
            tile_size,
            min_image_size,
        }
    }
}

/// Policy describing how an operation can be tiled.
///
/// Operations declare their tiling policy to enable the execution layer
/// to automatically apply tiled execution when appropriate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilePolicy {
    /// No dependencies between pixels - can process each pixel independently.
    ///
    /// Examples: threshold, grayscale, scale, relu, clamp, cast.
    ///
    /// These operations have halo=0 and can be tiled without overlap.
    PointWise,

    /// Needs neighboring pixels within a radius (halo region).
    ///
    /// Examples: blur (halo = 3*sigma), convolution, morphological ops.
    ///
    /// Tiles must include extra pixels around the core region to provide
    /// context for pixels at tile edges. Only the core region (excluding
    /// halo) is copied to the output.
    LocalNeighborhood {
        /// The number of pixels needed around each tile edge.
        ///
        /// For a Gaussian blur with sigma=2, halo would be 6 (3*sigma).
        halo: usize,
    },

    /// Cannot be tiled - needs access to the full image.
    ///
    /// Examples: resize (global resampling), normalize(minmax/zscore) (needs
    /// global statistics), perceptual hash, histogram.
    ///
    /// These operations are executed on the full buffer without tiling.
    Global,
}

impl TilePolicy {
    /// Returns the halo size for this policy.
    ///
    /// Returns 0 for PointWise and Global policies.
    #[inline]
    pub fn halo(&self) -> usize {
        match self {
            TilePolicy::PointWise => 0,
            TilePolicy::LocalNeighborhood { halo } => *halo,
            TilePolicy::Global => 0,
        }
    }

    /// Returns true if this operation can be tiled.
    #[inline]
    pub fn is_tileable(&self) -> bool {
        !matches!(self, TilePolicy::Global)
    }
}

// ============================================================
// Thread-Local Configuration
// ============================================================

/// Initializes the default tile configuration.
///
/// Priority order:
/// 1. Environment variable `VIEW_BUFFER_TILE_SIZE` (explicit tile size)
/// 2. Environment variable `VIEW_BUFFER_TILING=0` or `false` (disable tiling)
/// 3. Default: tiling ON with tile_size=256, min_image_size=512
///
/// Also sets the global atomic flag to match the initial configuration.
fn init_default_tile_config() -> Option<TileConfig> {
    // Check for explicit tile size first
    if let Ok(size_str) = std::env::var("VIEW_BUFFER_TILE_SIZE") {
        if let Ok(size) = size_str.parse::<usize>() {
            // Atomic flag already defaults to true, no change needed
            return Some(TileConfig::new(size));
        }
    }

    // Check if tiling is explicitly disabled
    if let Ok(val) = std::env::var("VIEW_BUFFER_TILING") {
        let lower = val.to_lowercase();
        if lower == "none"
            || lower.is_empty()
            || lower == "0"
            || lower == "false"
            || lower == "off"
            || lower == "no"
        {
            // Disable tiling - update atomic flag
            TILING_ENABLED.store(false, Ordering::Relaxed);
            return None;
        }
    }

    // Default: tiling ON with default settings
    // Atomic flag already defaults to true, no change needed
    Some(TileConfig::default())
}

thread_local! {
    /// Thread-local tile configuration.
    ///
    /// Default: Tiling enabled with tile_size=256, min_image_size=512.
    ///
    /// Can be configured via:
    /// - Environment variable `VIEW_BUFFER_TILE_SIZE=256` (set tile size)
    /// - Environment variable `VIEW_BUFFER_TILING=0` (disable tiling)
    /// - Programmatically via [`set_tile_config()`] or [`with_tile_config()`]
    static TILE_CONFIG: RefCell<Option<TileConfig>> = RefCell::new(init_default_tile_config());
}

/// Sets the global tile configuration.
///
/// This sets the thread-local tile configuration that will be used for
/// all subsequent operations on this thread. Also updates the global
/// atomic flag for fast-path checking.
///
/// # Arguments
///
/// * `config` - The tile configuration, or `None` to disable tiling.
///
/// # Example
///
/// ```ignore
/// use view_buffer::{TileConfig, set_tile_config};
///
/// // Enable tiling with custom settings
/// set_tile_config(Some(TileConfig::with_min_size(256, 1024)));
///
/// // Disable tiling entirely
/// set_tile_config(None);
/// ```
pub fn set_tile_config(config: Option<TileConfig>) {
    // Update atomic flag BEFORE TLS for enable (ensures no false negatives)
    // Update atomic flag AFTER TLS for disable (ensures fast path works)
    let is_enabled = config.is_some();
    if is_enabled {
        TILING_ENABLED.store(true, Ordering::Relaxed);
    }
    TILE_CONFIG.with(|c| {
        *c.borrow_mut() = config;
    });
    if !is_enabled {
        TILING_ENABLED.store(false, Ordering::Relaxed);
    }
}

/// Configures tiling with a minimum image size threshold.
///
/// This is a convenience function that enables tiling with the default
/// tile size (256) but allows customizing when tiling activates.
///
/// # Arguments
///
/// * `min_image_size` - Minimum dimension (height or width) for tiling to activate.
///   Pass `None` to disable tiling entirely.
///
/// # Example
///
/// ```ignore
/// use view_buffer::configure_tiling;
///
/// // Only tile images larger than 1024 pixels
/// configure_tiling(Some(1024));
///
/// // Disable tiling
/// configure_tiling(None);
/// ```
pub fn configure_tiling(min_image_size: Option<usize>) {
    let config = min_image_size.map(|size| TileConfig::with_min_size(256, size));
    set_tile_config(config);
}

/// Gets the current tile configuration (if enabled).
///
/// Returns `None` if tiling is disabled or not configured.
pub fn get_tile_config() -> Option<TileConfig> {
    TILE_CONFIG.with(|c| c.borrow().clone())
}

/// Executes a closure with a specific tile configuration.
///
/// This temporarily sets the thread-local tile configuration for the
/// duration of the closure, restoring the previous value afterward.
///
/// # Arguments
///
/// * `config` - The tile configuration to use, or `None` to disable tiling.
/// * `f` - The closure to execute with the specified configuration.
///
/// # Example
///
/// ```ignore
/// use view_buffer::{TileConfig, with_tile_config};
///
/// // Enable tiling with 128×128 tiles
/// let result = with_tile_config(Some(TileConfig::new(128)), || {
///     // Tiled execution is enabled here
///     pipeline.execute()
/// });
///
/// // Disable tiling explicitly
/// let result = with_tile_config(None, || {
///     // Tiling is disabled here
///     pipeline.execute()
/// });
/// ```
pub fn with_tile_config<T, F: FnOnce() -> T>(config: Option<TileConfig>, f: F) -> T {
    // Save current config
    let prev = TILE_CONFIG.with(|c| c.borrow().clone());

    // Set new config
    TILE_CONFIG.with(|c| {
        *c.borrow_mut() = config;
    });

    // Execute closure
    let result = f();

    // Restore previous config
    TILE_CONFIG.with(|c| {
        *c.borrow_mut() = prev;
    });

    result
}

// ============================================================
// Tiled Execution Functions
// ============================================================

/// Check if tiling should be applied based on image shape and configuration.
///
/// Returns `true` if the image is large enough to benefit from tiling.
pub fn should_tile(shape: &[usize], config: &TileConfig) -> bool {
    // Need at least 2D for image tiling
    if shape.len() < 2 {
        return false;
    }

    let h = shape[0];
    let w = shape[1];

    // Only tile if at least one dimension exceeds the minimum
    h > config.min_image_size || w > config.min_image_size
}

/// Convenience wrapper: apply tiling if enabled and beneficial.
///
/// This is the main entry point for tiled execution. It checks whether
/// tiling is enabled and appropriate, then either applies tiled execution
/// or falls through to direct execution.
///
/// # Arguments
///
/// * `input` - The input buffer to process.
/// * `halo` - The halo size needed by the operation (0 for point-wise ops).
/// * `config` - Optional tile configuration. If `None`, tiling is disabled.
/// * `op` - The operation closure to apply.
///
/// # Returns
///
/// The result of applying the operation, either tiled or directly.
///
/// # Example
///
/// ```ignore
/// let result = maybe_tiled(
///     input,
///     0, // halo for point-wise op
///     get_tile_config().as_ref(),
///     |tile| apply_threshold(tile, 128),
/// );
/// ```
pub fn maybe_tiled<F>(
    input: ViewBuffer,
    halo: usize,
    config: Option<&TileConfig>,
    op: F,
) -> ViewBuffer
where
    F: Fn(ViewBuffer) -> ViewBuffer,
{
    match config {
        Some(cfg) if should_tile(input.shape(), cfg) => {
            execute_tiled(input, halo, cfg.tile_size, op)
        }
        _ => op(input), // No tiling - pass through unchanged
    }
}

/// Execute an operation with tiled processing.
///
/// This function divides the input into tiles, applies the operation to each
/// tile, and assembles the results into the output buffer.
///
/// # Tile Processing
///
/// 1. **Input tile extraction**: Each tile is extracted as a zero-copy view
///    using `ViewBuffer::slice()`. The tile includes a halo region around
///    the core to provide context for operations like blur.
///
/// 2. **Operation application**: The operation is applied to the tile.
///    The operation sees a regular `ViewBuffer` and doesn't need to know
///    it's processing a tile.
///
/// 3. **Output assembly**: The core region of the output tile (excluding
///    halo) is copied to the pre-allocated output buffer.
///
/// # Halo Handling
///
/// For operations with local dependencies (like blur), the input tile
/// includes extra pixels (halo) around the core region:
///
/// ```text
/// ┌───────────────────┐
/// │   Halo (top)      │ ← Read from input
/// ├───────────────────┤
/// │                   │
/// │   Core Tile       │ ← Produce output
/// │                   │
/// ├───────────────────┤
/// │   Halo (bottom)   │ ← Read from input
/// └───────────────────┘
/// ```
///
/// At image edges, the halo is clamped to available pixels.
///
/// # Arguments
///
/// * `input` - The input buffer to tile.
/// * `halo` - Number of pixels needed around each tile edge.
/// * `tile_size` - Size of each tile (both width and height).
/// * `op` - The operation to apply to each tile.
///
/// # Returns
///
/// A new buffer containing the assembled output.
pub fn execute_tiled<F>(input: ViewBuffer, halo: usize, tile_size: usize, op: F) -> ViewBuffer
where
    F: Fn(ViewBuffer) -> ViewBuffer,
{
    let input_shape = input.shape();
    let input_ndim = input_shape.len();
    let (h, w) = (input_shape[0], input_shape[1]);
    let input_channels = input_shape.get(2).copied().unwrap_or(1);

    // We need to determine output characteristics by processing the first tile
    // because operations may change channels (e.g., grayscale) or dtype
    let first_tile_y: usize = 0;
    let first_tile_x: usize = 0;

    // Extract first input tile
    let in_y0 = first_tile_y.saturating_sub(halo);
    let in_y1 = (first_tile_y + tile_size + halo).min(h);
    let in_x0 = first_tile_x.saturating_sub(halo);
    let in_x1 = (first_tile_x + tile_size + halo).min(w);

    let (start, end) = if input_ndim >= 3 {
        (vec![in_y0, in_x0, 0], vec![in_y1, in_x1, input_channels])
    } else {
        (vec![in_y0, in_x0], vec![in_y1, in_x1])
    };
    let first_input_tile = input.slice(&start, &end);

    // Materialize if tile is non-contiguous (e.g., from flipped buffer)
    let first_input_tile = if first_input_tile.layout.is_contiguous() {
        first_input_tile
    } else {
        first_input_tile.to_contiguous()
    };
    let first_output_tile = op(first_input_tile);

    // Determine output characteristics from first tile
    let first_output_shape = first_output_tile.shape();
    let output_ndim = first_output_shape.len();
    let output_channels = first_output_shape.get(2).copied().unwrap_or(1);
    let output_dtype = first_output_tile.dtype();
    let output_dtype_size = output_dtype.size_of();

    // Pre-allocate output buffer with correct characteristics
    let output_bytes = h * w * output_channels * output_dtype_size;
    let mut output_data: Vec<u8> = vec![0u8; output_bytes];

    // Copy first tile's core region to output
    let core_h_first = tile_size.min(h);
    let core_w_first = tile_size.min(w);
    copy_tile_to_output(
        &first_output_tile,
        0, // core_y_offset (first tile has no halo at top-left)
        0, // core_x_offset
        &mut output_data,
        0, // dst_y
        0, // dst_x
        core_h_first,
        core_w_first,
        w,
        output_channels,
        output_dtype_size,
    );

    // Process remaining tiles in row-major order (cache-friendly)
    for tile_y in (0..h).step_by(tile_size) {
        for tile_x in (0..w).step_by(tile_size) {
            // Skip the first tile (already processed)
            if tile_y == 0 && tile_x == 0 {
                continue;
            }

            // === INPUT: ZERO-COPY TILE EXTRACTION ===
            let in_y0 = tile_y.saturating_sub(halo);
            let in_y1 = (tile_y + tile_size + halo).min(h);
            let in_x0 = tile_x.saturating_sub(halo);
            let in_x1 = (tile_x + tile_size + halo).min(w);

            let (start, end) = if input_ndim >= 3 {
                (vec![in_y0, in_x0, 0], vec![in_y1, in_x1, input_channels])
            } else {
                (vec![in_y0, in_x0], vec![in_y1, in_x1])
            };
            let input_tile = input.slice(&start, &end);

            // Materialize if tile is non-contiguous (e.g., from flipped buffer)
            // This ensures operations that require contiguous data work correctly
            let input_tile = if input_tile.layout.is_contiguous() {
                input_tile
            } else {
                input_tile.to_contiguous()
            };

            // === COMPUTE: APPLY OPERATION ===
            let output_tile = op(input_tile);

            // === OUTPUT: COPY CORE REGION TO FINAL BUFFER ===
            let out_y0 = tile_y;
            let out_y1 = (tile_y + tile_size).min(h);
            let out_x0 = tile_x;
            let out_x1 = (tile_x + tile_size).min(w);
            let core_h = out_y1 - out_y0;
            let core_w = out_x1 - out_x0;

            // Offset into the output_tile to skip halo
            let core_y_offset = tile_y - in_y0;
            let core_x_offset = tile_x - in_x0;

            copy_tile_to_output(
                &output_tile,
                core_y_offset,
                core_x_offset,
                &mut output_data,
                out_y0,
                out_x0,
                core_h,
                core_w,
                w,
                output_channels,
                output_dtype_size,
            );
        }
    }

    // Reconstruct ViewBuffer from output data
    let final_output_shape = if output_ndim >= 3 {
        vec![h, w, output_channels]
    } else {
        vec![h, w]
    };

    ViewBuffer {
        data: BufferStorage::Rust(Arc::new(output_data)),
        layout: Layout::new_contiguous(final_output_shape, output_dtype),
    }
}

/// Copy a tile's core region to the output buffer.
///
/// This function handles the strided copy from the output tile (which may
/// have halo regions) to the final output buffer.
#[allow(clippy::too_many_arguments)]
#[inline]
fn copy_tile_to_output(
    tile: &ViewBuffer,
    src_y_offset: usize,
    src_x_offset: usize,
    dst: &mut [u8],
    dst_y: usize,
    dst_x: usize,
    height: usize,
    width: usize,
    dst_width: usize,
    channels: usize,
    dtype_size: usize,
) {
    let tile_shape = tile.shape();
    let tile_strides = tile.strides_bytes();
    let tile_ptr = unsafe { tile.as_ptr::<u8>() };

    // Bytes per row in destination (contiguous)
    let dst_row_bytes = dst_width * channels * dtype_size;

    // Bytes to copy per row
    let copy_bytes_per_row = width * channels * dtype_size;

    for row in 0..height {
        let src_y = src_y_offset + row;
        let src_x = src_x_offset;

        // Source offset in tile (handle strides)
        let src_offset = if tile_shape.len() >= 3 {
            // HWC layout
            (src_y as isize * tile_strides[0]) + (src_x as isize * tile_strides[1])
        } else {
            // HW layout (treat as HW1)
            (src_y as isize * tile_strides[0]) + (src_x as isize * tile_strides[1])
        };

        // Destination offset (contiguous layout)
        let dst_offset = (dst_y + row) * dst_row_bytes + dst_x * channels * dtype_size;

        // Copy the row
        unsafe {
            let src = tile_ptr.offset(src_offset);
            let dst_ptr = dst.as_mut_ptr().add(dst_offset);
            std::ptr::copy_nonoverlapping(src, dst_ptr, copy_bytes_per_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_default() {
        let config = TileConfig::default();
        assert_eq!(config.tile_size, 256);
        assert_eq!(config.min_image_size, 512);
    }

    #[test]
    fn test_tile_config_new() {
        let config = TileConfig::new(128);
        assert_eq!(config.tile_size, 128);
        assert_eq!(config.min_image_size, 512);
    }

    #[test]
    fn test_tile_policy_halo() {
        assert_eq!(TilePolicy::PointWise.halo(), 0);
        assert_eq!(TilePolicy::LocalNeighborhood { halo: 6 }.halo(), 6);
        assert_eq!(TilePolicy::Global.halo(), 0);
    }

    #[test]
    fn test_tile_policy_is_tileable() {
        assert!(TilePolicy::PointWise.is_tileable());
        assert!(TilePolicy::LocalNeighborhood { halo: 6 }.is_tileable());
        assert!(!TilePolicy::Global.is_tileable());
    }

    #[test]
    fn test_should_tile_small_image() {
        let config = TileConfig::default();
        // 100x100 is below min_image_size
        assert!(!should_tile(&[100, 100, 3], &config));
    }

    #[test]
    fn test_should_tile_large_image() {
        let config = TileConfig::default();
        // 1000x1000 exceeds min_image_size
        assert!(should_tile(&[1000, 1000, 3], &config));
    }

    #[test]
    fn test_should_tile_one_large_dim() {
        let config = TileConfig::default();
        // 100x1000 - width exceeds min_image_size
        assert!(should_tile(&[100, 1000, 3], &config));
    }

    #[test]
    fn test_with_tile_config() {
        // Initially may or may not be set (depends on env)
        let initial = get_tile_config();

        // Enable tiling
        let result = with_tile_config(Some(TileConfig::new(128)), || {
            let config = get_tile_config();
            assert!(config.is_some());
            assert_eq!(config.unwrap().tile_size, 128);
            42
        });
        assert_eq!(result, 42);

        // Should be restored
        assert_eq!(get_tile_config(), initial);

        // Disable tiling
        let result = with_tile_config(None, || {
            assert!(get_tile_config().is_none());
            "done"
        });
        assert_eq!(result, "done");

        // Should still be restored
        assert_eq!(get_tile_config(), initial);
    }
}
